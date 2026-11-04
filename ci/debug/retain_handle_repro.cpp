// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// TEMPORARY DEBUGGING TOOL (do not merge).
//
// Standalone reproducer for the hang seen in the ucxx-async/cupy/thread-progress benchmark on
// V100 + CUDA 12.2.2 + driver 580: a UCX progress thread spins for minutes inside
//
//   cuMemRetainAllocationHandle <- uct_cuda_copy_detect_vmm <- ucp_mem_type_unpack
//
// This program uses only the CUDA *driver* API (no UCX, no ucxx, no cupy) to mimic that
// situation:
//
//   * an "allocator" thread churns allocations: stream-ordered ones from the device's default
//     memory pool (cuMemAllocAsync / cuMemFreeAsync / cuMemPoolTrimTo), like the CCCL buffers
//     ucxx uses for received messages, and explicit VMM ones (cuMemCreate / cuMemMap), which
//     are guaranteed to make cuMemRetainAllocationHandle succeed. On some driver versions pool
//     memory is not retainable at all, so the VMM slots keep the "success" path covered.
//   * N "progress" threads run, on addresses inside those allocations (and on plain host
//     memory), the same sequence of driver calls UCX issues for every address when
//     UCX_MEMTYPE_CACHE=n: cuMemRetainAllocationHandle, cuMemGetAllocationPropertiesFromHandle,
//     cuMemRelease, cuPointerGetAttributes.
//   * a watchdog reports and exits with status 124 if a single driver call stays in flight for
//     longer than --stuck-sec (one call that never returns), or if no thread completes any call
//     for that long (livelock).
//
// A hang here shows that the driver can hang in this pattern on its own. A clean run proves
// nothing about the benchmark.
//
// Build: see ci/debug/build_and_run_retain_repro.sh (g++ + cuda.h + libcuda.so.1 only).

#include <cuda.h>

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

using Clock = std::chrono::steady_clock;

static int64_t now_ns()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
    .count();
}

// Per-thread record of the driver call currently in flight.
struct Watch {
  std::string name;
  std::atomic<const char*> call{nullptr};  // nullptr when not inside a driver call
  std::atomic<int64_t> since_ns{0};
  std::atomic<uint64_t> completed{0};
};

static std::vector<std::unique_ptr<Watch>> g_watches;
static std::atomic<bool> g_stop{false};
static bool g_inject_hang = false;

#define TRACKED(w, label, expr)   \
  [&]() {                         \
    (w).since_ns.store(now_ns()); \
    (w).call.store(label);        \
    auto r_ = (expr);             \
    (w).call.store(nullptr);      \
    (w).completed.fetch_add(1);   \
    return r_;                    \
  }()

#define CU_OK(w, label, expr)                                                                 \
  do {                                                                                        \
    CUresult e_ = TRACKED(w, label, (expr));                                                  \
    if (e_ != CUDA_SUCCESS) {                                                                 \
      const char* s_ = "?";                                                                   \
      cuGetErrorString(e_, &s_);                                                              \
      fprintf(stderr, "FATAL %s:%d %s failed: %d (%s)\n", __FILE__, __LINE__, label, e_, s_); \
      _exit(2);                                                                               \
    }                                                                                         \
  } while (0)

struct Config {
  int device         = 0;
  int duration_sec   = 30;
  int stuck_sec      = 20;
  int retainers      = 2;
  int slots          = 8;
  size_t slot_bytes  = 8u << 20;  // 8 MiB, like the benchmark's buffers
  size_t query_bytes = 8176;      // size of the fragment that was being unpacked when it hung
  int trim_every     = 4;         // cuMemPoolTrimTo every N allocator sweeps
  int host_every     = 4;         // query a host address every N queries
  std::string alloc  = "mixed";   // pool | vmm | mixed
  int linger_sec     = 0;         // after reporting a hang, stay alive this long (for stack dumps)
};

struct Alloc {
  CUdeviceptr ptr                     = 0;
  bool vmm                            = false;
  CUmemGenericAllocationHandle handle = 0;
  size_t size                         = 0;
};

struct Slot {
  std::shared_mutex m;
  Alloc a;
};

static void usage(const char* argv0)
{
  fprintf(stderr,
          "usage: %s [--duration SEC] [--stuck-sec SEC] [--retainers N] [--slots N]\n"
          "          [--slot-bytes B] [--device N] [--alloc pool|vmm|mixed] [--linger-sec N] "
          "[--inject-hang]\n",
          argv0);
}

static size_t g_granularity = 0;

static Alloc vmm_alloc(Watch* w, int device, size_t bytes)
{
  CUmemAllocationProp prop = {};
  prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type       = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id         = device;
  if (g_granularity == 0) {
    CU_OK(*w,
          "cuMemGetAllocationGranularity",
          cuMemGetAllocationGranularity(&g_granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  }
  Alloc a;
  a.vmm  = true;
  a.size = (bytes + g_granularity - 1) / g_granularity * g_granularity;
  CU_OK(*w, "cuMemCreate", cuMemCreate(&a.handle, a.size, &prop, 0));
  CU_OK(*w, "cuMemAddressReserve", cuMemAddressReserve(&a.ptr, a.size, 0, 0, 0));
  CU_OK(*w, "cuMemMap", cuMemMap(a.ptr, a.size, 0, a.handle, 0));
  CUmemAccessDesc access = {};
  access.location        = prop.location;
  access.flags           = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CU_OK(*w, "cuMemSetAccess", cuMemSetAccess(a.ptr, a.size, &access, 1));
  return a;
}

static void free_alloc(Watch* w, const Alloc& a, CUstream stream)
{
  if (a.ptr == 0) return;
  if (a.vmm) {
    CU_OK(*w, "cuMemUnmap", cuMemUnmap(a.ptr, a.size));
    CU_OK(*w, "cuMemRelease(vmm)", cuMemRelease(a.handle));
    CU_OK(*w, "cuMemAddressFree", cuMemAddressFree(a.ptr, a.size));
  } else {
    CU_OK(*w, "cuMemFreeAsync", cuMemFreeAsync(a.ptr, stream));
  }
}

static void allocator_thread(Config cfg,
                             CUcontext ctx,
                             std::vector<std::unique_ptr<Slot>>* slots,
                             Watch* w)
{
  cuCtxSetCurrent(ctx);
  CUstream stream;
  CU_OK(*w, "cuStreamCreate", cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  CUmemoryPool pool;
  CU_OK(*w, "cuDeviceGetDefaultMemPool", cuDeviceGetDefaultMemPool(&pool, cfg.device));

  uint64_t sweep = 0;
  while (!g_stop.load()) {
    for (size_t i = 0; i < slots->size(); ++i) {
      if (g_stop.load()) break;
      auto& slot   = (*slots)[i];
      bool use_vmm = cfg.alloc == "vmm" || (cfg.alloc == "mixed" && i % 2 == 1);
      Alloc fresh;
      if (use_vmm) {
        fresh = vmm_alloc(w, cfg.device, cfg.slot_bytes);
      } else {
        fresh.size = cfg.slot_bytes;
        CU_OK(*w, "cuMemAllocAsync", cuMemAllocAsync(&fresh.ptr, fresh.size, stream));
        CU_OK(*w, "cuMemsetD8Async", cuMemsetD8Async(fresh.ptr, 0x5a, fresh.size, stream));
        CU_OK(*w, "cuStreamSynchronize", cuStreamSynchronize(stream));
      }
      Alloc old;
      {
        std::unique_lock<std::shared_mutex> lock(slot->m);  // wait for in-flight queries
        old     = slot->a;
        slot->a = fresh;
      }
      free_alloc(w, old, stream);
    }
    if (++sweep % cfg.trim_every == 0) {
      CU_OK(*w, "cuStreamSynchronize", cuStreamSynchronize(stream));
      CU_OK(*w, "cuMemPoolTrimTo", cuMemPoolTrimTo(pool, 0));
    }
  }
  // Drain.
  CU_OK(*w, "cuStreamSynchronize", cuStreamSynchronize(stream));
  for (auto& slot : *slots) {
    std::unique_lock<std::shared_mutex> lock(slot->m);
    free_alloc(w, slot->a, stream);
    slot->a = Alloc();
  }
  CU_OK(*w, "cuStreamSynchronize", cuStreamSynchronize(stream));
}

// Mimics what uct_cuda_copy_md_query_attributes() does for one address.
static void query_address(Watch* w, const void* addr, uint64_t* n_vmm, uint64_t* n_not_vmm)
{
  CUmemGenericAllocationHandle handle;
  CUresult r = TRACKED(*w,
                       "cuMemRetainAllocationHandle",
                       cuMemRetainAllocationHandle(&handle, const_cast<void*>(addr)));
  if (r == CUDA_SUCCESS) {
    ++*n_vmm;
    CUmemAllocationProp prop;
    TRACKED(*w,
            "cuMemGetAllocationPropertiesFromHandle",
            cuMemGetAllocationPropertiesFromHandle(&prop, handle));
    TRACKED(*w, "cuMemRelease", cuMemRelease(handle));
  } else {
    ++*n_not_vmm;
  }
  // UCX's non-VMM path.
  CUmemorytype mem_type            = CU_MEMORYTYPE_HOST;
  uint32_t is_managed              = 0;
  int ordinal                      = -1;
  CUcontext mem_ctx                = nullptr;
  CUpointer_attribute attr_type[4] = {CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                      CU_POINTER_ATTRIBUTE_IS_MANAGED,
                                      CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                      CU_POINTER_ATTRIBUTE_CONTEXT};
  void* attr_data[4]               = {&mem_type, &is_managed, &ordinal, &mem_ctx};
  TRACKED(*w,
          "cuPointerGetAttributes",
          cuPointerGetAttributes(4, attr_type, attr_data, reinterpret_cast<CUdeviceptr>(addr)));
}

static void retainer_thread(
  Config cfg, CUcontext ctx, std::vector<std::unique_ptr<Slot>>* slots, Watch* w, int id)
{
  cuCtxSetCurrent(ctx);
  std::vector<char> host_buf(cfg.slot_bytes);
  uint64_t n = 0, n_vmm = 0, n_not_vmm = 0;
  uint64_t rng = 0x9e3779b97f4a7c15ull * (id + 1);
  while (!g_stop.load()) {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    auto& slot = (*slots)[rng % slots->size()];
    if (++n % cfg.host_every == 0) {
      size_t off = rng % (host_buf.size() - cfg.query_bytes);
      query_address(w, host_buf.data() + off, &n_vmm, &n_not_vmm);
      continue;
    }
    std::shared_lock<std::shared_mutex> lock(slot->m);
    if (slot->a.ptr == 0) continue;
    size_t off = (rng >> 8) % (cfg.slot_bytes - cfg.query_bytes);
    if (g_inject_hang && n >= 1000 && id == 0) {
      // Test hook for the watchdog: pretend a driver call never returns.
      w->since_ns.store(now_ns());
      w->call.store("INJECTED cuMemRetainAllocationHandle");
      while (!g_stop.load())
        std::this_thread::sleep_for(std::chrono::seconds(1));
      return;
    }
    query_address(w, reinterpret_cast<void*>(slot->a.ptr + off), &n_vmm, &n_not_vmm);
  }
  fprintf(stderr,
          "retainer %d: %" PRIu64 " queries, %" PRIu64 " on VMM-backed memory, %" PRIu64 " not\n",
          id,
          n,
          n_vmm,
          n_not_vmm);
}

int main(int argc, char** argv)
{
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next     = [&]() -> const char* {
      if (i + 1 >= argc) {
        usage(argv[0]);
        exit(2);
      }
      return argv[++i];
    };
    if (a == "--duration") {
      cfg.duration_sec = atoi(next());
    } else if (a == "--stuck-sec") {
      cfg.stuck_sec = atoi(next());
    } else if (a == "--retainers") {
      cfg.retainers = atoi(next());
    } else if (a == "--slots") {
      cfg.slots = atoi(next());
    } else if (a == "--slot-bytes") {
      cfg.slot_bytes = strtoull(next(), nullptr, 0);
    } else if (a == "--device") {
      cfg.device = atoi(next());
    } else if (a == "--alloc") {
      cfg.alloc = next();
    } else if (a == "--linger-sec") {
      cfg.linger_sec = atoi(next());
    } else if (a == "--inject-hang") {
      g_inject_hang = true;
    } else {
      usage(argv[0]);
      return 2;
    }
  }

  if (cuInit(0) != CUDA_SUCCESS) {
    fprintf(stderr, "cuInit failed\n");
    return 2;
  }
  int driver_version = 0;
  cuDriverGetVersion(&driver_version);
  CUdevice dev;
  CUcontext ctx;
  if (cuDeviceGet(&dev, cfg.device) != CUDA_SUCCESS ||
      cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS) {
    fprintf(stderr, "cannot set up device %d\n", cfg.device);
    return 2;
  }
  char name[128] = {0};
  cuDeviceGetName(name, sizeof(name), dev);
  printf("device %d: %s, driver API version %d, header CUDA_VERSION %d\n",
         cfg.device,
         name,
         driver_version,
         CUDA_VERSION);
  printf(
    "duration %ds, stuck threshold %ds, %d retainer thread(s), %d slots x %zu bytes, alloc=%s\n",
    cfg.duration_sec,
    cfg.stuck_sec,
    cfg.retainers,
    cfg.slots,
    cfg.slot_bytes,
    cfg.alloc.c_str());
  fflush(stdout);

  std::vector<std::unique_ptr<Slot>> slots;
  for (int i = 0; i < cfg.slots; ++i)
    slots.emplace_back(new Slot);

  g_watches.emplace_back(new Watch);
  g_watches.back()->name = "allocator";
  for (int i = 0; i < cfg.retainers; ++i) {
    g_watches.emplace_back(new Watch);
    g_watches.back()->name = "retainer" + std::to_string(i);
  }

  std::vector<std::thread> threads;
  threads.emplace_back(allocator_thread, cfg, ctx, &slots, g_watches[0].get());
  for (int i = 0; i < cfg.retainers; ++i)
    threads.emplace_back(retainer_thread, cfg, ctx, &slots, g_watches[1 + i].get(), i);

  // Watchdog (main thread).
  const int64_t start      = now_ns();
  int64_t last_progress_ns = start;
  uint64_t last_total      = 0;
  int exit_code            = 0;
  while (now_ns() - start < static_cast<int64_t>(cfg.duration_sec) * 1000000000LL) {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    uint64_t total = 0;
    for (auto& w : g_watches) {
      total += w->completed.load();
      const char* call = w->call.load();
      if (call != nullptr) {
        double secs = (now_ns() - w->since_ns.load()) / 1e9;
        if (secs > cfg.stuck_sec) {
          printf("HANG: thread '%s' has been inside driver call %s for %.1fs (%" PRIu64
                 " calls completed by it)\n",
                 w->name.c_str(),
                 call,
                 secs,
                 w->completed.load());
          exit_code = 124;
        }
      }
    }
    if (total != last_total) {
      last_total       = total;
      last_progress_ns = now_ns();
    } else if ((now_ns() - last_progress_ns) / 1e9 > cfg.stuck_sec) {
      printf("HANG: no driver call completed on any thread for more than %ds (livelock)\n",
             cfg.stuck_sec);
      exit_code = 124;
    }
    if (exit_code != 0) {
      printf("per-thread state at detection time:\n");
      for (auto& w : g_watches) {
        const char* call = w->call.load();
        printf("  %-10s completed=%" PRIu64 " in_call=%s\n",
               w->name.c_str(),
               w->completed.load(),
               call ? call : "(none)");
      }
      // Threads may be stuck in the driver and cannot be joined, so never clean up. Optionally stay
      // alive so that the timeout wrapper can attach gdb and dump the stuck threads' stacks.
      if (cfg.linger_sec > 0) {
        printf("lingering for up to %ds so that stacks can be captured\n", cfg.linger_sec);
        fflush(stdout);
        sleep(cfg.linger_sec);
      }
      fflush(stdout);
      _exit(exit_code);
    }
  }

  g_stop.store(true);
  for (auto& t : threads)
    t.join();
  uint64_t total = 0;
  for (auto& w : g_watches)
    total += w->completed.load();
  printf("OK: no hang in %ds, %" PRIu64 " driver calls completed\n", cfg.duration_sec, total);
  cuDevicePrimaryCtxRelease(dev);
  return 0;
}
