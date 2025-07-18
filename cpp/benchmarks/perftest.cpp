/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <unistd.h>  // for getopt, optarg

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstring>
#include <future>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// CUDA includes (conditional)
#ifdef UCXX_BENCHMARKS_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#include <ucxx/api.h>
#include <ucxx/utils/sockaddr.h>
#include <ucxx/utils/ucx.h>

#define UCXX_EXIT_ON_ERROR(operation, context)                                                  \
  ([&]() {                                                                                      \
    try {                                                                                       \
      return operation;                                                                         \
    } catch (const ucxx::Error& e) {                                                            \
      std::cerr << "UCXX error in " << context << " at " << __FILE__ << ":" << __LINE__ << ": " \
                << e.what() << std::endl;                                                       \
      std::exit(-1);                                                                            \
    } catch (const std::exception& e) {                                                         \
      std::cerr << "Unexpected error in " << context << " at " << __FILE__ << ":" << __LINE__   \
                << ": " << e.what() << std::endl;                                               \
      std::exit(-1);                                                                            \
    }                                                                                           \
  })()

enum class ProgressMode {
  Polling,
  Blocking,
  Wait,
  ThreadPolling,
  ThreadBlocking,
};

enum class MemoryType {
  Host,
#ifdef UCXX_BENCHMARKS_ENABLE_CUDA
  Cuda,
  CudaManaged,
  CudaAsync,
#endif
};

enum transfer_type_t { SEND, RECV };

// CUDA memory buffer structure (conditional)
#ifdef UCXX_BENCHMARKS_ENABLE_CUDA
struct CudaBuffer {
  void* ptr{nullptr};
  size_t size{0};

  CudaBuffer() = default;
  explicit CudaBuffer(size_t buffer_size) : size(buffer_size)
  {
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
      throw std::runtime_error("Failed to allocate CUDA memory: " +
                               std::string(cudaGetErrorString(err)));
    }
  }

  ~CudaBuffer()
  {
    if (ptr) { cudaFree(ptr); }
  }

  CudaBuffer(const CudaBuffer&)            = delete;
  CudaBuffer& operator=(const CudaBuffer&) = delete;
  CudaBuffer(CudaBuffer&& other) noexcept : ptr(other.ptr), size(other.size)
  {
    other.ptr  = nullptr;
    other.size = 0;
  }
  CudaBuffer& operator=(CudaBuffer&& other) noexcept
  {
    if (this != &other) {
      if (ptr) cudaFree(ptr);
      ptr        = other.ptr;
      size       = other.size;
      other.ptr  = nullptr;
      other.size = 0;
    }
    return *this;
  }
};

// CUDA managed memory buffer structure
struct CudaManagedBuffer {
  void* ptr{nullptr};
  size_t size{0};

  CudaManagedBuffer() = default;
  explicit CudaManagedBuffer(size_t buffer_size) : size(buffer_size)
  {
    cudaError_t err = cudaMallocManaged(&ptr, size);
    if (err != cudaSuccess) {
      throw std::runtime_error("Failed to allocate CUDA managed memory: " +
                               std::string(cudaGetErrorString(err)));
    }
  }

  ~CudaManagedBuffer()
  {
    if (ptr) { cudaFree(ptr); }
  }

  CudaManagedBuffer(const CudaManagedBuffer&)            = delete;
  CudaManagedBuffer& operator=(const CudaManagedBuffer&) = delete;
  CudaManagedBuffer(CudaManagedBuffer&& other) noexcept : ptr(other.ptr), size(other.size)
  {
    other.ptr  = nullptr;
    other.size = 0;
  }
  CudaManagedBuffer& operator=(CudaManagedBuffer&& other) noexcept
  {
    if (this != &other) {
      if (ptr) cudaFree(ptr);
      ptr        = other.ptr;
      size       = other.size;
      other.ptr  = nullptr;
      other.size = 0;
    }
    return *this;
  }
};

// CUDA async memory buffer structure
struct CudaAsyncBuffer {
  void* ptr{nullptr};
  size_t size{0};
  cudaStream_t stream{nullptr};

  CudaAsyncBuffer() = default;
  explicit CudaAsyncBuffer(size_t buffer_size) : size(buffer_size)
  {
    cudaError_t err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
      throw std::runtime_error("Failed to create CUDA stream: " +
                               std::string(cudaGetErrorString(err)));
    }

    err = cudaMallocAsync(&ptr, size, stream);
    if (err != cudaSuccess) {
      cudaStreamDestroy(stream);
      throw std::runtime_error("Failed to allocate CUDA async memory: " +
                               std::string(cudaGetErrorString(err)));
    }
  }

  ~CudaAsyncBuffer()
  {
    if (ptr) { cudaFreeAsync(ptr, stream); }
    if (stream) { cudaStreamDestroy(stream); }
  }

  CudaAsyncBuffer(const CudaAsyncBuffer&)            = delete;
  CudaAsyncBuffer& operator=(const CudaAsyncBuffer&) = delete;
  CudaAsyncBuffer(CudaAsyncBuffer&& other) noexcept
    : ptr(other.ptr), size(other.size), stream(other.stream)
  {
    other.ptr    = nullptr;
    other.size   = 0;
    other.stream = nullptr;
  }
  CudaAsyncBuffer& operator=(CudaAsyncBuffer&& other) noexcept
  {
    if (this != &other) {
      if (ptr) cudaFreeAsync(ptr, stream);
      if (stream) cudaStreamDestroy(stream);
      ptr          = other.ptr;
      size         = other.size;
      stream       = other.stream;
      other.ptr    = nullptr;
      other.size   = 0;
      other.stream = nullptr;
    }
    return *this;
  }
};
#endif

typedef std::unordered_map<transfer_type_t, std::vector<char>> BufferMap;
#ifdef UCXX_BENCHMARKS_ENABLE_CUDA
typedef std::unordered_map<transfer_type_t, CudaBuffer> CudaBufferMap;
typedef std::unordered_map<transfer_type_t, CudaManagedBuffer> CudaManagedBufferMap;
typedef std::unordered_map<transfer_type_t, CudaAsyncBuffer> CudaAsyncBufferMap;
#endif
typedef std::unordered_map<transfer_type_t, ucxx::Tag> TagMap;

typedef std::shared_ptr<BufferMap> BufferMapPtr;
#ifdef UCXX_BENCHMARKS_ENABLE_CUDA
typedef std::shared_ptr<CudaBufferMap> CudaBufferMapPtr;
typedef std::shared_ptr<CudaManagedBufferMap> CudaManagedBufferMapPtr;
typedef std::shared_ptr<CudaAsyncBufferMap> CudaAsyncBufferMapPtr;
#endif
typedef std::shared_ptr<TagMap> TagMapPtr;

struct app_context_t {
  ProgressMode progress_mode   = ProgressMode::Blocking;
  const char* server_addr      = NULL;
  uint16_t listener_port       = 12345;
  size_t message_size          = 8;
  size_t n_iter                = 100;
  size_t warmup_iter           = 3;
  bool endpoint_error_handling = false;
  bool reuse_alloc             = false;
  bool verify_results          = false;
  MemoryType memory_type       = MemoryType::Host;
};

class ListenerContext {
 private:
  std::shared_ptr<ucxx::Worker> _worker{nullptr};
  std::shared_ptr<ucxx::Endpoint> _endpoint{nullptr};
  std::shared_ptr<ucxx::Listener> _listener{nullptr};
  std::atomic<bool> _isAvailable{true};
  bool _endpointErrorHandling{false};

 public:
  ListenerContext(std::shared_ptr<ucxx::Worker> worker, bool endpointErrorHandling)
    : _worker{worker}, _endpointErrorHandling(endpointErrorHandling)
  {
  }

  ~ListenerContext() { releaseEndpoint(); }

  void setListener(std::shared_ptr<ucxx::Listener> listener) { _listener = listener; }

  std::shared_ptr<ucxx::Listener> getListener() { return _listener; }

  std::shared_ptr<ucxx::Endpoint> getEndpoint() { return _endpoint; }

  bool isAvailable() const { return _isAvailable; }

  void createEndpointFromConnRequest(ucp_conn_request_h conn_request)
  {
    if (!isAvailable()) throw std::runtime_error("Listener context already has an endpoint");

    _endpoint    = _listener->createEndpointFromConnRequest(conn_request, _endpointErrorHandling);
    _isAvailable = false;
  }

  void releaseEndpoint()
  {
    _endpoint.reset();
    _isAvailable = true;
  }
};

static void listener_cb(ucp_conn_request_h conn_request, void* arg)
{
  char ip_str[INET6_ADDRSTRLEN];
  char port_str[INET6_ADDRSTRLEN];
  ucp_conn_request_attr_t attr{};
  ListenerContext* listener_ctx = reinterpret_cast<ListenerContext*>(arg);

  attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR;
  ucxx::utils::ucsErrorThrow(ucp_conn_request_query(conn_request, &attr));
  ucxx::utils::sockaddr_get_ip_port_str(&attr.client_address, ip_str, port_str, INET6_ADDRSTRLEN);
  std::cout << "Server received a connection request from client at address " << ip_str << ":"
            << port_str << std::endl;

  if (listener_ctx->isAvailable()) {
    listener_ctx->createEndpointFromConnRequest(conn_request);
  } else {
    // The server is already handling a connection request from a client,
    // reject this new one
    std::cout << "Rejecting a connection request from " << ip_str << ":" << port_str << "."
              << std::endl
              << "Only one client at a time is supported." << std::endl;
    ucxx::utils::ucsErrorThrow(
      ucp_listener_reject(listener_ctx->getListener()->getHandle(), conn_request));
  }
}

static void printUsage()
{
  std::cerr << "Basic performance test" << std::endl;
  std::cerr << std::endl;
  std::cerr << "Usage: ucxx_perftest [server-hostname] [options]" << std::endl;
  std::cerr << std::endl;
  std::cerr << "Parameters are:" << std::endl;
  std::cerr << "  -P          progress mode to use, valid values are: 'polling', 'blocking',"
            << std::endl
            << "              'thread-polling' and 'thread-blocking' (default: 'blocking')"
            << std::endl;
  std::cerr << "  -t          use thread progress mode (disabled)" << std::endl;
  std::cerr << "  -e          create endpoints with error handling support (disabled)" << std::endl;
  std::cerr << "  -p <port>   port number to listen at (12345)" << std::endl;
  std::cerr << "  -s <bytes>  message size (8)" << std::endl;
  std::cerr << "  -n <int>    number of iterations to run (100)" << std::endl;
  std::cerr << "  -r          reuse memory allocation (disabled)" << std::endl;
  std::cerr << "  -v          verify results (disabled)" << std::endl;
  std::cerr << "  -w <int>    number of warmup iterations to run (3)" << std::endl;
#ifdef UCXX_BENCHMARKS_ENABLE_CUDA
  std::cerr << "  -m <type>   memory type to use, valid values are: 'host' (default), 'cuda', "
            << std::endl
            << "              'cuda-managed', and 'cuda-async'" << std::endl;
#else
  std::cerr << "  -m <type>   memory type to use, valid values are: 'host' (default)" << std::endl;
#endif
  std::cerr << "  -h          print this help" << std::endl;
  std::cerr << std::endl;
}

ucs_status_t parseCommand(app_context_t* app_context, int argc, char* const argv[])
{
  optind = 1;
  int c;
#ifdef UCXX_BENCHMARKS_ENABLE_CUDA
  while ((c = getopt(argc, argv, "P:p:s:w:n:ervm:h")) != -1) {
#else
  while ((c = getopt(argc, argv, "P:p:s:w:n:ervh")) != -1) {
#endif
    switch (c) {
      case 'P':
        if (strcmp(optarg, "blocking") == 0) {
          app_context->progress_mode = ProgressMode::Blocking;
          break;
        } else if (strcmp(optarg, "polling") == 0) {
          app_context->progress_mode = ProgressMode::Polling;
          break;
        } else if (strcmp(optarg, "thread-blocking") == 0) {
          app_context->progress_mode = ProgressMode::ThreadBlocking;
          break;
        } else if (strcmp(optarg, "thread-polling") == 0) {
          app_context->progress_mode = ProgressMode::ThreadPolling;
          break;
        } else if (strcmp(optarg, "wait") == 0) {
          app_context->progress_mode = ProgressMode::Wait;
          break;
        } else {
          std::cerr << "Invalid progress mode: " << optarg << std::endl;
          return UCS_ERR_INVALID_PARAM;
        }
      case 'p':
        app_context->listener_port = atoi(optarg);
        if (app_context->listener_port <= 0) {
          std::cerr << "Wrong listener port: " << app_context->listener_port << std::endl;
          return UCS_ERR_INVALID_PARAM;
        }
        break;
      case 's':
        app_context->message_size = atoi(optarg);
        if (app_context->message_size <= 0) {
          std::cerr << "Wrong message size: " << app_context->message_size << std::endl;
          return UCS_ERR_INVALID_PARAM;
        }
        break;
      case 'w':
        app_context->warmup_iter = atoi(optarg);
        if (app_context->warmup_iter <= 0) {
          std::cerr << "Wrong number of warmup iterations: " << app_context->warmup_iter
                    << std::endl;
          return UCS_ERR_INVALID_PARAM;
        }
        break;
      case 'n':
        app_context->n_iter = atoi(optarg);
        if (app_context->n_iter <= 0) {
          std::cerr << "Wrong number of iterations: " << app_context->n_iter << std::endl;
          return UCS_ERR_INVALID_PARAM;
        }
        break;
      case 'e': app_context->endpoint_error_handling = true; break;
      case 'r': app_context->reuse_alloc = true; break;
      case 'v': app_context->verify_results = true; break;
      case 'm':
        if (strcmp(optarg, "host") == 0) {
          app_context->memory_type = MemoryType::Host;
          break;
#ifdef UCXX_BENCHMARKS_ENABLE_CUDA
        } else if (strcmp(optarg, "cuda") == 0) {
          app_context->memory_type = MemoryType::Cuda;
          break;
        } else if (strcmp(optarg, "cuda-managed") == 0) {
          app_context->memory_type = MemoryType::CudaManaged;
          break;
        } else if (strcmp(optarg, "cuda-async") == 0) {
          app_context->memory_type = MemoryType::CudaAsync;
          break;
#endif
        } else {
          std::cerr << "Invalid memory type: " << optarg << std::endl;
          return UCS_ERR_INVALID_PARAM;
        }
      case 'h':
      default: printUsage(); return UCS_ERR_INVALID_PARAM;
    }
  }

  if (optind < argc) { app_context->server_addr = argv[optind]; }

  return UCS_OK;
}

std::function<void()> getProgressFunction(std::shared_ptr<ucxx::Worker> worker,
                                          ProgressMode progressMode)
{
  switch (progressMode) {
    case ProgressMode::Polling: return std::bind(std::mem_fn(&ucxx::Worker::progress), worker);
    case ProgressMode::Blocking:
      return std::bind(std::mem_fn(&ucxx::Worker::progressWorkerEvent), worker, -1);
    case ProgressMode::Wait: return std::bind(std::mem_fn(&ucxx::Worker::waitProgress), worker);
    default: return []() {};
  }
}

void waitRequests(ProgressMode progressMode,
                  std::shared_ptr<ucxx::Worker> worker,
                  const std::vector<std::shared_ptr<ucxx::Request>>& requests)
{
  auto progress = getProgressFunction(worker, progressMode);
  // Wait until all messages are completed
  for (auto& r : requests) {
    while (!r->isCompleted())
      progress();
    r->checkError();
  }
}

std::string parseTime(size_t countNs)
{
  if (countNs < 1e3)
    return std::to_string(countNs) + std::string("ns");
  else if (countNs < 1e6)
    return std::to_string(countNs / 1e3) + std::string("us");
  else if (countNs < 1e9)
    return std::to_string(countNs / 1e6) + std::string("ms");
  else
    return std::to_string(countNs / 1e9) + std::string("s");
}

std::string parseBandwidth(size_t totalBytes, size_t countNs)
{
  double bw = totalBytes / (countNs / 1e9);

  if (bw < 1024)
    return std::to_string(bw) + std::string("B/s");
  else if (bw < (1024 * 1024))
    return std::to_string(bw / 1024) + std::string("KB/s");
  else if (bw < (1024 * 1024 * 1024))
    return std::to_string(bw / (1024 * 1024)) + std::string("MB/s");
  else
    return std::to_string(bw / (1024 * 1024 * 1024)) + std::string("GB/s");
}

BufferMapPtr allocateTransferBuffers(size_t message_size)
{
  return std::make_shared<BufferMap>(BufferMap{{SEND, std::vector<char>(message_size, 0xaa)},
                                               {RECV, std::vector<char>(message_size)}});
}

#ifdef UCXX_BENCHMARKS_ENABLE_CUDA
CudaBufferMapPtr allocateCudaTransferBuffers(size_t message_size)
{
  auto bufferMap     = std::make_shared<CudaBufferMap>();
  (*bufferMap)[SEND] = CudaBuffer(message_size);
  (*bufferMap)[RECV] = CudaBuffer(message_size);

  // Initialize send buffer with pattern
  std::vector<char> pattern(message_size, 0xaa);
  cudaError_t err =
    cudaMemcpy((*bufferMap)[SEND].ptr, pattern.data(), message_size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to initialize CUDA send buffer: " +
                             std::string(cudaGetErrorString(err)));
  }

  return bufferMap;
}

CudaManagedBufferMapPtr allocateCudaManagedTransferBuffers(size_t message_size)
{
  auto bufferMap     = std::make_shared<CudaManagedBufferMap>();
  (*bufferMap)[SEND] = CudaManagedBuffer(message_size);
  (*bufferMap)[RECV] = CudaManagedBuffer(message_size);

  // Initialize send buffer with pattern (managed memory can be accessed from host)
  std::vector<char> pattern(message_size, 0xaa);
  std::memcpy((*bufferMap)[SEND].ptr, pattern.data(), message_size);

  return bufferMap;
}

CudaAsyncBufferMapPtr allocateCudaAsyncTransferBuffers(size_t message_size)
{
  auto bufferMap     = std::make_shared<CudaAsyncBufferMap>();
  (*bufferMap)[SEND] = CudaAsyncBuffer(message_size);
  (*bufferMap)[RECV] = CudaAsyncBuffer(message_size);

  // Initialize send buffer with pattern using async copy
  std::vector<char> pattern(message_size, 0xaa);
  cudaError_t err = cudaMemcpyAsync((*bufferMap)[SEND].ptr,
                                    pattern.data(),
                                    message_size,
                                    cudaMemcpyHostToDevice,
                                    (*bufferMap)[SEND].stream);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to initialize CUDA async send buffer: " +
                             std::string(cudaGetErrorString(err)));
  }

  // Synchronize the stream to ensure the copy is complete
  err = cudaStreamSynchronize((*bufferMap)[SEND].stream);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to synchronize CUDA stream: " +
                             std::string(cudaGetErrorString(err)));
  }

  return bufferMap;
}
#endif

auto doTransfer(const app_context_t& app_context,
                std::shared_ptr<ucxx::Worker> worker,
                std::shared_ptr<ucxx::Endpoint> endpoint,
                TagMapPtr tagMap,
                BufferMapPtr bufferMapReuse)
{
#ifdef UCXX_BENCHMARKS_ENABLE_CUDA
  if (app_context.memory_type == MemoryType::Cuda) {
    // CUDA memory transfer
    static CudaBufferMapPtr cudaBufferMapReuse;
    CudaBufferMapPtr localCudaBufferMap;

    if (app_context.reuse_alloc) {
      if (!cudaBufferMapReuse) {
        cudaBufferMapReuse = allocateCudaTransferBuffers(app_context.message_size);
      }
    } else {
      localCudaBufferMap = allocateCudaTransferBuffers(app_context.message_size);
    }

    CudaBufferMapPtr cudaBufferMap =
      app_context.reuse_alloc ? cudaBufferMapReuse : localCudaBufferMap;

    // Safety check to ensure we have a valid buffer map
    if (!cudaBufferMap) { throw std::runtime_error("Failed to allocate CUDA buffer map"); }

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::shared_ptr<ucxx::Request>> requests = {
      endpoint->tagSend((*cudaBufferMap)[SEND].ptr, app_context.message_size, (*tagMap)[SEND]),
      endpoint->tagRecv(
        (*cudaBufferMap)[RECV].ptr, app_context.message_size, (*tagMap)[RECV], ucxx::TagMaskFull)};

    // Wait for requests and clear requests
    waitRequests(app_context.progress_mode, worker, requests);
    auto stop = std::chrono::high_resolution_clock::now();

    if (app_context.verify_results) {
      // Copy data back to host for verification
      std::vector<char> send_data(app_context.message_size);
      std::vector<char> recv_data(app_context.message_size);

      cudaError_t err1 = cudaMemcpy(send_data.data(),
                                    (*cudaBufferMap)[SEND].ptr,
                                    app_context.message_size,
                                    cudaMemcpyDeviceToHost);
      cudaError_t err2 = cudaMemcpy(recv_data.data(),
                                    (*cudaBufferMap)[RECV].ptr,
                                    app_context.message_size,
                                    cudaMemcpyDeviceToHost);

      if (err1 != cudaSuccess || err2 != cudaSuccess) {
        throw std::runtime_error("Failed to copy CUDA data for verification");
      }

      for (size_t j = 0; j < send_data.size(); ++j)
        assert(recv_data[j] == send_data[j]);
    }

    return std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
  } else if (app_context.memory_type == MemoryType::CudaManaged) {
    // CUDA managed memory transfer
    static CudaManagedBufferMapPtr cudaManagedBufferMapReuse;
    CudaManagedBufferMapPtr localCudaManagedBufferMap;

    if (app_context.reuse_alloc) {
      if (!cudaManagedBufferMapReuse) {
        cudaManagedBufferMapReuse = allocateCudaManagedTransferBuffers(app_context.message_size);
      }
    } else {
      localCudaManagedBufferMap = allocateCudaManagedTransferBuffers(app_context.message_size);
    }

    CudaManagedBufferMapPtr cudaManagedBufferMap =
      app_context.reuse_alloc ? cudaManagedBufferMapReuse : localCudaManagedBufferMap;

    // Safety check to ensure we have a valid buffer map
    if (!cudaManagedBufferMap) {
      throw std::runtime_error("Failed to allocate CUDA managed buffer map");
    }

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::shared_ptr<ucxx::Request>> requests = {
      endpoint->tagSend(
        (*cudaManagedBufferMap)[SEND].ptr, app_context.message_size, (*tagMap)[SEND]),
      endpoint->tagRecv((*cudaManagedBufferMap)[RECV].ptr,
                        app_context.message_size,
                        (*tagMap)[RECV],
                        ucxx::TagMaskFull)};

    // Wait for requests and clear requests
    waitRequests(app_context.progress_mode, worker, requests);
    auto stop = std::chrono::high_resolution_clock::now();

    if (app_context.verify_results) {
      // Managed memory can be accessed directly from host
      std::vector<char> send_data(app_context.message_size);
      std::vector<char> recv_data(app_context.message_size);

      std::memcpy(send_data.data(), (*cudaManagedBufferMap)[SEND].ptr, app_context.message_size);
      std::memcpy(recv_data.data(), (*cudaManagedBufferMap)[RECV].ptr, app_context.message_size);

      for (size_t j = 0; j < send_data.size(); ++j)
        assert(recv_data[j] == send_data[j]);
    }

    return std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
  } else if (app_context.memory_type == MemoryType::CudaAsync) {
    // CUDA async memory transfer
    static CudaAsyncBufferMapPtr cudaAsyncBufferMapReuse;
    CudaAsyncBufferMapPtr localCudaAsyncBufferMap;

    if (app_context.reuse_alloc) {
      if (!cudaAsyncBufferMapReuse) {
        cudaAsyncBufferMapReuse = allocateCudaAsyncTransferBuffers(app_context.message_size);
      }
    } else {
      localCudaAsyncBufferMap = allocateCudaAsyncTransferBuffers(app_context.message_size);
    }

    CudaAsyncBufferMapPtr cudaAsyncBufferMap =
      app_context.reuse_alloc ? cudaAsyncBufferMapReuse : localCudaAsyncBufferMap;

    // Safety check to ensure we have a valid buffer map
    if (!cudaAsyncBufferMap) {
      throw std::runtime_error("Failed to allocate CUDA async buffer map");
    }

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::shared_ptr<ucxx::Request>> requests = {
      endpoint->tagSend((*cudaAsyncBufferMap)[SEND].ptr, app_context.message_size, (*tagMap)[SEND]),
      endpoint->tagRecv((*cudaAsyncBufferMap)[RECV].ptr,
                        app_context.message_size,
                        (*tagMap)[RECV],
                        ucxx::TagMaskFull)};

    // Wait for requests and clear requests
    waitRequests(app_context.progress_mode, worker, requests);
    auto stop = std::chrono::high_resolution_clock::now();

    if (app_context.verify_results) {
      // Copy data back to host for verification using async copy
      std::vector<char> send_data(app_context.message_size);
      std::vector<char> recv_data(app_context.message_size);

      cudaError_t err1 = cudaMemcpyAsync(send_data.data(),
                                         (*cudaAsyncBufferMap)[SEND].ptr,
                                         app_context.message_size,
                                         cudaMemcpyDeviceToHost,
                                         (*cudaAsyncBufferMap)[SEND].stream);
      cudaError_t err2 = cudaMemcpyAsync(recv_data.data(),
                                         (*cudaAsyncBufferMap)[RECV].ptr,
                                         app_context.message_size,
                                         cudaMemcpyDeviceToHost,
                                         (*cudaAsyncBufferMap)[RECV].stream);

      if (err1 != cudaSuccess || err2 != cudaSuccess) {
        throw std::runtime_error("Failed to copy CUDA async data for verification");
      }

      // Synchronize streams
      err1 = cudaStreamSynchronize((*cudaAsyncBufferMap)[SEND].stream);
      err2 = cudaStreamSynchronize((*cudaAsyncBufferMap)[RECV].stream);

      if (err1 != cudaSuccess || err2 != cudaSuccess) {
        throw std::runtime_error("Failed to synchronize CUDA streams for verification");
      }

      for (size_t j = 0; j < send_data.size(); ++j)
        assert(recv_data[j] == send_data[j]);
    }

    return std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
  } else {
#endif
    // Host memory transfer
    BufferMapPtr localBufferMap;
    if (!app_context.reuse_alloc)
      localBufferMap = allocateTransferBuffers(app_context.message_size);
    BufferMapPtr bufferMap = app_context.reuse_alloc ? bufferMapReuse : localBufferMap;

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::shared_ptr<ucxx::Request>> requests = {
      endpoint->tagSend((*bufferMap)[SEND].data(), app_context.message_size, (*tagMap)[SEND]),
      endpoint->tagRecv(
        (*bufferMap)[RECV].data(), app_context.message_size, (*tagMap)[RECV], ucxx::TagMaskFull)};

    // Wait for requests and clear requests
    waitRequests(app_context.progress_mode, worker, requests);
    auto stop = std::chrono::high_resolution_clock::now();

    if (app_context.verify_results) {
      for (size_t j = 0; j < (*bufferMap)[SEND].size(); ++j)
        assert((*bufferMap)[RECV][j] == (*bufferMap)[SEND][j]);
    }

    return std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
#ifdef UCXX_BENCHMARKS_ENABLE_CUDA
  }
#endif
}

void performWireup(const app_context_t& app_context,
                   std::shared_ptr<ucxx::Worker> worker,
                   std::shared_ptr<ucxx::Endpoint> endpoint,
                   TagMapPtr tagMap)
{
  // Allocate wireup buffers
  auto wireupBufferMap = std::make_shared<BufferMap>(
    BufferMap{{SEND, std::vector<char>{1, 2, 3}}, {RECV, std::vector<char>(3, 0)}});

  std::vector<std::shared_ptr<ucxx::Request>> requests;

  // Schedule small wireup messages to let UCX identify capabilities between endpoints
  requests.push_back(endpoint->tagSend((*wireupBufferMap)[SEND].data(),
                                       (*wireupBufferMap)[SEND].size() * sizeof(int),
                                       (*tagMap)[SEND]));
  requests.push_back(endpoint->tagRecv((*wireupBufferMap)[RECV].data(),
                                       (*wireupBufferMap)[RECV].size() * sizeof(int),
                                       (*tagMap)[RECV],
                                       ucxx::TagMaskFull));

  // Wait for wireup requests and clear requests
  waitRequests(app_context.progress_mode, worker, requests);

  // Verify wireup result
  for (size_t i = 0; i < (*wireupBufferMap)[SEND].size(); ++i)
    assert((*wireupBufferMap)[RECV][i] == (*wireupBufferMap)[SEND][i]);
}

int main(int argc, char** argv)
{
  app_context_t app_context;
  if (parseCommand(&app_context, argc, argv) != UCS_OK) return -1;

#ifdef UCXX_BENCHMARKS_ENABLE_CUDA
  // Check CUDA support if CUDA memory is requested
  if (app_context.memory_type == MemoryType::Cuda ||
      app_context.memory_type == MemoryType::CudaManaged ||
      app_context.memory_type == MemoryType::CudaAsync) {
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
      std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << std::endl;
      return -1;
    }
  }
#endif

  // Setup: create UCP context, worker, listener and client endpoint.
  auto context = UCXX_EXIT_ON_ERROR(ucxx::createContext({}, ucxx::Context::defaultFeatureFlags),
                                    "Context creation");
  auto worker  = UCXX_EXIT_ON_ERROR(context->createWorker(), "Worker creation");

  bool is_server = app_context.server_addr == NULL;
  auto tagMap    = std::make_shared<TagMap>(TagMap{
       {SEND, is_server ? ucxx::Tag(0) : ucxx::Tag(1)},
       {RECV, is_server ? ucxx::Tag(1) : ucxx::Tag(0)},
  });

  std::shared_ptr<ListenerContext> listener_ctx;
  std::shared_ptr<ucxx::Endpoint> endpoint;
  std::shared_ptr<ucxx::Listener> listener;
  if (is_server) {
    listener_ctx = std::make_unique<ListenerContext>(worker, app_context.endpoint_error_handling);
    listener     = UCXX_EXIT_ON_ERROR(
      worker->createListener(app_context.listener_port, listener_cb, listener_ctx.get()),
      "Listener creation");
    listener_ctx->setListener(listener);
  }

  // Initialize worker progress
  if (app_context.progress_mode == ProgressMode::Blocking)
    worker->initBlockingProgressMode();
  else if (app_context.progress_mode == ProgressMode::ThreadBlocking)
    worker->startProgressThread(false);
  else if (app_context.progress_mode == ProgressMode::ThreadPolling)
    worker->startProgressThread(true);

  auto progress = getProgressFunction(worker, app_context.progress_mode);

  // Block until client connects
  while (is_server && listener_ctx->isAvailable())
    progress();

  if (is_server)
    endpoint = listener_ctx->getEndpoint();
  else
    endpoint = UCXX_EXIT_ON_ERROR(
      worker->createEndpointFromHostname(
        app_context.server_addr, app_context.listener_port, app_context.endpoint_error_handling),
      "Endpoint creation");

  // Perform wireup to let UCX identify capabilities between endpoints
  UCXX_EXIT_ON_ERROR(performWireup(app_context, worker, endpoint, tagMap), "Wireup");

  BufferMapPtr bufferMapReuse;
#ifdef UCXX_BENCHMARKS_ENABLE_CUDA
  if (app_context.reuse_alloc) {
    if (app_context.memory_type == MemoryType::Cuda ||
        app_context.memory_type == MemoryType::CudaManaged ||
        app_context.memory_type == MemoryType::CudaAsync) {
      // CUDA reuse buffers are handled inside doTransfer
    } else {
      bufferMapReuse = allocateTransferBuffers(app_context.message_size);
    }
  }
#else
  if (app_context.reuse_alloc) bufferMapReuse = allocateTransferBuffers(app_context.message_size);
#endif

  // Warmup
  for (size_t n = 0; n < app_context.warmup_iter; ++n)
    UCXX_EXIT_ON_ERROR(doTransfer(app_context, worker, endpoint, tagMap, bufferMapReuse),
                       "Warmup iteration " + std::to_string(n));

  // Schedule send and recv messages on different tags and different ordering
  size_t total_duration_ns = 0;
  for (size_t n = 0; n < app_context.n_iter; ++n) {
    auto duration_ns =
      UCXX_EXIT_ON_ERROR(doTransfer(app_context, worker, endpoint, tagMap, bufferMapReuse),
                         "Transfer iteration " + std::to_string(n));
    total_duration_ns += duration_ns;
    auto elapsed   = parseTime(duration_ns);
    auto bandwidth = parseBandwidth(app_context.message_size * 2, duration_ns);

    if (!is_server)
      std::cout << "Elapsed, bandwidth: " << elapsed << ", " << bandwidth << std::endl;
  }

  auto total_elapsed = parseTime(total_duration_ns);
  auto total_bandwidth =
    parseBandwidth(app_context.n_iter * app_context.message_size * 2, total_duration_ns);

  if (!is_server)
    std::cout << "Total elapsed, bandwidth: " << total_elapsed << ", " << total_bandwidth
              << std::endl;

  // Stop progress thread
  if (app_context.progress_mode == ProgressMode::ThreadBlocking ||
      app_context.progress_mode == ProgressMode::ThreadPolling)
    worker->stopProgressThread();

  return 0;
}
