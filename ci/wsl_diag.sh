#!/bin/bash
# TEMP diagnostics for WSL Python benchmark slowness
set -uo pipefail

nvidia-smi -q | grep -iE "driver model|product name|addressing mode|bar1|persistence" || true
python -c "import ucxx; print('ucx', ucxx.__ucx_version__)"

python - <<'PY'
import time
import cupy as cp
import numpy as np

for nbytes in (8 << 10, 256 << 10, 512 << 10, 8 << 20):
    d = cp.empty(nbytes, dtype="u1")
    pageable = np.empty(nbytes, dtype="u1")
    pinned_mem = cp.cuda.alloc_pinned_memory(nbytes)
    pinned = np.frombuffer(pinned_mem, dtype="u1", count=nbytes)
    registered = np.empty(nbytes, dtype="u1")
    cp.cuda.runtime.hostRegister(registered.ctypes.data, nbytes, 0)
    for name, h in (("pageable", pageable), ("pinned", pinned), ("hostreg", registered)):
        for direction in ("D2H", "H2D"):
            n = 50
            cp.cuda.runtime.deviceSynchronize()
            t = time.perf_counter()
            for _ in range(n):
                if direction == "D2H":
                    d.get(out=h)
                else:
                    d.set(h)
            cp.cuda.runtime.deviceSynchronize()
            dt = (time.perf_counter() - t) / n
            print(f"memcpy {direction} {name:8s} {nbytes:>9d} B: {dt*1e6:9.1f} us/copy, {nbytes/dt/2**20:9.1f} MiB/s")
PY

bench() {
  local obj=$1; shift
  echo "=== -o ${obj} $*"
  env UCX_KEEPALIVE_INTERVAL=1ms "$@" timeout 300 python -m ucxx.benchmarks.send_recv \
    --backend ucxx-core -o "${obj}" --reuse-alloc -n 8388608 --n-buffers 1 --progress-mode thread --n-iter 3 2>&1 \
    | grep -E "Bandwidth \(average\)|Latency \(average\)|UCX_TLS  |Object type|ERROR|Error"
  echo "exit=${PIPESTATUS[0]} elapsed=${SECONDS}s"
}

SECONDS=0; bench numpy UCX_TLS=^cuda_ipc
SECONDS=0; bench cupy UCX_TLS=^cuda_ipc
SECONDS=0; bench cupy UCX_TLS=all
SECONDS=0; bench cupy UCX_TLS=^cuda_ipc UCX_RNDV_FRAG_MEM_TYPE=cuda
SECONDS=0; bench cupy UCX_TLS=^cuda_ipc UCX_RNDV_FRAG_SIZE=host:8m
SECONDS=0; bench numpy UCX_TLS=tcp
SECONDS=0; bench cupy UCX_TLS=tcp,cuda_copy UCX_RNDV_THRESH=inf
SECONDS=0; bench cupy UCX_TLS=^cuda_ipc UCX_RNDV_PIPELINE_SEND_THRESH=inf
SECONDS=0; bench cupy UCX_TLS=^cuda_ipc UCX_CUDA_COPY_MAX_REG_RATIO=0

echo "=== PROTO_INFO ^cuda_ipc cupy"
UCX_PROTO_INFO=y UCX_TLS=^cuda_ipc UCX_KEEPALIVE_INTERVAL=1ms timeout 300 python -m ucxx.benchmarks.send_recv \
  --backend ucxx-core -o cupy --reuse-alloc -n 8388608 --n-iter 1 2>&1 \
  | grep -A8 -E "from cuda/GPU0|into cuda/GPU0" | head -60

echo "=== nsys-free timing of ucx cuda_copy with UCX_LOG_LEVEL=info"
UCX_LOG_LEVEL=info UCX_TLS=^cuda_ipc UCX_KEEPALIVE_INTERVAL=1ms timeout 300 python -m ucxx.benchmarks.send_recv \
  --backend ucxx-core -o cupy --reuse-alloc -n 8388608 --n-iter 1 2>&1 | grep -viE "^\s*$" | grep -iE "cuda|mpool|warn" | head -40
