# UCXX

UCXX is an object-oriented C++ interface for UCX, with native support for Python bindings.

## Building

### Convenience Script

For convenience, we provide the `./build.sh` script. By default, it will build and install both C++ and Python libraries. For a detailed description on available options please check `./build.sh --help`.

Building C++ and Python libraries manually is also possible, see instructions on building [C++](#c) and [Python](#python).

Additionally, there is a `./build_and_run.sh` script that will call `./build.sh` to build everything as well as running C++ and Python tests and a few benchmarks. Similarly, details on existing options can be queried with `./build_and_run.sh`.

### C++

To build and install C++ library to `${CONDA_PREFIX}`, with both Python and RMM support, as well as building all tests run:

```
mkdir cpp/build
cd cpp/build
cmake .. -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
      -DBUILD_TESTS=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DUCXX_ENABLE_PYTHON=ON \
      -DUCXX_ENABLE_RMM=ON
make -j install
```

### Python

```
cd python
python setup.py install
```

## Running benchmarks

### C++

Currently there is only one C++ benchmark with few options. It can be found under `cpp/build/benchmarks/ucxx_perftest` and for a full list of options `--help` argument can be used.

The benchmark is composed of two processes: a server and a client. The server must not specify an IP address or hostname and will bind to all available interfaces, whereas the client must specify the IP address or hostname where the server can be reached.

Below is an example of running a server first, followed by the client connecting to the server on the `localhost` (`127.0.0.1`). Both processes specify a list of parameters, which are the message size in bytes (`-s 8388608`), that allocations should be reused (`-r`), the number of iterations to perform (`-n 10`) and the progress mode (`-m polling`).

```
$ UCX_TCP_CM_REUSEADDR=y ./benchmarks/ucxx_perftest -s 800000000 -r -n 10 -m polling &
$ ./benchmarks/ucxx_perftest -s 800000000 -r -n 10 -m polling 127.0.0.1
```

It is recommended to use `UCX_TCP_CM_REUSEADDR=y` when binding to interfaces with TCP support to prevent waiting for the process' `TIME_WAIT` state to complete, which often takes 60 seconds after the server has terminated.

### Python

Benchmarks are available for both the Python "core" (synchronous) API and the "high-level" (asynchronous) API.

#### Synchronous

```python
# Thread progress without delayed notification NumPy transfer, 100 iterations
# of single buffer with 100 bytes
python -m ucxx.benchmarks.send_recv \
    --backend ucxx-core \
    --object_type numpy \
    --n-iter 100 \
    --n-bytes 100

# Blocking progress without delayed notification RMM transfer between GPUs 0
# and 3, 100 iterations of 2 buffers (using multi-buffer interface) each with
# 1 MiB
python -m ucxx.benchmarks.send_recv \
    --backend ucxx-core \
    --object_type rmm \
    --server-dev 0 \
    --client-dev 3 \
    --n-iter 100 \
    --n-bytes 100 \
    --progress-mode blocking
```

#### Asynchronous

```python
# NumPy transfer, 100 iterations of 8 buffers (using multi-buffer interface)
# each with 100 bytes
python -m ucxx.benchmarks.send_recv \
    --backend ucxx-async \
    --object_type numpy \
    --n-iter 100 \
    --n-bytes 100 \
    --n-buffers 8

# RMM transfer between GPUs 0 and 3, 100 iterations of 2 buffers (using
# multi-buffer interface) each with 1 MiB
python -m ucxx.benchmarks.send_recv \
    --backend ucxx-async \
    --object_type rmm \
    --server-dev 0 \
    --client-dev 3 \
    --n-iter 100 \
    --n-bytes 1MiB \
    --n-buffers 2

# Polling progress mode without delayed notification NumPy transfer,
# 100 iterations of single buffer with 1 MiB
UCXPY_ENABLE_DELAYED_SUBMISSION=0 \
    python -m ucxx.benchmarks.send_recv \
    --backend ucxx-async \
    --object_type numpy \
    --n-iter 100 \
    --n-bytes 1MiB \
    --progress-mode polling
```

## Logging

Logging is independently available for both C++ and Python APIs. Since the Python interface uses the C++ backend, C++ logging can be enabled when running Python code as well.

### C++

The C++ interface reuses the UCX logger and provides the same log levels and can be enabled via the `UCXX_LOG_LEVEL` environment variable. However, it will not enable UCX logging, one must still set `UCX_LOG_LEVEL` for UCX logging. A few examples are below:

```
# Request trace log level
UCXX_LOG_LEVEL=TRACE_REQ

# Debug log level
UCXX_LOG_LEVEL=DEBUG
```

### Python

The UCXX Python interface uses the `logging` library included in Python. The only used levels currently are `INFO` and `DEBUG`, and can be enabled via the `UCXPY_LOG_LEVEL` environment variable. A few examples are below:

```
# Enable Python info log level
UCXPY_LOG_LEVEL=INFO

# Enable Python debug log level, UCXX request trace log level and UCX data log level
UCXPY_LOG_LEVEL=DEBUG UCXX_LOG_LEVEL=TRACE_REQ UCX_LOG_LEVEL=DATA
```
