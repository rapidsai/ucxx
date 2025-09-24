/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#include <ucxx/api.h>
#include <ucxx/buffer.h>
#include <ucxx/utils/sockaddr.h>
#include <ucxx/utils/ucx.h>

#if UCXX_ENABLE_RMM
#include <rmm/device_buffer.hpp>
#endif

class ListenerContext {
 private:
  std::shared_ptr<ucxx::Worker> _worker{nullptr};
  std::shared_ptr<ucxx::Endpoint> _endpoint{nullptr};
  std::shared_ptr<ucxx::Listener> _listener{nullptr};

 public:
  explicit ListenerContext(std::shared_ptr<ucxx::Worker> worker) : _worker{worker} {}

  ~ListenerContext() { releaseEndpoint(); }

  void setListener(std::shared_ptr<ucxx::Listener> listener) { _listener = listener; }

  std::shared_ptr<ucxx::Listener> getListener() { return _listener; }

  std::shared_ptr<ucxx::Endpoint> getEndpoint() { return _endpoint; }

  bool isAvailable() const { return _endpoint == nullptr; }

  void createEndpointFromConnRequest(ucp_conn_request_h conn_request)
  {
    if (!isAvailable()) throw std::runtime_error("Listener context already has an endpoint");

    static bool endpoint_error_handling = true;
    _endpoint = _listener->createEndpointFromConnRequest(conn_request, endpoint_error_handling);
  }

  void releaseEndpoint() { _endpoint.reset(); }
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
  std::cerr << "Usage: basic [parameters]" << std::endl;
  std::cerr << " basic client/server example" << std::endl;
  std::cerr << std::endl;
  std::cerr << "Parameters are:" << std::endl;
  std::cerr << "  -m          progress mode to use, valid values are: 'polling', 'blocking',"
            << std::endl;
  std::cerr << "              'thread-polling', 'thread-blocking' and 'wait' (default: 'blocking')"
            << std::endl;
  std::cerr << "  -p <port>   Port number to listen at" << std::endl;
  std::cerr << "  -s <send_buffer_type>   Send buffer type, valid values are: 'host', 'rmm' "
               "(default: 'host')"
            << std::endl;
  std::cerr << "  -r <recv_buffer_type>   Recv buffer type, valid values are: 'host', 'rmm' "
               "(default: 'host')"
            << std::endl;
  std::cerr << "  -h          Print this help" << std::endl;
  std::cerr << std::endl;
}

enum class ProgressMode {
  Polling,
  Blocking,
  Wait,
  ThreadPolling,
  ThreadBlocking,
};

struct args {
  ProgressMode progress_mode{ProgressMode::Blocking};
  uint16_t listener_port{12345};
  ucxx::BufferType send_buf_type{ucxx::BufferType::Host};
  ucxx::BufferType recv_buf_type{ucxx::BufferType::Host};

  ucs_status_t parse(int argc, char* const argv[])
  {
    int c;

    auto parseBufferType = [](const std::string& bufferTypeString) {
      if (bufferTypeString == "host") {
        return ucxx::BufferType::Host;
      } else if (bufferTypeString == "rmm") {
#if UCXX_ENABLE_RMM
        return ucxx::BufferType::RMM;
#else
        std::cerr << "RMM support not enabled, please compile with -DUCXX_ENABLE_RMM=1"
                  << std::endl;
        return ucxx::BufferType::Invalid;
#endif
      } else {
        return ucxx::BufferType::Invalid;
      }
    };

    while ((c = getopt(argc, argv, "m:p:s:r:h")) != -1) {
      switch (c) {
        case 'm':
          if (strcmp(optarg, "blocking") == 0) {
            progress_mode = ProgressMode::Blocking;
            break;
          } else if (strcmp(optarg, "polling") == 0) {
            progress_mode = ProgressMode::Polling;
            break;
          } else if (strcmp(optarg, "thread-blocking") == 0) {
            progress_mode = ProgressMode::ThreadBlocking;
            break;
          } else if (strcmp(optarg, "thread-polling") == 0) {
            progress_mode = ProgressMode::ThreadPolling;
            break;
          } else if (strcmp(optarg, "wait") == 0) {
            progress_mode = ProgressMode::Wait;
            break;
          } else {
            std::cerr << "Invalid progress mode: " << optarg << std::endl;
            return UCS_ERR_INVALID_PARAM;
          }
        case 'p':
          listener_port = atoi(optarg);
          if (listener_port <= 0) {
            std::cerr << "Wrong listener port: " << listener_port << std::endl;
            return UCS_ERR_UNSUPPORTED;
          }
          break;
        case 's':
          send_buf_type = parseBufferType(optarg);
          if (send_buf_type == ucxx::BufferType::Invalid) {
            std::cerr << "Invalid send buffer type: " << optarg << std::endl;
            return UCS_ERR_INVALID_PARAM;
          }
          break;
        case 'r':
          recv_buf_type = parseBufferType(optarg);
          if (recv_buf_type == ucxx::BufferType::Invalid) {
            std::cerr << "Invalid recv buffer type: " << optarg << std::endl;
            return UCS_ERR_INVALID_PARAM;
          }
          break;
        case 'h':
        default: printUsage(); return UCS_ERR_UNSUPPORTED;
      }
    }

    return UCS_OK;
  }
};

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

template <typename T>
std::shared_ptr<ucxx::Buffer> makeBuffer(ucxx::BufferType bufferType, T* values, size_t size)
{
  switch (bufferType) {
    case ucxx::BufferType::Host:
      return std::make_shared<ucxx::HostBuffer>(values, size * sizeof(T));
    case ucxx::BufferType::RMM:
#if UCXX_ENABLE_RMM
    {
      auto buf =
        std::make_unique<rmm::device_buffer>(values, size * sizeof(T), rmm::cuda_stream_default);
      rmm::cuda_stream_default.synchronize();
      return std::make_shared<ucxx::RMMBuffer>(std::move(buf));
    }
#endif
    default: throw std::runtime_error("Unable to make buffer from values");
  }
}

auto verify_buffers(ucxx::Buffer* expected, ucxx::Buffer* actual)
{
  std::vector<uint8_t> host_expected, host_actual;
  void *host_expected_ptr, *host_actual_ptr;

#if UCXX_ENABLE_RMM
  auto copy_to_host = [](auto& buffer, auto& host_buffer) {
    // copy RMM buffer to host
    host_buffer.resize(buffer->getSize());
    auto stream = rmm::cuda_stream_default;
    assert(
      cudaMemcpyAsync(
        host_buffer.data(), buffer->data(), buffer->getSize(), cudaMemcpyDefault, stream.value()) ==
      cudaSuccess);
    stream.synchronize();
    return host_buffer.data();
  };
#endif

  if (expected->getType() == ucxx::BufferType::RMM) {
#if UCXX_ENABLE_RMM
    host_expected_ptr = copy_to_host(expected, host_expected);
#else
    throw std::runtime_error("RMM support not enabled, please compile with -DUCXX_ENABLE_RMM=1");
#endif
  } else {
    host_expected_ptr = expected->data();
  }

  if (actual->getType() == ucxx::BufferType::RMM) {
#if UCXX_ENABLE_RMM
    host_actual_ptr = copy_to_host(actual, host_actual);
#else
    throw std::runtime_error("RMM support not enabled, please compile with -DUCXX_ENABLE_RMM=1");
#endif
  } else {
    host_actual_ptr = actual->data();
  }

  assert(std::memcmp(host_expected_ptr, host_actual_ptr, expected->getSize()) == 0);
}

int main(int argc, char** argv)
{
  args args;
  if (args.parse(argc, argv) != UCS_OK) return -1;

  // Setup: create UCP context, worker, listener and client endpoint.
  auto context      = ucxx::createContext({}, ucxx::Context::defaultFeatureFlags);
  auto worker       = context->createWorker();
  auto listener_ctx = std::make_unique<ListenerContext>(worker);
  auto listener     = worker->createListener(args.listener_port, listener_cb, listener_ctx.get());
  listener_ctx->setListener(listener);
  auto endpoint = worker->createEndpointFromHostname("127.0.0.1", args.listener_port, true);

  // Initialize worker progress
  if (args.progress_mode == ProgressMode::Blocking)
    worker->initBlockingProgressMode();
  else if (args.progress_mode == ProgressMode::ThreadBlocking)
    worker->startProgressThread(false);
  else if (args.progress_mode == ProgressMode::ThreadPolling)
    worker->startProgressThread(true);

  auto progress = getProgressFunction(worker, args.progress_mode);

  // Block until client connects
  while (listener_ctx->isAvailable())
    progress();

  std::vector<std::shared_ptr<ucxx::Request>> requests;

  // Allocate send buffers
  std::vector<int> wireupValues{1, 2, 3};
  auto sendWireupBuffer = makeBuffer(
    ucxx::BufferType::Host, wireupValues.data(), wireupValues.size());  // host wireup buffer

  std::vector<int> sendValues(50000);
  std::iota(sendValues.begin(), sendValues.end(), 0);
  std::vector<std::shared_ptr<ucxx::Buffer>> sendBuffers{
    makeBuffer(args.send_buf_type, sendValues.data(), 5),
    makeBuffer(args.send_buf_type, sendValues.data(), 500),
    makeBuffer(args.send_buf_type, sendValues.data(), 50000),
  };

  // Allocate receive buffers
  auto recvWireupBuffer =
    allocateBuffer(ucxx::BufferType::Host, sendWireupBuffer->getSize());  // host wireup buffer
  std::vector<std::shared_ptr<ucxx::Buffer>> recvBuffers;
  for (const auto& v : sendBuffers)
    recvBuffers.push_back(allocateBuffer(args.recv_buf_type, v->getSize()));

  // Schedule small wireup messages to let UCX identify capabilities between endpoints
  requests.push_back(listener_ctx->getEndpoint()->tagSend(
    sendWireupBuffer->data(), sendWireupBuffer->getSize(), ucxx::Tag{0}));
  requests.push_back(endpoint->tagRecv(
    recvWireupBuffer->data(), sendWireupBuffer->getSize(), ucxx::Tag{0}, ucxx::TagMaskFull));
  ::waitRequests(args.progress_mode, worker, requests);
  requests.clear();

  // Schedule send and recv messages on different tags and different ordering
  requests.push_back(listener_ctx->getEndpoint()->tagSend(
    sendBuffers[0]->data(), sendBuffers[0]->getSize(), ucxx::Tag{0}));
  requests.push_back(listener_ctx->getEndpoint()->tagRecv(
    recvBuffers[1]->data(), recvBuffers[1]->getSize(), ucxx::Tag{1}, ucxx::TagMaskFull));
  requests.push_back(listener_ctx->getEndpoint()->tagSend(
    sendBuffers[2]->data(), sendBuffers[2]->getSize(), ucxx::Tag{2}, ucxx::TagMaskFull));
  requests.push_back(endpoint->tagRecv(
    recvBuffers[2]->data(), recvBuffers[2]->getSize(), ucxx::Tag{2}, ucxx::TagMaskFull));
  requests.push_back(
    endpoint->tagSend(sendBuffers[1]->data(), sendBuffers[1]->getSize(), ucxx::Tag{1}));
  requests.push_back(endpoint->tagRecv(
    recvBuffers[0]->data(), recvBuffers[0]->getSize(), ucxx::Tag{0}, ucxx::TagMaskFull));

  // Wait for requests to be set, i.e., transfers complete
  ::waitRequests(args.progress_mode, worker, requests);

  // Verify results
  verify_buffers(sendWireupBuffer.get(), recvWireupBuffer.get());
  for (size_t i = 0; i < sendBuffers.size(); ++i) {
    verify_buffers(sendBuffers[i].get(), recvBuffers[i].get());
  }

  // Stop progress thread
  if (args.progress_mode == ProgressMode::ThreadBlocking ||
      args.progress_mode == ProgressMode::ThreadPolling)
    worker->stopProgressThread();

  std::cout << "Example completed successfully" << std::endl;

  return 0;
}
