#include <cassert>
#include <chrono>
#include <iostream>
#include <numeric>
#include <thread>
#include <unistd.h>
#include <vector>

#include <ucxx/api.h>
#include <ucxx/utils/sockaddr.h>
#include <ucxx/utils/ucx.h>

enum class ProgressMode {
  Polling,
  Blocking,
  Wait,
  ThreadPolling,
  ThreadBlocking,
} progress_mode = ProgressMode::Polling;

static uint16_t listener_port = 12345;

class ListenerContext {
 private:
  std::shared_ptr<ucxx::Worker> _worker{nullptr};
  std::shared_ptr<ucxx::Endpoint> _endpoint{nullptr};
  std::shared_ptr<ucxx::Listener> _listener{nullptr};

 public:
  ListenerContext(std::shared_ptr<ucxx::Worker> worker) : _worker{worker} {}

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
  ListenerContext* listener_ctx = (ListenerContext*)arg;

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
  std::cerr << "              'thread-polling' and 'thread-blocking' (default: 'blocking')"
            << std::endl;
  std::cerr << "  -p <port>   Port number to listen at" << std::endl;
  std::cerr << "  -h          Print this help" << std::endl;
  std::cerr << std::endl;
}

ucs_status_t parseCommand(int argc, char* const argv[])
{
  int c;
  while ((c = getopt(argc, argv, "btp:")) != -1) {
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
      case 'h':
      default: printUsage(); return UCS_ERR_UNSUPPORTED;
    }
  }

  return UCS_OK;
}

// void waitRequests(std::shared_ptr<ucxx::Worker> worker,
//                   std::vector<std::shared_ptr<ucxx::Request>>& requests)
// {
//   // Wait until all messages are completed
//   if (progress_mode == ProgressMode::Blocking) {
//     for (auto& r : requests) {
//       do {
//         worker->progressWorkerEvent();
//       } while (!r->isCompleted());
//       r->checkError();
//     }
//   } else {
//     for (auto& r : requests)
//       r->checkError();
//   }
// }

std::function<void()> getProgressFunction(std::shared_ptr<ucxx::Worker> worker,
                                          ProgressMode progressMode)
{
  if (progressMode == ProgressMode::Polling)
    return std::bind(std::mem_fn(&ucxx::Worker::progress), worker);
  else if (progressMode == ProgressMode::Blocking)
    return std::bind(std::mem_fn(&ucxx::Worker::progressWorkerEvent), worker);
  else if (progressMode == ProgressMode::Wait)
    return std::bind(std::mem_fn(&ucxx::Worker::waitProgress), worker);
  else
    return []() {};
}

void waitRequests(ProgressMode progressMode,
                  std::shared_ptr<ucxx::Worker> worker,
                  std::vector<std::shared_ptr<ucxx::Request>>& requests)
{
  auto progress = getProgressFunction(worker, progressMode);
  // Wait until all messages are completed
  for (auto& r : requests) {
    while (!r->isCompleted())
      progress();
    r->checkError();
  }
}

int main(int argc, char** argv)
{
  if (parseCommand(argc, argv) != UCS_OK) return -1;

  // Setup: create UCP context, worker, listener and client endpoint.
  auto context      = ucxx::createContext({}, ucxx::Context::defaultFeatureFlags);
  auto worker       = context->createWorker();
  auto listener_ctx = std::make_unique<ListenerContext>(worker);
  auto listener     = worker->createListener(listener_port, listener_cb, listener_ctx.get());
  listener_ctx->setListener(listener);
  auto endpoint = worker->createEndpointFromHostname("127.0.0.1", listener_port, true);

  // Initialize worker progress
  if (progress_mode == ProgressMode::Blocking)
    worker->initBlockingProgressMode();
  else if (progress_mode == ProgressMode::ThreadBlocking)
    worker->startProgressThread(false);
  else if (progress_mode == ProgressMode::ThreadPolling)
    worker->startProgressThread(true);

  auto progress = getProgressFunction(worker, progress_mode);

  // Block until client connects
  while (listener_ctx->isAvailable())
    progress();

  std::vector<std::shared_ptr<ucxx::Request>> requests;

  // Allocate send buffers
  std::vector<int> sendWireupBuffer{1, 2, 3};
  std::vector<std::vector<int>> sendBuffers{
    std::vector<int>(5), std::vector<int>(500), std::vector<int>(50000)};

  // Allocate receive buffers
  std::vector<int> recvWireupBuffer(sendWireupBuffer.size(), 0);
  std::vector<std::vector<int>> recvBuffers;
  for (const auto& v : sendBuffers)
    recvBuffers.push_back(std::vector<int>(v.size(), 0));

  // Schedule small wireup messages to let UCX identify capabilities between endpoints
  requests.push_back(listener_ctx->getEndpoint()->tagSend(
    sendWireupBuffer.data(), sendWireupBuffer.size() * sizeof(int), 0));
  requests.push_back(
    endpoint->tagRecv(recvWireupBuffer.data(), sendWireupBuffer.size() * sizeof(int), 0));
  ::waitRequests(progress_mode, worker, requests);
  requests.clear();

  // Schedule send and recv messages on different tags and different ordering
  requests.push_back(listener_ctx->getEndpoint()->tagSend(
    sendBuffers[0].data(), sendBuffers[0].size() * sizeof(int), 0));
  requests.push_back(listener_ctx->getEndpoint()->tagRecv(
    recvBuffers[1].data(), recvBuffers[1].size() * sizeof(int), 1));
  requests.push_back(listener_ctx->getEndpoint()->tagSend(
    sendBuffers[2].data(), sendBuffers[2].size() * sizeof(int), 2));
  requests.push_back(
    endpoint->tagRecv(recvBuffers[2].data(), recvBuffers[2].size() * sizeof(int), 2));
  requests.push_back(
    endpoint->tagSend(sendBuffers[1].data(), sendBuffers[1].size() * sizeof(int), 1));
  requests.push_back(
    endpoint->tagRecv(recvBuffers[0].data(), recvBuffers[0].size() * sizeof(int), 0));

  // Wait for requests to be set, i.e., transfers complete
  ::waitRequests(progress_mode, worker, requests);

  // Verify results
  for (size_t i = 0; i < sendWireupBuffer.size(); ++i)
    assert(recvWireupBuffer[i] == sendWireupBuffer[i]);
  for (size_t i = 0; i < sendBuffers.size(); ++i)
    for (size_t j = 0; j < sendBuffers[i].size(); ++j)
      assert(recvBuffers[i][j] == sendBuffers[i][j]);

  // Stop progress thread
  if (progress_mode == ProgressMode::ThreadBlocking || progress_mode == ProgressMode::ThreadPolling)
    worker->stopProgressThread();

  return 0;
}
