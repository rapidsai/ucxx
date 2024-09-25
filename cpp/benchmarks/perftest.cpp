/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <unistd.h>  // for getopt, optarg

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
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
};

enum transfer_type_t { SEND, RECV };

typedef std::unordered_map<transfer_type_t, std::vector<char>> BufferMap;
typedef std::unordered_map<transfer_type_t, ucxx::Tag> TagMap;

typedef std::shared_ptr<BufferMap> BufferMapPtr;
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

static void printUsage(std::string_view executable_name)
{
  std::cerr << "UCXX performance testing tool" << std::endl;
  std::cerr << std::endl;
  std::cerr << "Usage: " << executable_name << " [server-hostname] [options]" << std::endl;
  std::cerr << std::endl;
  std::cerr << "Parameters are:" << std::endl;
  std::cerr << "  -m          progress mode to use, valid values are: 'polling', 'blocking',"
            << std::endl;
  std::cerr << "              'thread-polling' and 'thread-blocking' (default: 'blocking')"
            << std::endl;
  std::cerr << "  -t          use thread progress mode (disabled)" << std::endl;
  std::cerr << "  -e          create endpoints with error handling support (disabled)" << std::endl;
  std::cerr << "  -p <port>   port number to listen at (12345)" << std::endl;
  std::cerr << "  -s <bytes>  message size (8)" << std::endl;
  std::cerr << "  -n <int>    number of iterations to run (100)" << std::endl;
  std::cerr << "  -r          reuse memory allocation (disabled)" << std::endl;
  std::cerr << "  -v          verify results (disabled)" << std::endl;
  std::cerr << "  -w <int>    number of warmup iterations to run (3)" << std::endl;
  std::cerr << "  -h          print this help" << std::endl;
  std::cerr << std::endl;
}

ucs_status_t parseCommand(app_context_t* app_context, int argc, char* const argv[])
{
  optind = 1;
  int c;
  while ((c = getopt(argc, argv, "m:p:s:w:n:ervh")) != -1) {
    switch (c) {
      case 'm':
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
      case 'h':
      default: printUsage(std::string_view(argv[0])); return UCS_ERR_INVALID_PARAM;
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

std::string appendSpaces(const std::string_view input, const int maxLength = 91)
{
  int spacesToAdd = std::max(0, maxLength - static_cast<int>(input.length()));
  return std::string(input) + std::string(spacesToAdd, ' ');
}

void printHeader(std::string_view sendMemory, std::string_view recvMemory, size_t size)
{
  // clang-format off
  std::cout << "+--------------+--------------+------------------------------+---------------------+-----------------------+" << std::endl;
  std::cout << "|              |              |       overhead (usec)        |   bandwidth (MB/s)  |  message rate (msg/s) |" << std::endl;
  std::cout << "+----------------------------------------------------------------------------------------------------------+" << std::endl;
  std::cout << "+--------------+--------------+----------+---------+---------+----------+----------+-----------+-----------+" << std::endl;
  std::cout << "| Test:         tag match bandwidth                                                                        |" << std::endl;
  std::cout << "|    Stage     | # iterations | 50.0%ile | average | overall |  average |  overall |  average  |  overall  |" << std::endl;
  std::cout << "+--------------+--------------+----------+---------+---------+----------+----------+-----------+-----------+" << std::endl;
  std::cout << "| Send memory:  " << appendSpaces(sendMemory) << "|" << std::endl;
  std::cout << "| Recv memory:  " << appendSpaces(recvMemory) << "|" << std::endl;
  std::cout << "| Message size: " << appendSpaces(std::to_string(size)) << "|" << std::endl;
  std::cout << "+----------------------------------------------------------------------------------------------------------+" << std::endl;
  // clang-format on
}

std::string floatToString(double number, size_t precision = 2)
{
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2) << number;
  return oss.str();
}

void printProgress(size_t iteration,
                   double overhead_50,
                   double overhead_avg,
                   double overhead_overall,
                   double bandwidthAverage,
                   double bandwidthOverall,
                   size_t messageRateAverage,
                   size_t messageRateOverall)
{
  std::cout << "                " << appendSpaces(std::to_string(iteration), 15)
            << appendSpaces("N/A", 11) << appendSpaces("N/A", 10) << appendSpaces("N/A", 10)
            << appendSpaces(floatToString(bandwidthAverage), 11)
            << appendSpaces(floatToString(bandwidthOverall), 11)
            << appendSpaces(std::to_string(messageRateAverage), 12)
            << appendSpaces(std::to_string(messageRateOverall), 0) << std::endl;
}

auto doTransfer(const app_context_t& app_context,
                std::shared_ptr<ucxx::Worker> worker,
                std::shared_ptr<ucxx::Endpoint> endpoint,
                TagMapPtr tagMap,
                BufferMapPtr bufferMapReuse)
{
  BufferMapPtr localBufferMap;
  if (!app_context.reuse_alloc) localBufferMap = allocateTransferBuffers(app_context.message_size);
  BufferMapPtr bufferMap = app_context.reuse_alloc ? bufferMapReuse : localBufferMap;

  auto start                                           = std::chrono::high_resolution_clock::now();
  std::vector<std::shared_ptr<ucxx::Request>> requests = {
    endpoint->tagSend((*bufferMap)[SEND].data(), app_context.message_size, (*tagMap)[SEND]),
    endpoint->tagRecv(
      (*bufferMap)[RECV].data(), app_context.message_size, (*tagMap)[RECV], ucxx::TagMaskFull)};

  // Wait for requests and clear requests
  waitRequests(app_context.progress_mode, worker, requests);
  auto stop = std::chrono::high_resolution_clock::now();

  if (app_context.verify_results) {
    for (size_t j = 0; j < (*bufferMap)[SEND].size(); ++j)
      assert((*bufferMap)[RECV][j] == (*bufferMap)[RECV][j]);
  }

  return std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
}

auto doWireup(const app_context_t& app_context,
              std::shared_ptr<ucxx::Worker> worker,
              std::shared_ptr<ucxx::Endpoint> endpoint,
              TagMapPtr tagMap)
{
  std::vector<std::shared_ptr<ucxx::Request>> requests;

  // Allocate wireup buffers
  auto wireupBufferMap = std::make_shared<BufferMap>(
    BufferMap{{SEND, std::vector<char>{1, 2, 3}}, {RECV, std::vector<char>(3, 0)}});

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
}

int main(int argc, char** argv)
{
  app_context_t app_context;
  if (parseCommand(&app_context, argc, argv) != UCS_OK) return -1;

  bool is_server = app_context.server_addr == NULL;
  if (!is_server) printHeader("host", "host", app_context.message_size);

  // Setup: create UCP context, worker, listener and client endpoint.
  auto context = ucxx::createContext({}, ucxx::Context::defaultFeatureFlags);
  auto worker  = context->createWorker();

  auto tagMap = std::make_shared<TagMap>(TagMap{
    {SEND, is_server ? ucxx::Tag{0} : ucxx::Tag{1}},
    {RECV, is_server ? ucxx::Tag{1} : ucxx::Tag{0}},
  });

  std::shared_ptr<ListenerContext> listener_ctx;
  std::shared_ptr<ucxx::Endpoint> endpoint;
  std::shared_ptr<ucxx::Listener> listener;
  if (is_server) {
    listener_ctx = std::make_unique<ListenerContext>(worker, app_context.endpoint_error_handling);
    listener = worker->createListener(app_context.listener_port, listener_cb, listener_ctx.get());
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
    endpoint = worker->createEndpointFromHostname(
      app_context.server_addr, app_context.listener_port, app_context.endpoint_error_handling);

  std::vector<std::shared_ptr<ucxx::Request>> requests;

  // Do wireup
  doWireup(app_context, worker, endpoint, tagMap);

  // Verify wireup result
  for (size_t i = 0; i < (*wireupBufferMap)[SEND].size(); ++i)
    assert((*wireupBufferMap)[RECV][i] == (*wireupBufferMap)[SEND][i]);

  BufferMapPtr bufferMapReuse;
  if (app_context.reuse_alloc) bufferMapReuse = allocateTransferBuffers(app_context.message_size);

  // Warmup
  for (size_t n = 0; n < app_context.warmup_iter; ++n)
    doTransfer(app_context, worker, endpoint, tagMap, bufferMapReuse);

  // Schedule send and recv messages on different tags and different ordering
  size_t total_duration_ns = 0;
  auto last_print_time     = std::chrono::steady_clock::now();

  size_t groupDuration   = 0;
  size_t totalDuration   = 0;
  size_t groupIterations = 0;

  for (size_t n = 0; n < app_context.n_iter; ++n) {
    auto duration_ns = doTransfer(app_context, worker, endpoint, tagMap, bufferMapReuse);
    total_duration_ns += duration_ns;
    auto elapsed   = parseTime(duration_ns);
    auto bandwidth = parseBandwidth(app_context.message_size * 2, duration_ns);

    groupDuration += duration_ns;
    totalDuration += duration_ns;
    ++groupIterations;

    auto current_time = std::chrono::steady_clock::now();
    auto elapsed_time =
      std::chrono::duration_cast<std::chrono::seconds>(current_time - last_print_time);

    if (!is_server && (elapsed_time.count() >= 1 || n == app_context.n_iter - 1)) {
      auto groupBytes       = app_context.message_size * 2 * groupIterations;
      auto groupBandwidth   = groupBytes / (groupDuration / 1e3);
      auto totalBytes       = app_context.message_size * 2 * (n + 1);
      auto totalBandwidth   = totalBytes / (totalDuration / 1e3);
      auto groupMessageRate = groupIterations * 2 / (groupDuration / 1e9);
      auto totalMessageRate = (n + 1) * 2 / (totalDuration / 1e9);

      printProgress(n + 1,
                    0.0f,
                    0.0f,
                    0.0f,
                    groupBandwidth,
                    totalBandwidth,
                    groupMessageRate,
                    totalMessageRate);

      groupDuration   = 0;
      groupIterations = 0;

      last_print_time = current_time;
    }
  }

  auto total_elapsed = parseTime(total_duration_ns);
  auto total_bandwidth =
    parseBandwidth(app_context.n_iter * app_context.message_size * 2, total_duration_ns);

  // Stop progress thread
  if (app_context.progress_mode == ProgressMode::ThreadBlocking ||
      app_context.progress_mode == ProgressMode::ThreadPolling)
    worker->stopProgressThread();

  return 0;
}
