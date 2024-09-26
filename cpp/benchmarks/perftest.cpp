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

typedef std::shared_ptr<TagMap> TagMapPtr;

struct ApplicationContext {
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

ucs_status_t parseCommand(ApplicationContext* appContext, int argc, char* const argv[])
{
  optind = 1;
  int c;
  while ((c = getopt(argc, argv, "m:p:s:w:n:ervh")) != -1) {
    switch (c) {
      case 'm':
        if (strcmp(optarg, "blocking") == 0) {
          appContext->progress_mode = ProgressMode::Blocking;
          break;
        } else if (strcmp(optarg, "polling") == 0) {
          appContext->progress_mode = ProgressMode::Polling;
          break;
        } else if (strcmp(optarg, "thread-blocking") == 0) {
          appContext->progress_mode = ProgressMode::ThreadBlocking;
          break;
        } else if (strcmp(optarg, "thread-polling") == 0) {
          appContext->progress_mode = ProgressMode::ThreadPolling;
          break;
        } else if (strcmp(optarg, "wait") == 0) {
          appContext->progress_mode = ProgressMode::Wait;
          break;
        } else {
          std::cerr << "Invalid progress mode: " << optarg << std::endl;
          return UCS_ERR_INVALID_PARAM;
        }
      case 'p':
        appContext->listener_port = atoi(optarg);
        if (appContext->listener_port <= 0) {
          std::cerr << "Wrong listener port: " << appContext->listener_port << std::endl;
          return UCS_ERR_INVALID_PARAM;
        }
        break;
      case 's':
        appContext->message_size = atoi(optarg);
        if (appContext->message_size <= 0) {
          std::cerr << "Wrong message size: " << appContext->message_size << std::endl;
          return UCS_ERR_INVALID_PARAM;
        }
        break;
      case 'w':
        appContext->warmup_iter = atoi(optarg);
        if (appContext->warmup_iter <= 0) {
          std::cerr << "Wrong number of warmup iterations: " << appContext->warmup_iter
                    << std::endl;
          return UCS_ERR_INVALID_PARAM;
        }
        break;
      case 'n':
        appContext->n_iter = atoi(optarg);
        if (appContext->n_iter <= 0) {
          std::cerr << "Wrong number of iterations: " << appContext->n_iter << std::endl;
          return UCS_ERR_INVALID_PARAM;
        }
        break;
      case 'e': appContext->endpoint_error_handling = true; break;
      case 'r': appContext->reuse_alloc = true; break;
      case 'v': appContext->verify_results = true; break;
      case 'h':
      default: printUsage(std::string_view(argv[0])); return UCS_ERR_INVALID_PARAM;
    }
  }

  if (optind < argc) { appContext->server_addr = argv[optind]; }

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

BufferMap allocateTransferBuffers(size_t message_size)
{
  return BufferMap{{SEND, std::vector<char>(message_size, 0xaa)},
                   {RECV, std::vector<char>(message_size)}};
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

class Application {
 private:
  ApplicationContext _appContext{};
  bool _isServer{false};
  std::shared_ptr<ucxx::Context> _context{nullptr};
  std::shared_ptr<ucxx::Worker> _worker{nullptr};
  std::shared_ptr<ucxx::Endpoint> _endpoint{nullptr};
  std::shared_ptr<ucxx::Listener> _listener{nullptr};
  std::shared_ptr<ListenerContext> _listenerContext{nullptr};
  std::shared_ptr<TagMap> _tagMap{nullptr};
  BufferMap _bufferMapReuse{};

  void doWireup()
  {
    std::vector<std::shared_ptr<ucxx::Request>> requests;

    // Allocate wireup buffers
    auto wireupBufferMap = std::make_shared<BufferMap>(
      BufferMap{{SEND, std::vector<char>{1, 2, 3}}, {RECV, std::vector<char>(3, 0)}});

    // Schedule small wireup messages to let UCX identify capabilities between endpoints
    requests.push_back(_endpoint->tagSend((*wireupBufferMap)[SEND].data(),
                                          (*wireupBufferMap)[SEND].size() * sizeof(int),
                                          (*_tagMap)[SEND]));
    requests.push_back(_endpoint->tagRecv((*wireupBufferMap)[RECV].data(),
                                          (*wireupBufferMap)[RECV].size() * sizeof(int),
                                          (*_tagMap)[RECV],
                                          ucxx::TagMaskFull));

    // Wait for wireup requests and clear requests
    waitRequests(_appContext.progress_mode, _worker, requests);

    // Verify wireup result
    for (size_t i = 0; i < (*wireupBufferMap)[SEND].size(); ++i)
      assert((*wireupBufferMap)[RECV][i] == (*wireupBufferMap)[SEND][i]);
  }

  auto doTransfer()
  {
    BufferMap localBufferMap;
    if (!_appContext.reuse_alloc)
      localBufferMap = allocateTransferBuffers(_appContext.message_size);
    BufferMap& bufferMap = _appContext.reuse_alloc ? _bufferMapReuse : localBufferMap;

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::shared_ptr<ucxx::Request>> requests = {
      _endpoint->tagSend((bufferMap)[SEND].data(), _appContext.message_size, (*_tagMap)[SEND]),
      _endpoint->tagRecv(
        (bufferMap)[RECV].data(), _appContext.message_size, (*_tagMap)[RECV], ucxx::TagMaskFull)};

    // Wait for requests and clear requests
    waitRequests(_appContext.progress_mode, _worker, requests);
    auto stop = std::chrono::high_resolution_clock::now();

    if (_appContext.verify_results) {
      for (size_t j = 0; j < (bufferMap)[SEND].size(); ++j)
        assert((bufferMap)[RECV][j] == (bufferMap)[RECV][j]);
    }

    return std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
  }

 public:
  explicit Application(ApplicationContext&& appContext)
    : _appContext(appContext), _isServer(appContext.server_addr == NULL)
  {
    if (!_isServer) printHeader("host", "host", appContext.message_size);

    // Setup: create UCP context, worker, listener and client endpoint.
    _context = ucxx::createContext({}, ucxx::Context::defaultFeatureFlags);
    _worker  = _context->createWorker();

    _tagMap = std::make_shared<TagMap>(TagMap{
      {SEND, _isServer ? ucxx::Tag{0} : ucxx::Tag{1}},
      {RECV, _isServer ? ucxx::Tag{1} : ucxx::Tag{0}},
    });

    if (_isServer) {
      _listenerContext =
        std::make_unique<ListenerContext>(_worker, _appContext.endpoint_error_handling);
      _listener =
        _worker->createListener(_appContext.listener_port, listener_cb, _listenerContext.get());
      _listenerContext->setListener(_listener);
    }

    // Initialize worker progress
    if (_appContext.progress_mode == ProgressMode::Blocking)
      _worker->initBlockingProgressMode();
    else if (_appContext.progress_mode == ProgressMode::ThreadBlocking)
      _worker->startProgressThread(false);
    else if (_appContext.progress_mode == ProgressMode::ThreadPolling)
      _worker->startProgressThread(true);

    auto progress = getProgressFunction(_worker, _appContext.progress_mode);

    // Block until client connects
    while (_isServer && _listenerContext->isAvailable())
      progress();

    if (_isServer)
      _endpoint = _listenerContext->getEndpoint();
    else
      _endpoint = _worker->createEndpointFromHostname(
        _appContext.server_addr, _appContext.listener_port, _appContext.endpoint_error_handling);

    std::vector<std::shared_ptr<ucxx::Request>> requests;
  }

  void run()
  {
    // Do wireup
    doWireup();

    if (_appContext.reuse_alloc)
      _bufferMapReuse = allocateTransferBuffers(_appContext.message_size);

    // Warmup
    for (size_t n = 0; n < _appContext.warmup_iter; ++n)
      doTransfer();

    // Schedule send and recv messages on different tags and different ordering
    size_t total_duration_ns = 0;
    auto last_print_time     = std::chrono::steady_clock::now();

    size_t groupDuration   = 0;
    size_t totalDuration   = 0;
    size_t groupIterations = 0;

    for (size_t n = 0; n < _appContext.n_iter; ++n) {
      auto duration_ns = doTransfer();
      total_duration_ns += duration_ns;
      auto elapsed   = parseTime(duration_ns);
      auto bandwidth = parseBandwidth(_appContext.message_size * 2, duration_ns);

      groupDuration += duration_ns;
      totalDuration += duration_ns;
      ++groupIterations;

      auto current_time = std::chrono::steady_clock::now();
      auto elapsed_time =
        std::chrono::duration_cast<std::chrono::seconds>(current_time - last_print_time);

      if (!_isServer && (elapsed_time.count() >= 1 || n == _appContext.n_iter - 1)) {
        auto groupBytes       = _appContext.message_size * 2 * groupIterations;
        auto groupBandwidth   = groupBytes / (groupDuration / 1e3);
        auto totalBytes       = _appContext.message_size * 2 * (n + 1);
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
      parseBandwidth(_appContext.n_iter * _appContext.message_size * 2, total_duration_ns);

    // Stop progress thread
    if (_appContext.progress_mode == ProgressMode::ThreadBlocking ||
        _appContext.progress_mode == ProgressMode::ThreadPolling)
      _worker->stopProgressThread();
  }
};

int main(int argc, char** argv)
{
  ApplicationContext appContext;
  if (parseCommand(&appContext, argc, argv) != UCS_OK) return -1;

  auto app = Application(std::move(appContext));
  app.run();

  return 0;
}
