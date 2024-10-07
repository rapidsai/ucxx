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

enum class DirectionType { Send, Recv };

typedef std::unordered_map<DirectionType, std::vector<char>> BufferMap;
typedef std::unordered_map<DirectionType, ucxx::Tag> TagMap;

typedef std::shared_ptr<TagMap> TagMapPtr;

// Define an enum for each attribute that can be one of a fixed set of values
enum class CommandType { Tag, Undefined };
enum class TestType { PingPong, Unidirectional };

// Define a struct to hold the attributes of a test type
struct TestAttributes {
  CommandType commandType;
  TestType testType;
  std::string description;
  std::string category;
  int priority;  // Assume this is an integer for demonstration purposes

  // Constructor to initialize attributes
  TestAttributes(CommandType command,
                 TestType testType,
                 const std::string& description,
                 const std::string& category)
    : commandType(commandType), testType(testType), description(description), category(category)
  {
  }

  TestAttributes() = delete;
};

// Use a std::unordered_map to store instances of TestAttributes with name as the key
const std::unordered_map<std::string, TestAttributes> testAttributesDefinitions = {
  {"tag_lat", {CommandType::Tag, TestType::PingPong, "tag match latency", "latency"}},
  {"tag_bw", {CommandType::Tag, TestType::Unidirectional, "tag match bandwidth", "overhead"}},
};

struct ApplicationContext {
  ProgressMode progressMode                    = ProgressMode::Polling;
  std::optional<TestAttributes> testAttributes = std::nullopt;
  const char* serverAddress                    = NULL;
  uint16_t listenerPort                        = 12345;
  size_t messageSize                           = 8;
  size_t numIterations                         = 100;
  size_t numWarmupIterations                   = 3;
  bool endpointErrorHandling                   = false;
  bool reuseAllocations                        = false;
  bool verifyResults                           = false;
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

  void createEndpointFromConnRequest(ucp_conn_request_h connRequest)
  {
    if (!isAvailable()) throw std::runtime_error("Listener context already has an endpoint");

    _endpoint    = _listener->createEndpointFromConnRequest(connRequest, _endpointErrorHandling);
    _isAvailable = false;
  }

  void releaseEndpoint()
  {
    _endpoint.reset();
    _isAvailable = true;
  }
};

static void listenerCallback(ucp_conn_request_h connRequest, void* arg)
{
  char ipString[INET6_ADDRSTRLEN];
  char portString[INET6_ADDRSTRLEN];
  ucp_conn_request_attr_t attr{};
  ListenerContext* listenerContext = reinterpret_cast<ListenerContext*>(arg);

  attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR;
  ucxx::utils::ucsErrorThrow(ucp_conn_request_query(connRequest, &attr));
  ucxx::utils::sockaddr_get_ip_port_str(
    &attr.client_address, ipString, portString, INET6_ADDRSTRLEN);
  std::cout << "Server received a connection request from client at address " << ipString << ":"
            << portString << std::endl;

  if (listenerContext->isAvailable()) {
    listenerContext->createEndpointFromConnRequest(connRequest);
  } else {
    // The server is already handling a connection request from a client,
    // reject this new one
    std::cout << "Rejecting a connection request from " << ipString << ":" << portString << "."
              << std::endl
              << "Only one client at a time is supported." << std::endl;
    ucxx::utils::ucsErrorThrow(
      ucp_listener_reject(listenerContext->getListener()->getHandle(), connRequest));
  }
}

static void printUsage(std::string_view executablePath)
{
  std::cerr << "UCXX performance testing tool" << std::endl;
  std::cerr << std::endl;
  std::cerr << "Usage: " << executablePath << " [server-hostname] [options]" << std::endl;
  std::cerr << std::endl;
  std::cerr << "Parameters are:" << std::endl;
  std::cerr << "  -t <test>           test to run (required)" << std::endl;
  std::cerr << "            tag_lat - UCP tag match latency" << std::endl;
  std::cerr << "             tag_bw - UCP tag match bandwidth" << std::endl;
  std::cerr << "  -m <progress_mode>  worker progress mode to use" << std::endl;
  std::cerr << "           blocking - Blocking progress mode, equivalent to `ucx_perftest -E sleep`"
            << std::endl;
  std::cerr
    << "            polling - Polling progress mode, equivalent to `ucx_perftest -E poll` (default)"
    << std::endl;
  std::cerr << "    thread-blocking - Blocking progress mode in exclusive progress thread"
            << std::endl;
  std::cerr << "     thread-polling - Polling progress mode in exclusive progress thread"
            << std::endl;
  std::cerr << "  -e                  create endpoints with error handling support (disabled)"
            << std::endl;
  std::cerr << "  -p <port>           port number to listen at (12345)" << std::endl;
  std::cerr << "  -s <bytes>          message size (8)" << std::endl;
  std::cerr << "  -n <int>            number of iterations to run (100)" << std::endl;
  std::cerr << "  -r                  reuse memory allocation (disabled)" << std::endl;
  std::cerr << "  -v                  verify results (disabled)" << std::endl;
  std::cerr << "  -w <int>            number of warmup iterations to run (3)" << std::endl;
  std::cerr << "  -h                  print this help" << std::endl;
  std::cerr << std::endl;
}

ucs_status_t parseCommand(ApplicationContext* appContext, int argc, char* const argv[])
{
  optind = 1;
  int c;
  while ((c = getopt(argc, argv, "m:t:p:s:w:n:ervh")) != -1) {
    switch (c) {
      case 'm':
        if (strcmp(optarg, "blocking") == 0) {
          appContext->progressMode = ProgressMode::Blocking;
          break;
        } else if (strcmp(optarg, "polling") == 0) {
          appContext->progressMode = ProgressMode::Polling;
          break;
        } else if (strcmp(optarg, "thread-blocking") == 0) {
          appContext->progressMode = ProgressMode::ThreadBlocking;
          break;
        } else if (strcmp(optarg, "thread-polling") == 0) {
          appContext->progressMode = ProgressMode::ThreadPolling;
          break;
        } else if (strcmp(optarg, "wait") == 0) {
          appContext->progressMode = ProgressMode::Wait;
          break;
        } else {
          std::cerr << "Invalid progress mode: " << optarg << std::endl;
          return UCS_ERR_INVALID_PARAM;
        }
      case 't': {
        auto testAttributes = testAttributesDefinitions.find(optarg);
        if (testAttributes == testAttributesDefinitions.end()) {
          std::cerr << "Invalid test to run: " << optarg << std::endl;
          return UCS_ERR_INVALID_PARAM;
        }
        appContext->testAttributes = testAttributes->second;
      } break;
      case 'p':
        appContext->listenerPort = atoi(optarg);
        if (appContext->listenerPort <= 0) {
          std::cerr << "Wrong listener port: " << appContext->listenerPort << std::endl;
          return UCS_ERR_INVALID_PARAM;
        }
        break;
      case 's':
        appContext->messageSize = atoi(optarg);
        if (appContext->messageSize <= 0) {
          std::cerr << "Wrong message size: " << appContext->messageSize << std::endl;
          return UCS_ERR_INVALID_PARAM;
        }
        break;
      case 'w':
        appContext->numWarmupIterations = atoi(optarg);
        if (appContext->numWarmupIterations <= 0) {
          std::cerr << "Wrong number of warmup iterations: " << appContext->numWarmupIterations
                    << std::endl;
          return UCS_ERR_INVALID_PARAM;
        }
        break;
      case 'n':
        appContext->numIterations = atoi(optarg);
        if (appContext->numIterations <= 0) {
          std::cerr << "Wrong number of iterations: " << appContext->numIterations << std::endl;
          return UCS_ERR_INVALID_PARAM;
        }
        break;
      case 'e': appContext->endpointErrorHandling = true; break;
      case 'r': appContext->reuseAllocations = true; break;
      case 'v': appContext->verifyResults = true; break;
      case 'h':
      default: printUsage(std::string_view(argv[0])); return UCS_ERR_INVALID_PARAM;
    }
  }

  if (!appContext->testAttributes.has_value()) {
    std::cerr << "missing test to run (-t)" << std::endl;
    return UCS_ERR_INVALID_PARAM;
  }

  if (optind < argc) { appContext->serverAddress = argv[optind]; }

  return UCS_OK;
}

std::string appendSpaces(const std::string_view input,
                         const int maxLength = 91,
                         const bool bothEnds = false)
{
  int spaces = std::max(0, maxLength - static_cast<int>(input.length()));
  if (bothEnds) {
    int prefix = spaces / 2;
    int suffix = spaces / 2 + spaces % 2;
    return std::string(prefix, ' ') + std::string(input) + std::string(suffix, ' ');
  } else {
    return std::string(input) + std::string(spaces, ' ');
  }
}

std::string floatToString(double number, size_t precision = 2)
{
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision) << number;
  return oss.str();
}

struct Result {
  std::chrono::duration<double> duration{0};
  size_t iterations{0};
  size_t bytes{0};
  size_t messages{0};
};

struct Results {
  Result total{};
  Result current{};

  void update(decltype(Result::duration) duration,
              decltype(Result::iterations) iterations,
              decltype(Result::bytes) bytes,
              decltype(Result::messages) messages)
  {
    total.duration += duration;
    current.duration += duration;
    total.iterations += iterations;
    current.iterations += iterations;
    total.bytes += bytes;
    current.bytes += bytes;
    total.messages += messages;
    current.messages += messages;
  }

  void resetCurrent() { current = Result{}; }
};

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

  std::function<void()> getProgressFunction()
  {
    switch (_appContext.progressMode) {
      case ProgressMode::Polling: return std::bind(std::mem_fn(&ucxx::Worker::progress), _worker);
      case ProgressMode::Blocking:
        return std::bind(std::mem_fn(&ucxx::Worker::progressWorkerEvent), _worker, -1);
      case ProgressMode::Wait: return std::bind(std::mem_fn(&ucxx::Worker::waitProgress), _worker);
      default: return []() {};
    }
  }

  void waitRequests(const std::vector<std::shared_ptr<ucxx::Request>>& requests)
  {
    auto progress = getProgressFunction();
    // Wait until all messages are completed
    for (auto& r : requests) {
      while (!r->isCompleted())
        progress();
      r->checkError();
    }
  }

  BufferMap allocateTransferBuffers()
  {
    return BufferMap{{DirectionType::Send, std::vector<char>(_appContext.messageSize, 0xaa)},
                     {DirectionType::Recv, std::vector<char>(_appContext.messageSize)}};
  }

  void doWireup()
  {
    std::vector<std::shared_ptr<ucxx::Request>> requests;

    // Allocate wireup buffers
    auto wireupBufferMap =
      std::make_shared<BufferMap>(BufferMap{{DirectionType::Send, std::vector<char>{1, 2, 3}},
                                            {DirectionType::Recv, std::vector<char>(3, 0)}});

    // Schedule small wireup messages to let UCX identify capabilities between endpoints
    requests.push_back(
      _endpoint->tagSend((*wireupBufferMap)[DirectionType::Send].data(),
                         (*wireupBufferMap)[DirectionType::Send].size() * sizeof(int),
                         (*_tagMap)[DirectionType::Send]));
    requests.push_back(
      _endpoint->tagRecv((*wireupBufferMap)[DirectionType::Recv].data(),
                         (*wireupBufferMap)[DirectionType::Recv].size() * sizeof(int),
                         (*_tagMap)[DirectionType::Recv],
                         ucxx::TagMaskFull));

    // Wait for wireup requests and clear requests
    waitRequests(requests);

    // Verify wireup result
    for (size_t i = 0; i < (*wireupBufferMap)[DirectionType::Send].size(); ++i)
      assert((*wireupBufferMap)[DirectionType::Recv][i] ==
             (*wireupBufferMap)[DirectionType::Send][i]);
  }

  auto doTransfer()
  {
    BufferMap localBufferMap;
    if (!_appContext.reuseAllocations) localBufferMap = allocateTransferBuffers();
    BufferMap& bufferMap = _appContext.reuseAllocations ? _bufferMapReuse : localBufferMap;

    std::vector<std::shared_ptr<ucxx::Request>> requests;

    auto start = std::chrono::high_resolution_clock::now();
    if (_appContext.testAttributes->testType == TestType::PingPong) {
      requests = {_endpoint->tagSend((bufferMap)[DirectionType::Send].data(),
                                     _appContext.messageSize,
                                     (*_tagMap)[DirectionType::Send]),
                  _endpoint->tagRecv((bufferMap)[DirectionType::Recv].data(),
                                     _appContext.messageSize,
                                     (*_tagMap)[DirectionType::Recv],
                                     ucxx::TagMaskFull)};
    } else {
      if (_isServer)
        requests = {_endpoint->tagRecv((bufferMap)[DirectionType::Recv].data(),
                                       _appContext.messageSize,
                                       (*_tagMap)[DirectionType::Recv],
                                       ucxx::TagMaskFull)};
      else
        requests = {_endpoint->tagSend((bufferMap)[DirectionType::Send].data(),
                                       _appContext.messageSize,
                                       (*_tagMap)[DirectionType::Send])};
    }

    // Wait for requests and clear requests
    waitRequests(requests);
    auto stop = std::chrono::high_resolution_clock::now();

    if (_appContext.verifyResults) {
      for (size_t j = 0; j < (bufferMap)[DirectionType::Send].size(); ++j)
        assert((bufferMap)[DirectionType::Recv][j] == (bufferMap)[DirectionType::Recv][j]);
    }

    return stop - start;
  }

  void printHeader(std::string_view description,
                   std::string_view category,
                   std::string_view sendMemory,
                   std::string_view recvMemory)
  {
    std::string categoryWithUnit = std::string(category) + std::string{" (usec)"};

    // clang-format off
    std::cout << "+--------------+--------------+------------------------------+---------------------+-----------------------+" << std::endl;
    std::cout << "|              |              | " << appendSpaces(categoryWithUnit, 28, true) << " |   bandwidth (MB/s)  |  message rate (msg/s) |" << std::endl;
    std::cout << "+----------------------------------------------------------------------------------------------------------+" << std::endl;
    std::cout << "+--------------+--------------+----------+---------+---------+----------+----------+-----------+-----------+" << std::endl;
    std::cout << "|    Stage     | # iterations | 50.0%ile | average | overall |  average |  overall |  average  |  overall  |" << std::endl;
    std::cout << "+--------------+--------------+----------+---------+---------+----------+----------+-----------+-----------+" << std::endl;
    std::cout << "| Test:         " << appendSpaces(description) << "|" << std::endl;
    std::cout << "| Send memory:  " << appendSpaces(sendMemory) << "|" << std::endl;
    std::cout << "| Recv memory:  " << appendSpaces(recvMemory) << "|" << std::endl;
    std::cout << "| Message size: " << appendSpaces(std::to_string(_appContext.messageSize)) << "|" << std::endl;
    std::cout << "+----------------------------------------------------------------------------------------------------------+" << std::endl;
    // clang-format on
  }

  void printProgress(size_t iteration,
                     double overhead50,
                     double overheadAverage,
                     double overheadOverall,
                     double bandwidthAverage,
                     double bandwidthOverall,
                     size_t messageRateAverage,
                     size_t messageRateOverall)
  {
    std::cout << "                " << appendSpaces(std::to_string(iteration), 15)
              << appendSpaces("N/A", 11) << appendSpaces(floatToString(overheadAverage, 3), 10)
              << appendSpaces(floatToString(overheadOverall, 3), 10)
              << appendSpaces(floatToString(bandwidthAverage), 11)
              << appendSpaces(floatToString(bandwidthOverall), 11)
              << appendSpaces(std::to_string(messageRateAverage), 12)
              << appendSpaces(std::to_string(messageRateOverall), 0) << std::endl;
  }

 public:
  explicit Application(ApplicationContext&& appContext)
    : _appContext(appContext), _isServer(appContext.serverAddress == NULL)
  {
    if (!_isServer)
      printHeader(appContext.testAttributes->description,
                  appContext.testAttributes->category,
                  "host",
                  "host");

    // Setup: create UCP context, worker, listener and client endpoint.
    _context = ucxx::createContext({}, ucxx::Context::defaultFeatureFlags);
    _worker  = _context->createWorker();

    _tagMap = std::make_shared<TagMap>(TagMap{
      {DirectionType::Send, _isServer ? ucxx::Tag{0} : ucxx::Tag{1}},
      {DirectionType::Recv, _isServer ? ucxx::Tag{1} : ucxx::Tag{0}},
    });

    if (_isServer) {
      _listenerContext =
        std::make_unique<ListenerContext>(_worker, _appContext.endpointErrorHandling);
      _listener =
        _worker->createListener(_appContext.listenerPort, listenerCallback, _listenerContext.get());
      _listenerContext->setListener(_listener);
    }

    // Initialize worker progress
    if (_appContext.progressMode == ProgressMode::Blocking)
      _worker->initBlockingProgressMode();
    else if (_appContext.progressMode == ProgressMode::ThreadBlocking)
      _worker->startProgressThread(false);
    else if (_appContext.progressMode == ProgressMode::ThreadPolling)
      _worker->startProgressThread(true);

    auto progress = getProgressFunction();

    // Block until client connects
    while (_isServer && _listenerContext->isAvailable())
      progress();

    if (_isServer)
      _endpoint = _listenerContext->getEndpoint();
    else
      _endpoint = _worker->createEndpointFromHostname(
        _appContext.serverAddress, _appContext.listenerPort, _appContext.endpointErrorHandling);

    std::vector<std::shared_ptr<ucxx::Request>> requests;
  }

  ~Application()
  {
    // Stop progress thread
    if (_appContext.progressMode == ProgressMode::ThreadBlocking ||
        _appContext.progressMode == ProgressMode::ThreadPolling)
      _worker->stopProgressThread();
  }

  void run()
  {
    // Do wireup
    doWireup();

    if (_appContext.reuseAllocations) _bufferMapReuse = allocateTransferBuffers();

    // Warmup
    for (size_t n = 0; n < _appContext.numWarmupIterations; ++n)
      doTransfer();

    auto lastPrintTime = std::chrono::steady_clock::now();

    Results results{};

    const double factor = (_appContext.testAttributes->testType == TestType::PingPong) ? 2.0 : 1.0;

    for (size_t n = 0; n < _appContext.numIterations; ++n) {
      auto duration = doTransfer();

      results.update(duration, 1, _appContext.messageSize * factor, 1 * factor);

      auto currentTime = std::chrono::steady_clock::now();
      auto elapsedTime =
        std::chrono::duration_cast<std::chrono::seconds>(currentTime - lastPrintTime);

      if (!_isServer &&
          (elapsedTime.count() >= 1 || results.total.iterations == _appContext.numIterations)) {
        auto currentLatency =
          results.current.duration.count() / results.current.iterations / factor * 1e6;
        auto totalLatency =
          results.total.duration.count() / results.total.iterations / factor * 1e6;
        auto currentBandwidth   = results.current.bytes / (results.current.duration.count() * 1e6);
        auto totalBandwidth     = results.total.bytes / (results.total.duration.count() * 1e6);
        auto currentMessageRate = results.current.messages / (results.current.duration.count());
        auto totalMessageRate   = results.total.messages / (results.total.duration.count());

        printProgress(results.total.iterations,
                      0.0f,
                      currentLatency,
                      totalLatency,
                      currentBandwidth,
                      totalBandwidth,
                      currentMessageRate,
                      totalMessageRate);

        results.resetCurrent();
        lastPrintTime = currentTime;
      }
    }
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
