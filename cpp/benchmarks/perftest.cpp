/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <unistd.h>  // for getopt, optarg

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstring>
#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef UCXX_BENCHMARKS_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#include <ucxx/api.h>
#include <ucxx/utils/sockaddr.h>
#include <ucxx/utils/ucx.h>

#include "include/buffer_interface.hpp"

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

#ifdef UCXX_BENCHMARKS_ENABLE_CUDA
#define CUDA_EXIT_ON_ERROR(operation, context)                                                  \
  ([&]() {                                                                                      \
    cudaError_t err = operation;                                                                \
    if (err != cudaSuccess) {                                                                   \
      std::cerr << "CUDA error in " << context << " at " << __FILE__ << ":" << __LINE__ << ": " \
                << cudaGetErrorString(err) << std::endl;                                        \
      std::exit(-1);                                                                            \
    }                                                                                           \
    return err;                                                                                 \
  })()
#endif

enum class ProgressMode {
  Polling,
  Blocking,
  Wait,
  ThreadPolling,
  ThreadBlocking,
};

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
  double percentileRank                        = 50.0;
  bool endpointErrorHandling                   = false;
  bool reuseAllocations                        = true;
  bool verifyResults                           = false;
  MemoryType memoryType                        = MemoryType::Host;
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
#ifdef UCXX_BENCHMARKS_ENABLE_CUDA
  std::cerr << "  -m <type>   memory type to use, valid values are: 'host' (default), 'cuda', "
            << std::endl
            << "              'cuda-managed', and 'cuda-async'" << std::endl;
#else
  std::cerr << "  -m <type>   memory type to use, valid values are: 'host' (default)" << std::endl;
#endif
  std::cerr << "  -P <progress_mode>  worker progress mode to use" << std::endl;
  std::cerr << "           blocking - Blocking progress mode, equivalent to `ucx_perftest -E sleep`"
            << std::endl;
  std::cerr
    << "            polling - Polling progress mode, equivalent to `ucx_perftest -E poll` (default)"
    << std::endl;
  std::cerr << "    thread-blocking - Blocking progress mode in exclusive progress thread"
            << std::endl;
  std::cerr << "     thread-polling - Polling progress mode in exclusive progress thread"
            << std::endl;
  std::cerr << "  -p <port>           port number to listen at (12345)" << std::endl;
  std::cerr << "  -s <size>           message size (8)" << std::endl;
  std::cerr << "  -n <iters>          number of iterations to run (100)" << std::endl;
  std::cerr << "  -w <iters>          number of warmup iterations to run (3)" << std::endl;
  std::cerr << "  -L                  disable reuse memory allocation (enabled)" << std::endl;
  std::cerr << "  -e                  create endpoints with error handling support (disabled)"
            << std::endl;
  std::cerr << "  -v                  verify results (disabled)" << std::endl;
  std::cerr
    << "  -R <rank>           percentile rank of the percentile data in latency tests (50.0)"
    << std::endl;
  std::cerr << "  -h                  print this help" << std::endl;
  std::cerr << std::endl;
}

ucs_status_t parseCommand(ApplicationContext* appContext, int argc, char* const argv[])
{
  optind = 1;
  int c;
  while ((c = getopt(argc, argv, "t:m:P:p:s:n:w:LevR:h")) != -1) {
    switch (c) {
      case 't': {
        auto testAttributes = testAttributesDefinitions.find(optarg);
        if (testAttributes == testAttributesDefinitions.end()) {
          std::cerr << "Invalid test to run: " << optarg << std::endl;
          return UCS_ERR_INVALID_PARAM;
        }
        appContext->testAttributes = testAttributes->second;
      } break;
      case 'm':
        if (strcmp(optarg, "host") == 0) {
          appContext->memoryType = MemoryType::Host;
          break;
#ifdef UCXX_BENCHMARKS_ENABLE_CUDA
        } else if (strcmp(optarg, "cuda") == 0) {
          appContext->memoryType = MemoryType::Cuda;
          break;
        } else if (strcmp(optarg, "cuda-managed") == 0) {
          appContext->memoryType = MemoryType::CudaManaged;
          break;
        } else if (strcmp(optarg, "cuda-async") == 0) {
          appContext->memoryType = MemoryType::CudaAsync;
          break;
#endif
        } else {
          std::cerr << "Invalid memory type: " << optarg << std::endl;
          return UCS_ERR_INVALID_PARAM;
        }
      case 'P':
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
      case 'n':
        appContext->numIterations = atoi(optarg);
        if (appContext->numIterations <= 0) {
          std::cerr << "Wrong number of iterations: " << appContext->numIterations << std::endl;
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
      case 'L': appContext->reuseAllocations = false; break;
      case 'e': appContext->endpointErrorHandling = true; break;
      case 'v': appContext->verifyResults = true; break;
      case 'R':
        appContext->percentileRank = atof(optarg);
        if (appContext->percentileRank < 0.0 || appContext->percentileRank > 100.0) {
          std::cerr << "Wrong percentile rank: " << appContext->percentileRank << std::endl;
          return UCS_ERR_INVALID_PARAM;
        }
        break;
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
  std::queue<decltype(Result::duration)> _timingQueue{};
  const size_t _timingQueueSize{2048};

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

    if (_timingQueue.size() == _timingQueueSize) _timingQueue.pop();
    _timingQueue.push(duration);
  }

  double calculatePercentile(double percentile = 50.0)
  {
    if (percentile < 0.0 || percentile > 100.0) {
      throw std::invalid_argument("Percentile must be between 0.0 and 100.0");
    }

    std::vector<decltype(Result::duration)> timingVector;
    decltype(_timingQueue) tmpQueue;
    while (!_timingQueue.empty()) {
      auto duration = _timingQueue.front();
      _timingQueue.pop();
      timingVector.push_back(duration);
      tmpQueue.push(duration);
    }
    std::swap(tmpQueue, _timingQueue);

    std::sort(timingVector.begin(), timingVector.end());

    double index = (timingVector.size() - 1) * (percentile / 100.0);
    size_t lower = static_cast<size_t>(std::floor(index));
    if (index == static_cast<double>(lower)) {
      return timingVector[lower].count();
    } else {
      return (timingVector[lower] + timingVector[lower + 1]).count() / 2.0;
    }
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
    std::unique_ptr<BufferInterface> bufferInterface;

    // Allocate buffers based on memory type
    if (_appContext.memoryType == MemoryType::Host) {
      // Use the factory method to create the appropriate host buffer interface
      bufferInterface = HostBufferInterface::createBufferInterface(_appContext.messageSize,
                                                                   _appContext.reuseAllocations);
#ifdef UCXX_BENCHMARKS_ENABLE_CUDA
    } else {
      // Use the factory method to create the appropriate CUDA buffer interface
      bufferInterface = CudaBufferInterfaceBase::createBufferInterface(
        _appContext.memoryType, _appContext.messageSize, _appContext.reuseAllocations);
#endif
    }

    std::vector<std::shared_ptr<ucxx::Request>> requests;

    auto start = std::chrono::high_resolution_clock::now();
    if (_appContext.testAttributes->testType == TestType::PingPong) {
      requests = {
        _endpoint->tagSend(
          bufferInterface->getSendPtr(), _appContext.messageSize, (*_tagMap)[DirectionType::Send]),
        _endpoint->tagRecv(bufferInterface->getRecvPtr(),
                           _appContext.messageSize,
                           (*_tagMap)[DirectionType::Recv],
                           ucxx::TagMaskFull)};
    } else {
      if (_isServer)
        requests = {_endpoint->tagRecv(bufferInterface->getRecvPtr(),
                                       _appContext.messageSize,
                                       (*_tagMap)[DirectionType::Recv],
                                       ucxx::TagMaskFull)};
      else
        requests = {_endpoint->tagSend(
          bufferInterface->getSendPtr(), _appContext.messageSize, (*_tagMap)[DirectionType::Send])};
    }

    // Wait for requests and clear requests
    waitRequests(requests);
    auto stop = std::chrono::high_resolution_clock::now();

    if (_appContext.verifyResults) { bufferInterface->verifyResults(_appContext.messageSize); }

    return stop - start;
  }

  void printHeader(std::string_view description,
                   std::string_view category,
                   std::string_view sendMemory,
                   std::string_view recvMemory)
  {
    std::string categoryWithUnit = std::string(category) + std::string{" (usec)"};
    auto percentileRank          = floatToString(_appContext.percentileRank, 1);

    // clang-format off
    std::cout << "+--------------+--------------+------------------------------+---------------------+-----------------------+" << std::endl;
    std::cout << "|              |              | " << appendSpaces(categoryWithUnit, 28, true) << " |   bandwidth (MB/s)  |  message rate (msg/s) |" << std::endl;
    std::cout << "+----------------------------------------------------------------------------------------------------------+" << std::endl;
    std::cout << "+--------------+--------------+----------+---------+---------+----------+----------+-----------+-----------+" << std::endl;
    std::cout << "|    Stage     | # iterations | " << percentileRank << "%ile | average | overall |  average |  overall |  average  |  overall  |" << std::endl;
    std::cout << "+--------------+--------------+----------+---------+---------+----------+----------+-----------+-----------+" << std::endl;
    std::cout << "| Test:         " << appendSpaces(description) << "|" << std::endl;
    std::cout << "| Send memory:  " << appendSpaces(sendMemory) << "|" << std::endl;
    std::cout << "| Recv memory:  " << appendSpaces(recvMemory) << "|" << std::endl;
    std::cout << "| Message size: " << appendSpaces(std::to_string(_appContext.messageSize)) << "|" << std::endl;
    std::cout << "+----------------------------------------------------------------------------------------------------------+" << std::endl;
    // clang-format on
  }

  void printProgress(size_t iteration,
                     double percentile,
                     double overheadAverage,
                     double overheadOverall,
                     double bandwidthAverage,
                     double bandwidthOverall,
                     size_t messageRateAverage,
                     size_t messageRateOverall)
  {
    std::cout << "                " << appendSpaces(std::to_string(iteration), 15)
              << appendSpaces(floatToString(percentile, 3), 11)
              << appendSpaces(floatToString(overheadAverage, 3), 10)
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
#ifdef UCXX_BENCHMARKS_ENABLE_CUDA
    // Check CUDA support if CUDA memory is requested
    if (appContext.memoryType == MemoryType::Cuda ||
        appContext.memoryType == MemoryType::CudaManaged ||
        appContext.memoryType == MemoryType::CudaAsync) {
      CUDA_EXIT_ON_ERROR(cudaSetDevice(0), "CUDA device initialization");
    }
#endif

    if (!_isServer)
      printHeader(appContext.testAttributes->description,
                  appContext.testAttributes->category,
                  "host",
                  "host");

    // Setup: create UCP context, worker, listener and client endpoint.
    _context = UCXX_EXIT_ON_ERROR(ucxx::createContext({}, ucxx::Context::defaultFeatureFlags),
                                  "Context creation");
    _worker  = UCXX_EXIT_ON_ERROR(_context->createWorker(), "Worker creation");

    _tagMap = std::make_shared<TagMap>(TagMap{
      {DirectionType::Send, _isServer ? ucxx::Tag{0} : ucxx::Tag{1}},
      {DirectionType::Recv, _isServer ? ucxx::Tag{1} : ucxx::Tag{0}},
    });

    if (_isServer) {
      _listenerContext =
        std::make_unique<ListenerContext>(_worker, _appContext.endpointErrorHandling);
      _listener = UCXX_EXIT_ON_ERROR(
        _worker->createListener(_appContext.listenerPort, listenerCallback, _listenerContext.get()),
        "Listener creation");
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
      _endpoint = UCXX_EXIT_ON_ERROR(
        _worker->createEndpointFromHostname(
          _appContext.serverAddress, _appContext.listenerPort, _appContext.endpointErrorHandling),
        "Endpoint creation");
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
    UCXX_EXIT_ON_ERROR(doWireup(), "Wireup");

    if (_appContext.reuseAllocations) _bufferMapReuse = allocateTransferBuffers();

    // Warmup
    for (size_t n = 0; n < _appContext.numWarmupIterations; ++n)
      UCXX_EXIT_ON_ERROR(doTransfer(), "Warmup iteration " + std::to_string(n));

    auto lastPrintTime = std::chrono::steady_clock::now();

    Results results{};

    const double factor = (_appContext.testAttributes->testType == TestType::PingPong) ? 2.0 : 1.0;

    for (size_t n = 0; n < _appContext.numIterations; ++n) {
      auto duration = UCXX_EXIT_ON_ERROR(doTransfer(), "Transfer iteration " + std::to_string(n));

      results.update(duration, 1, _appContext.messageSize * factor, 1 * factor);

      auto currentTime = std::chrono::steady_clock::now();
      auto elapsedTime =
        std::chrono::duration_cast<std::chrono::seconds>(currentTime - lastPrintTime);

      if (!_isServer &&
          (elapsedTime.count() >= 1 || results.total.iterations == _appContext.numIterations)) {
        auto percentile = results.calculatePercentile() / factor * 1e6;
        auto currentLatency =
          results.current.duration.count() / results.current.iterations / factor * 1e6;
        auto totalLatency =
          results.total.duration.count() / results.total.iterations / factor * 1e6;
        auto currentBandwidth   = results.current.bytes / (results.current.duration.count() * 1e6);
        auto totalBandwidth     = results.total.bytes / (results.total.duration.count() * 1e6);
        auto currentMessageRate = results.current.messages / (results.current.duration.count());
        auto totalMessageRate   = results.total.messages / (results.total.duration.count());

        printProgress(results.total.iterations,
                      percentile,
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
