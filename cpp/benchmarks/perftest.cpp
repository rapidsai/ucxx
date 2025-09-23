/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <unistd.h>  // for getopt, optarg

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
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

// Define an enum for each attribute that can be one of a fixed set of values
enum class CommandType { Tag, Undefined };
enum class TestType { PingPong, Unidirectional };

// Define a struct to hold the attributes of a test type
struct TestAttributes {
  CommandType commandType;
  TestType testType;
  std::string description;
  std::string category;

  /**
   * @brief Constructs a TestAttributes instance with the specified command type, test type,
   * description, and category.
   *
   * @param commandType The type of command associated with the test.
   * @param testType The type of test (e.g., ping-pong or unidirectional).
   * @param description A human-readable description of the test.
   * @param category The category to which the test belongs.
   */
  TestAttributes(CommandType commandType,
                 TestType testType,
                 std::string description,
                 std::string category)
    : commandType(commandType),
      testType(testType),
      description(std::move(description)),
      category(std::move(category))
  {
  }

  /**
   * @brief Deleted default constructor for TestAttributes.
   *
   * Prevents creation of TestAttributes without specifying all required fields.
   */
  TestAttributes() = delete;
};

// Use a std::unordered_map to store instances of TestAttributes with name as the key
const std::unordered_map<std::string, TestAttributes> testAttributesDefinitions = {
  {"tag_lat",
   {CommandType::Tag, TestType::PingPong, std::move("tag match latency"), std::move("latency")}},
  {"tag_bw",
   {CommandType::Tag,
    TestType::Unidirectional,
    std::move("tag match bandwidth"),
    std::move("overhead")}},
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

  /**
   * @brief Checks if the listener context is available to accept a new connection.
   *
   * @return true if the listener context is available; false otherwise.
   */
  bool isAvailable() const { return _isAvailable; }

  /**
   * @brief Creates an endpoint from a UCX connection request if the listener is available.
   *
   * Associates a new endpoint with the listener context using the provided UCX connection request.
   *
   * @param connRequest The UCX connection request handle.
   *
   * @throws std::runtime_error if an endpoint is already present.
   */
  void createEndpointFromConnRequest(ucp_conn_request_h connRequest)
  {
    if (!isAvailable()) throw std::runtime_error("Listener context already has an endpoint");

    _endpoint    = _listener->createEndpointFromConnRequest(connRequest, _endpointErrorHandling);
    _isAvailable = false;
  }

  /**
   * @brief Releases the current endpoint and marks the listener context as available for new
   * connections.
   */
  void releaseEndpoint()
  {
    _endpoint.reset();
    _isAvailable = true;
  }
};

/**
 * @brief Handles incoming connection requests on the server listener.
 *
 * Queries the client's address and prints connection information. If the server is available,
 * creates an endpoint for the new connection; otherwise, rejects the request and notifies that
 * only one client at a time is supported.
 *
 * @param connRequest The UCX connection request handle.
 * @param arg Pointer to the ListenerContext managing listener state.
 */
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
  std::cout << "Accepted connection from " << ipString << ":" << portString << std::endl;

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

/**
 * @brief Prints usage instructions and available command-line options for the UCXX performance
 * testing tool.
 *
 * @param executablePath The path or name of the executable, used in the usage message.
 */
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
  std::cerr << "               wait - Wait progress mode" << std::endl;
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

/**
 * @brief Parses a string as an unsigned long long integer with range and error checking.
 *
 * Converts the input string to an unsigned long long, validates that it is within the specified
 * range, and stores the result.
 *
 * @param arg Input string to parse.
 * @param result Pointer to store the parsed value.
 * @param paramName Name of the parameter for error reporting.
 * @param minValue Minimum allowed value (inclusive).
 * @param maxValue Maximum allowed value (inclusive).
 * @return UCS_OK on success, UCS_ERR_INVALID_PARAM on failure.
 */
static ucs_status_t parseUnsignedLongLong(const char* arg,
                                          size_t* result,
                                          const char* paramName,
                                          size_t minValue = 1,
                                          size_t maxValue = SIZE_MAX)
{
  char* endptr;
  errno         = 0;
  uint64_t temp = std::strtoull(arg, &endptr, 10);
  if (errno != 0 || *endptr != '\0' || temp < minValue || temp > maxValue) {
    std::cerr << "Invalid " << paramName << ": " << arg << std::endl;
    return UCS_ERR_INVALID_PARAM;
  }
  *result = static_cast<size_t>(temp);
  return UCS_OK;
}

/**
 * @brief Parses a string as a double with range and error checking.
 *
 * Attempts to convert the input string to a double, ensuring the value is within the specified
 * range and that the entire string is a valid number. Prints an error message if parsing fails or
 * the value is out of bounds.
 *
 * @param arg The input string to parse.
 * @param result Pointer to store the parsed double value.
 * @param paramName Name of the parameter for error reporting.
 * @param minValue Minimum allowed value (inclusive).
 * @param maxValue Maximum allowed value (inclusive).
 * @return UCS_OK on success, UCS_ERR_INVALID_PARAM on failure.
 */
static ucs_status_t parseDouble(const char* arg,
                                double* result,
                                const char* paramName,
                                double minValue = -std::numeric_limits<double>::infinity(),
                                double maxValue = std::numeric_limits<double>::infinity())
{
  char* endptr;
  errno   = 0;
  *result = std::strtod(arg, &endptr);
  if (errno != 0 || *endptr != '\0' || *result < minValue || *result > maxValue) {
    std::cerr << "Invalid " << paramName << ": " << arg << std::endl;
    return UCS_ERR_INVALID_PARAM;
  }
  return UCS_OK;
}

/**
 * @brief Parses a string as a port number with validation.
 *
 * Converts the input string to a uint16_t port number, ensuring it is within the valid range (1 to
 * 65535). Prints an error message and returns UCS_ERR_INVALID_PARAM if parsing fails or the value
 * is out of range.
 *
 * @param arg String representation of the port number.
 * @param result Pointer to store the parsed port number on success.
 * @param paramName Name of the parameter for error reporting.
 * @return UCS_OK on success, UCS_ERR_INVALID_PARAM on failure.
 */
static ucs_status_t parsePort(const char* arg, uint16_t* result, const char* paramName)
{
  char* endptr;
  errno         = 0;
  uint64_t temp = std::strtoull(arg, &endptr, 10);
  if (errno != 0 || *endptr != '\0' || temp == 0 || temp > UINT16_MAX) {
    std::cerr << "Invalid " << paramName << ": " << arg << std::endl;
    return UCS_ERR_INVALID_PARAM;
  }
  *result = static_cast<uint16_t>(temp);
  return UCS_OK;
}

/**
 * @brief Parses command-line arguments and populates the application context.
 *
 * Processes command-line options to configure the test type, memory type, progress mode, port,
 * message size, iteration counts, percentile rank, and other settings in the provided
 * ApplicationContext. Validates input values and prints usage or error messages on invalid input.
 *
 * @param appContext Pointer to the ApplicationContext to populate.
 * @param argc Argument count from main().
 * @param argv Argument vector from main().
 * @return UCS_OK on success, UCS_ERR_INVALID_PARAM on invalid input or if required options are
 * missing.
 */
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
      case 'm': try { appContext->memoryType = getMemoryTypeFromString(optarg);
        } catch (const std::runtime_error& e) {
          std::cerr << "Invalid memory type: " << optarg << std::endl;
          return UCS_ERR_INVALID_PARAM;
        }
        break;
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
      case 'p': {
        ucs_status_t status = parsePort(optarg, &appContext->listenerPort, "listener port");
        if (status != UCS_OK) return status;
        break;
      }
      case 's': {
        ucs_status_t status =
          parseUnsignedLongLong(optarg, &appContext->messageSize, "message size");
        if (status != UCS_OK) return status;
        break;
      }
      case 'n': {
        ucs_status_t status =
          parseUnsignedLongLong(optarg, &appContext->numIterations, "number of iterations");
        if (status != UCS_OK) return status;
        break;
      }
      case 'w': {
        ucs_status_t status = parseUnsignedLongLong(
          optarg, &appContext->numWarmupIterations, "number of warmup iterations");
        if (status != UCS_OK) return status;
        break;
      }
      case 'L': appContext->reuseAllocations = false; break;
      case 'e': appContext->endpointErrorHandling = true; break;
      case 'v': appContext->verifyResults = true; break;
      case 'R': {
        ucs_status_t status =
          parseDouble(optarg, &appContext->percentileRank, "percentile rank", 0.0, 100.0);
        if (status != UCS_OK) return status;
        break;
      }
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

/**
 * @brief Pads a string with spaces to reach a specified length.
 *
 * If `bothEnds` is true, spaces are added evenly to both sides of the input string; otherwise,
 * spaces are appended to the end. If the input string is longer than `maxLength`, it is returned
 * unchanged.
 *
 * @param input The string to pad.
 * @param maxLength The desired total length after padding.
 * @param bothEnds Whether to pad spaces on both sides (true) or only at the end (false).
 * @return The padded string.
 */
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

/**
 * @brief Converts a floating-point number to a string with fixed decimal precision.
 *
 * @param number The floating-point value to convert.
 * @param precision The number of digits after the decimal point (default is 2).
 * @return std::string The formatted string representation of the number.
 */
std::string floatToString(double number, std::optional<size_t> precisionOverride = std::nullopt)
{
  std::ostringstream oss;
  size_t precision;

  if (precisionOverride) {
    precision = *precisionOverride;
  } else {
    if (number < 10.0) {
      precision = 5;
    } else if (number < 100.0) {
      precision = 4;
    } else if (number < 1000.0) {
      precision = 3;
    } else if (number < 10000.0) {
      precision = 2;
    } else if (number < 100000.0) {
      precision = 1;
    } else {
      precision = 0;
    }
  }

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

  /**
   * @brief Updates the total and current performance metrics with new results.
   *
   * Adds the provided duration, iterations, bytes, and messages to both the total and current
   * results. Maintains a fixed-size queue of recent durations for percentile calculations.
   *
   * @param duration Duration of the operation to add.
   * @param iterations Number of iterations to add.
   * @param bytes Number of bytes transferred to add.
   * @param messages Number of messages processed to add.
   */
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

  /**
   * @brief Calculates the specified percentile latency from the recorded timing queue.
   *
   * Computes the latency value at the given percentile (e.g., median for 50.0) from the most recent
   * recorded durations. If no timings are recorded, returns 0.0.
   *
   * @param percentile The percentile to compute (0.0 to 100.0, inclusive).
   * @return double The latency value at the specified percentile, in seconds.
   *
   * @throws std::invalid_argument If the percentile is outside the range [0.0, 100.0].
   */
  double calculatePercentile(double percentile = 50.0)
  {
    if (percentile < 0.0 || percentile > 100.0) {
      throw std::invalid_argument("Percentile must be between 0.0 and 100.0");
    }

    if (_timingQueue.empty()) {
      return 0.0;  // Nothing recorded yet
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

  /**
   * @brief Resets the current result metrics to their default values.
   */
  void resetCurrent() { current = Result{}; }
};

/**
 * @brief Performs message transfer according to the configured test type and memory settings.
 *
 * Allocates send and receive buffers based on the selected memory type (host or CUDA), posts tag
 * send and/or receive operations depending on whether the test is PingPong or Unidirectional, waits
 * for completion, and optionally verifies the received data.
 *
 * @return Duration of the transfer operation.
 */
class Application {
 private:
  ApplicationContext _appContext{};
  bool _isServer{false};
  std::shared_ptr<ucxx::Context> _context{nullptr};
  std::shared_ptr<ucxx::Worker> _worker{nullptr};
  std::shared_ptr<ucxx::Endpoint> _endpoint{nullptr};
  std::shared_ptr<ucxx::Listener> _listener{nullptr};
  std::unique_ptr<ListenerContext> _listenerContext{nullptr};
  std::shared_ptr<TagMap> _tagMap{nullptr};

  std::function<void()> getProgressFunction()
  {
    switch (_appContext.progressMode) {
      case ProgressMode::Polling: return std::bind(std::mem_fn(&ucxx::Worker::progress), _worker);
      case ProgressMode::Blocking:
        return std::bind(std::mem_fn(&ucxx::Worker::progressWorkerEvent), _worker, -1);
      case ProgressMode::Wait: return std::bind(std::mem_fn(&ucxx::Worker::waitProgress), _worker);
      case ProgressMode::ThreadBlocking:  // progress thread already running
      case ProgressMode::ThreadPolling:   // progress thread already running
      default: return []() { std::this_thread::yield(); };
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

  void doWireup()
  {
    std::vector<std::shared_ptr<ucxx::Request>> requests;

    // Allocate wireup buffers
    auto wireupBufferMap =
      std::make_shared<BufferMap>(BufferMap{{DirectionType::Send, std::vector<char>{1, 2, 3}},
                                            {DirectionType::Recv, std::vector<char>(3, 0)}});

    // Schedule small wireup messages to let UCX identify capabilities between endpoints
    requests.push_back(_endpoint->tagSend((*wireupBufferMap)[DirectionType::Send].data(),
                                          (*wireupBufferMap)[DirectionType::Send].size(),
                                          (*_tagMap)[DirectionType::Send]));
    requests.push_back(_endpoint->tagRecv((*wireupBufferMap)[DirectionType::Recv].data(),
                                          (*wireupBufferMap)[DirectionType::Recv].size(),
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
#else
    } else {
      throw std::runtime_error("Memory type not supported.");
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

  void printServerHeader(std::string_view description, MemoryType sendMemory, MemoryType recvMemory)
  {
    // clang-format off
    std::cout << "+----------------------------------------------------------------------------------------------------------+" << std::endl;
    std::cout << "| Test:         " << appendSpaces(description) << "|" << std::endl;
    std::cout << "| Send memory:  " << appendSpaces(getMemoryTypeString(sendMemory)) << "|" << std::endl;
    std::cout << "| Recv memory:  " << appendSpaces(getMemoryTypeString(recvMemory)) << "|" << std::endl;
    std::cout << "| Message size: " << appendSpaces(std::to_string(_appContext.messageSize)) << "|" << std::endl;
    std::cout << "+----------------------------------------------------------------------------------------------------------+" << std::endl;
    // clang-format on
  }

  /**
   * @brief Prints the formatted client-side test header for performance results.
   *
   * Displays column headers for latency percentiles, bandwidth, and message rate, including the
   * specified percentile rank and test category.
   *
   * @param category The name of the latency metric or test category to display in the header.
   */
  void printClientHeader(std::string_view category)
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
    // clang-format on
  }

  /**
   * @brief Prints a formatted progress line displaying current test metrics.
   *
   * Outputs the current iteration, percentile latency, average and overall latency, bandwidth, and
   * message rate in aligned columns for progress reporting during performance tests.
   *
   * @param iteration Current test iteration.
   * @param percentile Calculated latency percentile value.
   * @param overheadAverage Average latency for the current interval (microseconds).
   * @param overheadOverall Overall average latency (microseconds).
   * @param bandwidthAverage Average bandwidth for the current interval (GB/s).
   * @param bandwidthOverall Overall average bandwidth (GB/s).
   * @param messageRateAverage Average message rate for the current interval (messages/s).
   * @param messageRateOverall Overall average message rate (messages/s).
   */
  void printProgress(size_t iteration,
                     double percentile,
                     double overheadAverage,
                     double overheadOverall,
                     double bandwidthAverage,
                     double bandwidthOverall,
                     double messageRateAverage,
                     double messageRateOverall)
  {
    std::cout << "                " << appendSpaces(std::to_string(iteration), 15)
              << appendSpaces(floatToString(percentile, 3), 11)
              << appendSpaces(floatToString(overheadAverage, 3), 10)
              << appendSpaces(floatToString(overheadOverall, 3), 10)
              << appendSpaces(floatToString(bandwidthAverage), 11)
              << appendSpaces(floatToString(bandwidthOverall), 11)
              << appendSpaces(floatToString(messageRateAverage), 12)
              << appendSpaces(floatToString(messageRateOverall), 0) << std::endl;
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

    // Setup: create UCP context, worker, listener and client endpoint.
    uint64_t ucpFeatures = UCP_FEATURE_TAG;
    if (appContext.progressMode == ProgressMode::Blocking ||
        appContext.progressMode == ProgressMode::ThreadBlocking ||
        appContext.progressMode == ProgressMode::Wait) {
      ucpFeatures |= UCP_FEATURE_WAKEUP;
    }
    _context = UCXX_EXIT_ON_ERROR(ucxx::createContext({}, ucpFeatures), "Context creation");
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

    if (_isServer) {
      std::cout << "Waiting for connection..." << std::endl;

      // Block until client connects
      while (_listenerContext->isAvailable())
        progress();
    }

    if (_isServer) {
      _endpoint = _listenerContext->getEndpoint();
      printServerHeader(
        appContext.testAttributes->description, appContext.memoryType, appContext.memoryType);
    } else {
      _endpoint = UCXX_EXIT_ON_ERROR(
        _worker->createEndpointFromHostname(
          _appContext.serverAddress, _appContext.listenerPort, _appContext.endpointErrorHandling),
        "Endpoint creation");
      printClientHeader(appContext.testAttributes->category);
    }
  }

  /**
   * @brief Destructor for the Application class.
   *
   * Stops the worker's progress thread if the configured progress mode uses a background thread.
   */
  ~Application()
  {
    // Stop progress thread
    if (_appContext.progressMode == ProgressMode::ThreadBlocking ||
        _appContext.progressMode == ProgressMode::ThreadPolling)
      _worker->stopProgressThread();
  }

  /**
   * @brief Executes the main performance test loop.
   *
   * Performs initial endpoint wireup, runs warmup transfers, and then executes the configured
   * number of test iterations. During the test, it collects and updates performance metrics,
   * periodically printing progress statistics such as latency percentiles, average latency,
   * bandwidth, and message rate.
   */
  void run()
  {
    // Do wireup
    UCXX_EXIT_ON_ERROR(doWireup(), "Wireup");

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
        const auto percentile =
          results.calculatePercentile(_appContext.percentileRank) / factor * 1e6;
        const auto currentLatency =
          results.current.duration.count() / results.current.iterations / factor * 1e6;
        const auto totalLatency =
          results.total.duration.count() / results.total.iterations / factor * 1e6;

        const auto curSec           = results.current.duration.count();
        const auto totalSec         = results.total.duration.count();
        const auto currentBandwidth = (curSec > 0.0) ? results.current.bytes / (curSec * 1e6) : 0.0;
        const auto totalBandwidth = (totalSec > 0.0) ? results.total.bytes / (totalSec * 1e6) : 0.0;
        const auto currentMessageRate = (curSec > 0.0) ? results.current.messages / curSec : 0.0;
        const auto totalMessageRate   = (totalSec > 0.0) ? results.total.messages / totalSec : 0.0;

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

/**
 * @brief Entry point for the UCXX performance test application.
 *
 * Parses command-line arguments, initializes the application context, and runs the selected UCX tag
 * matching performance test as either server or client.
 *
 * @return int Returns 0 on success, or -1 if argument parsing fails.
 */
int main(int argc, char** argv)
{
  ApplicationContext appContext;
  if (parseCommand(&appContext, argc, argv) != UCS_OK) return -1;

  auto app = Application(std::move(appContext));
  app.run();

  return 0;
}
