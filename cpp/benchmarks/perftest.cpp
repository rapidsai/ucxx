#include <atomic>
#include <cassert>
#include <chrono>
#include <numeric>
#include <thread>
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
typedef std::unordered_map<transfer_type_t, ucp_tag_t> TagMap;

struct app_context_t {
  ProgressMode progress_mode = ProgressMode::Blocking;
  const char* server_addr    = NULL;
  uint16_t listener_port     = 12345;
  size_t message_size        = 8;
  size_t n_iter              = 100;
  size_t warmup_iter         = 3;
  bool reuse_alloc           = false;
  bool verify_results        = false;
};

class ListenerContext {
 private:
  std::shared_ptr<ucxx::Worker> _worker{nullptr};
  std::shared_ptr<ucxx::Endpoint> _endpoint{nullptr};
  std::shared_ptr<ucxx::Listener> _listener{nullptr};
  std::atomic<bool> _isAvailable{true};

 public:
  ListenerContext(std::shared_ptr<ucxx::Worker> worker) : _worker{worker} {}

  ~ListenerContext() { releaseEndpoint(); }

  void setListener(std::shared_ptr<ucxx::Listener> listener) { _listener = listener; }

  std::shared_ptr<ucxx::Listener> getListener() { return _listener; }

  std::shared_ptr<ucxx::Endpoint> getEndpoint() { return _endpoint; }

  bool isAvailable() const { return _isAvailable; }

  void createEndpointFromConnRequest(ucp_conn_request_h conn_request)
  {
    if (!isAvailable()) throw std::runtime_error("Listener context already has an endpoint");

    static bool endpoint_error_handling = true;
    _endpoint    = _listener->createEndpointFromConnRequest(conn_request, endpoint_error_handling);
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
  static size_t MAX_STRING_LEN = 50;
  char ip_str[MAX_STRING_LEN];
  char port_str[MAX_STRING_LEN];
  ucp_conn_request_attr_t attr{};
  ListenerContext* listener_ctx = (ListenerContext*)arg;

  attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR;
  ucxx::utils::ucsErrorThrow(ucp_conn_request_query(conn_request, &attr));
  ucxx::utils::sockaddr_get_ip_port_str(&attr.client_address, ip_str, port_str, MAX_STRING_LEN);
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
  std::cerr << " basic client/server example" << std::endl;
  std::cerr << std::endl;
  std::cerr << "Usage: basic [server-hostname] [options]" << std::endl;
  std::cerr << std::endl;
  std::cerr << "Parameters are:" << std::endl;
  std::cerr << "  -m          progress mode to use, valid values are: 'polling', 'blocking',"
            << std::endl;
  std::cerr << "              'thread-polling' and 'thread-blocking' (default: 'blocking')"
            << std::endl;
  std::cerr << "  -t          use thread progress mode (disabled)" << std::endl;
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
  while ((c = getopt(argc, argv, "m:p:s:w:n:rv")) != -1) {
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
      case 'r': app_context->reuse_alloc = true; break;
      case 'v': app_context->verify_results = true; break;
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

auto doTransfer(app_context_t& app_context,
                std::shared_ptr<ucxx::Worker> worker,
                std::shared_ptr<ucxx::Endpoint> endpoint,
                TagMap& tagMap,
                BufferMap& bufferMapReuse)
{
  BufferMap localBufferMap;
  if (!app_context.reuse_alloc) localBufferMap = allocateTransferBuffers(app_context.message_size);
  BufferMap& bufferMap = app_context.reuse_alloc ? bufferMapReuse : localBufferMap;

  auto start                                           = std::chrono::high_resolution_clock::now();
  std::vector<std::shared_ptr<ucxx::Request>> requests = {
    endpoint->tagSend(bufferMap[SEND].data(), app_context.message_size, tagMap[SEND]),
    endpoint->tagRecv(bufferMap[RECV].data(), app_context.message_size, tagMap[RECV])};

  // Wait for requests and clear requests
  waitRequests(app_context.progress_mode, worker, requests);
  auto stop = std::chrono::high_resolution_clock::now();

  if (app_context.verify_results)
    for (size_t j = 0; j < bufferMap[SEND].size(); ++j)
      assert(bufferMap[RECV][j] == bufferMap[RECV][j]);

  return std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
}

int main(int argc, char** argv)
{
  app_context_t app_context;
  if (parseCommand(&app_context, argc, argv) != UCS_OK) return -1;

  // Setup: create UCP context, worker, listener and client endpoint.
  auto context = ucxx::createContext({}, ucxx::Context::defaultFeatureFlags);
  auto worker  = context->createWorker();

  bool is_server = app_context.server_addr == NULL;
  TagMap tagMap  = {
    {SEND, is_server ? 0 : 1},
    {RECV, is_server ? 1 : 0},
  };

  std::shared_ptr<ListenerContext> listener_ctx;
  std::shared_ptr<ucxx::Endpoint> endpoint;
  std::shared_ptr<ucxx::Listener> listener;
  if (is_server) {
    listener_ctx = std::make_unique<ListenerContext>(worker);
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
    endpoint =
      worker->createEndpointFromHostname(app_context.server_addr, app_context.listener_port, true);

  std::vector<std::shared_ptr<ucxx::Request>> requests;

  // Allocate wireup buffers
  BufferMap wireupBufferMap = {{SEND, std::vector<char>{1, 2, 3}}, {RECV, std::vector<char>(3, 0)}};

  // Schedule small wireup messages to let UCX identify capabilities between endpoints
  requests.push_back(endpoint->tagSend(
    wireupBufferMap[SEND].data(), wireupBufferMap[SEND].size() * sizeof(int), tagMap[SEND]));
  requests.push_back(endpoint->tagRecv(
    wireupBufferMap[RECV].data(), wireupBufferMap[RECV].size() * sizeof(int), tagMap[RECV]));

  // Wait for wireup requests and clear requests
  waitRequests(app_context.progress_mode, worker, requests);
  requests.clear();

  // Verify wireup result
  for (size_t i = 0; i < wireupBufferMap[SEND].size(); ++i)
    assert(wireupBufferMap[RECV][i] == wireupBufferMap[SEND][i]);

  BufferMap bufferMapReuse;
  if (app_context.reuse_alloc) bufferMapReuse = allocateTransferBuffers(app_context.message_size);

  // Warmup
  for (size_t n = 0; n < app_context.warmup_iter; ++n)
    doTransfer(app_context, worker, endpoint, tagMap, bufferMapReuse);

  // Schedule send and recv messages on different tags and different ordering
  size_t total_duration_ns = 0;
  for (size_t n = 0; n < app_context.n_iter; ++n) {
    auto duration_ns = doTransfer(app_context, worker, endpoint, tagMap, bufferMapReuse);
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
