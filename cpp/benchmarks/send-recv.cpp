#include <atomic>
#include <cassert>
#include <chrono>
#include <numeric>
#include <thread>
#include <vector>

#include <ucxx/api.h>
#include <ucxx/sockaddr_utils.h>


enum progress_mode_t {
    PROGRESS_MODE_BLOCKING,
    PROGRESS_MODE_THREADED
};

enum transfer_type_t {
    SEND,
    RECV
};

typedef std::unordered_map<transfer_type_t, std::vector<char>> BufferMap;
typedef std::unordered_map<transfer_type_t, ucp_tag_t> TagMap;

struct app_context_t {
    progress_mode_t progress_mode = PROGRESS_MODE_BLOCKING;
    const char* server_addr = NULL;
    uint16_t listener_port = 12345;
    size_t message_size = 8;
    size_t n_iter = 100;
    size_t warmup_iter = 3;
    bool reuse_alloc = false;
    bool verify_results = false;
};

class ListenerContext
{
    private:
        std::shared_ptr<ucxx::UCXXWorker> _worker{nullptr};
        std::shared_ptr<ucxx::UCXXEndpoint> _endpoint{nullptr};
        std::shared_ptr<ucxx::UCXXListener> _listener{nullptr};
        std::atomic<bool> _isAvailable{true};

    public:
        ListenerContext(std::shared_ptr<ucxx::UCXXWorker> worker) : _worker{worker}
        {
        }

        ~ListenerContext()
        {
            releaseEndpoint();
        }

        void setListener(std::shared_ptr<ucxx::UCXXListener> listener)
        {
            _listener = listener;
        }

        std::shared_ptr<ucxx::UCXXListener> getListener()
        {
            return _listener;
        }

        std::shared_ptr<ucxx::UCXXEndpoint> getEndpoint()
        {
            return _endpoint;
        }

        bool isAvailable() const
        {
            return _isAvailable;
        }

        void createEndpointFromConnRequest(ucp_conn_request_h conn_request)
        {
            if (!isAvailable())
                throw std::runtime_error("Listener context already has an endpoint");

            static bool endpoint_error_handling = true;
            _endpoint = _listener->createEndpointFromConnRequest(conn_request, endpoint_error_handling);
            _isAvailable = false;
        }

        void releaseEndpoint()
        {
            _endpoint.reset();
            _isAvailable = true;
        }
};

static void listener_cb(ucp_conn_request_h conn_request, void *arg)
{
    static size_t MAX_STRING_LEN = 50;
    char ip_str[MAX_STRING_LEN];
    char port_str[MAX_STRING_LEN];
    ucp_conn_request_attr_t attr;
    ListenerContext* listener_ctx = (ListenerContext*)arg;

    attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR;
    assert_ucs_status(ucp_conn_request_query(conn_request, &attr));
    sockaddr_utils_get_ip_port_str(&attr.client_address, ip_str, port_str, MAX_STRING_LEN);
    std::cout << "Server received a connection request from client at address " <<
        ip_str << ":" << port_str << std::endl;

    if (listener_ctx->isAvailable()) {
        listener_ctx->createEndpointFromConnRequest(conn_request);
    } else {
        // The server is already handling a connection request from a client,
        // reject this new one
        std::cout << "Rejecting a connection request from " << ip_str << ":" <<
            port_str << "." << std::endl << "Only one client at a time is supported." <<
            std::endl;
        assert_ucs_status(ucp_listener_reject(listener_ctx->getListener()->get_handle(), conn_request));
    }
}

static void printUsage()
{
    std::cerr << "UCXX basic client/server example" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Usage: basic [server-hostname] [options]" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Parameters are:" << std::endl;
    std::cerr << "  -b          use blocking progress mode (enabled)" << std::endl;
    std::cerr << "  -t          use threaded progress mode (disabled)" << std::endl;
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
    while ((c = getopt(argc, argv, "btp:s:w:n:rv")) != -1)
    {
        switch (c)
        {
            case 'b':
                app_context->progress_mode = PROGRESS_MODE_BLOCKING;
                break;
            case 't':
                app_context->progress_mode = PROGRESS_MODE_THREADED;
                break;
            case 'p':
                app_context->listener_port = atoi(optarg);
                if (app_context->listener_port <= 0)
                {
                    std::cerr << "Wrong listener port: " << app_context->listener_port << std::endl;
                    return UCS_ERR_INVALID_PARAM;
                }
                break;
            case 's':
                app_context->message_size = atoi(optarg);
                if (app_context->message_size <= 0)
                {
                    std::cerr << "Wrong message size: " << app_context->message_size << std::endl;
                    return UCS_ERR_INVALID_PARAM;
                }
                break;
            case 'w':
                app_context->warmup_iter = atoi(optarg);
                if (app_context->warmup_iter <= 0)
                {
                    std::cerr << "Wrong number of warmup iterations: " << app_context->warmup_iter << std::endl;
                    return UCS_ERR_INVALID_PARAM;
                }
                break;
            case 'n':
                app_context->n_iter = atoi(optarg);
                if (app_context->n_iter <= 0)
                {
                    std::cerr << "Wrong number of iterations: " << app_context->n_iter << std::endl;
                    return UCS_ERR_INVALID_PARAM;
                }
                break;
            case 'r':
                app_context->reuse_alloc = true;
                break;
            case 'v':
                app_context->verify_results = true;
                break;
            case 'h':
            default:
                printUsage();
                return UCS_ERR_INVALID_PARAM;
        }
    }

    if (optind < argc) {
        app_context->server_addr = argv[optind];
    }

    return UCS_OK;
}

void waitRequests(progress_mode_t progress_mode, std::shared_ptr<ucxx::UCXXWorker> worker, std::vector<std::shared_ptr<ucxx::UCXXRequest>>& requests)
{
    // Wait until all messages are completed
    if (progress_mode == PROGRESS_MODE_BLOCKING)
    {
        for (auto& r : requests)
        {
            do
            {
                worker->progress_worker_event();
            } while (!r->isCompleted());
            r->checkError();
        }
    }
    else
    {
        for (auto& r : requests)
        {
            while (!r->isCompleted());
            r->checkError();
        }
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
    return BufferMap{
        {SEND, std::vector<char>(message_size, 0xaa)},
        {RECV, std::vector<char>(message_size)}
    };
}

void doTransfer(app_context_t& app_context,
                std::shared_ptr<ucxx::UCXXWorker> worker,
                std::shared_ptr<ucxx::UCXXEndpoint> endpoint,
                TagMap& tagMap,
                BufferMap& bufferMapReuse)
{
    BufferMap bufferMap;
    if (!app_context.reuse_alloc)
        bufferMap = allocateTransferBuffers(app_context.message_size);
    else
        bufferMap = bufferMapReuse;

    std::vector<std::shared_ptr<ucxx::UCXXRequest>> requests = {
        endpoint->tag_send(bufferMap[SEND].data(), app_context.message_size, tagMap[SEND]),
        endpoint->tag_recv(bufferMap[RECV].data(), app_context.message_size, tagMap[RECV])
    };

    // Wait for requests and clear requests
    waitRequests(app_context.progress_mode, worker, requests);

    if (app_context.verify_results)
        for (size_t j = 0; j < bufferMap[SEND].size(); ++j)
            assert(bufferMap[RECV][j] == bufferMap[RECV][j]);
}

int main(int argc, char **argv)
{
    app_context_t app_context;
    if (parseCommand(&app_context, argc, argv) != UCS_OK)
        return -1;

    // Setup: create UCP context, worker, listener and client endpoint.
    auto context = ucxx::UCXXContext::create({}, ucxx::UCXXContext::default_feature_flags);
    auto worker = context->createWorker();

    bool is_server = app_context.server_addr == NULL;
    TagMap tagMap = {
        {SEND, is_server ? 0 : 1},
        {RECV, is_server ? 1 : 0},
    };

    std::shared_ptr<ListenerContext> listener_ctx;
    std::shared_ptr<ucxx::UCXXEndpoint> endpoint;
    std::shared_ptr<ucxx::UCXXListener> listener;
    if (is_server)
    {
        listener_ctx = std::make_unique<ListenerContext>(worker);
        listener = worker->createListener(app_context.listener_port, listener_cb, listener_ctx.get());
        listener_ctx->setListener(listener);
    }

    // Initialize worker progress
    if (app_context.progress_mode == PROGRESS_MODE_BLOCKING)
        worker->init_blocking_progress_mode();
    else
        worker->startProgressThread();

    // Block until client connects
    while (is_server && listener_ctx->isAvailable())
    {
        if (app_context.progress_mode == PROGRESS_MODE_BLOCKING)
            worker->progress_worker_event();
        // Else progress thread is enabled
    }

    if (is_server)
        endpoint = listener_ctx->getEndpoint();
    else
        endpoint = worker->createEndpointFromHostname(app_context.server_addr, app_context.listener_port, true);

    std::vector<std::shared_ptr<ucxx::UCXXRequest>> requests;

    // Allocate wireup buffers
    BufferMap wireupBufferMap = {
        {SEND, std::vector<char>{1, 2, 3}},
        {RECV, std::vector<char>(3, 0)}
    };

    // Schedule small wireup messages to let UCX identify capabilities between endpoints
    requests.push_back(endpoint->tag_send(wireupBufferMap[SEND].data(), wireupBufferMap[SEND].size() * sizeof(int), tagMap[SEND]));
    requests.push_back(endpoint->tag_recv(wireupBufferMap[RECV].data(), wireupBufferMap[RECV].size() * sizeof(int), tagMap[RECV]));

    // Wait for wireup requests and clear requests
    waitRequests(app_context.progress_mode, worker, requests);
    requests.clear();

    // Verify wireup result
    for (size_t i = 0; i < wireupBufferMap[SEND].size(); ++i)
        assert(wireupBufferMap[RECV][i] == wireupBufferMap[SEND][i]);

    BufferMap bufferMapReuse;
    if (app_context.reuse_alloc)
        bufferMapReuse = allocateTransferBuffers(app_context.message_size);

    // Warmup
    for (size_t n = 0; n < app_context.warmup_iter; ++n)
        doTransfer(app_context, worker, endpoint, tagMap, bufferMapReuse);

    // Schedule send and recv messages on different tags and different ordering
    size_t total_duration_ns = 0;
    for (size_t n = 0; n < app_context.n_iter; ++n)
    {
        auto start = std::chrono::high_resolution_clock::now();

        doTransfer(app_context, worker, endpoint, tagMap, bufferMapReuse);

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
        total_duration_ns += duration_ns;
        auto elapsed = parseTime(duration_ns);
        auto bandwidth = parseBandwidth(app_context.message_size * 2, duration_ns);

        std::cout << "Elapsed, bandwidth: " << elapsed << ", " << bandwidth << std::endl;
    }

    auto total_elapsed = parseTime(total_duration_ns);
    auto total_bandwidth = parseBandwidth(app_context.n_iter * app_context.message_size * 2, total_duration_ns);

    std::cout << "Total elapsed, bandwidth: " << total_elapsed << ", " << total_bandwidth << std::endl;

    // Stop progress thread
    if (app_context.progress_mode == PROGRESS_MODE_THREADED)
        worker->stopProgressThread();

    return 0;
}
