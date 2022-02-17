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
} progress_mode = PROGRESS_MODE_BLOCKING;

static uint16_t listener_port = 12345;

class ListenerContext
{
    private:
        std::shared_ptr<ucxx::UCXXWorker> _worker{nullptr};
        std::shared_ptr<ucxx::UCXXEndpoint> _endpoint{nullptr};
        std::shared_ptr<ucxx::UCXXListener> _listener{nullptr};

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
            return _endpoint == nullptr;
        }

        void createEndpointFromConnRequest(ucp_conn_request_h conn_request)
        {
            if (!isAvailable())
                throw std::runtime_error("Listener context already has an endpoint");

            static bool endpoint_error_handling = true;
            _endpoint = _listener->createEndpointFromConnRequest(conn_request, endpoint_error_handling);
        }

        void releaseEndpoint()
        {
            _endpoint.reset();
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
    std::cerr << "Usage: basic [parameters]" << std::endl;
    std::cerr << "UCXX basic client/server example" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Parameters are:" << std::endl;
    std::cerr << "  -b          Use blocking progress mode" << std::endl;
    std::cerr << "  -t          Use threaded progress mode" << std::endl;
    std::cerr << "  -p <port>   Port number to listen at" << std::endl;
    std::cerr << "  -h          Print this help" << std::endl;
    std::cerr << std::endl;
}

ucs_status_t parseCommand(int argc, char* const argv[])
{
    int c;
    while ((c = getopt(argc, argv, "btp:")) != -1)
    {
        switch (c)
        {
            case 'b':
                progress_mode = PROGRESS_MODE_BLOCKING;
                break;
            case 't':
                progress_mode = PROGRESS_MODE_THREADED;
                break;
            case 'p':
                listener_port = atoi(optarg);
                if (listener_port <= 0)
                {
                    std::cerr << "Wrong listener port: " << listener_port << std::endl;
                    return UCS_ERR_UNSUPPORTED;
                }
                break;
            case 'h':
            default:
                printUsage();
                return UCS_ERR_UNSUPPORTED;
        }
    }

    return UCS_OK;
}

int main(int argc, char **argv)
{
    if (parseCommand(argc, argv) != UCS_OK)
        return -1;

    auto context = ucxx::UCXXContext::create({}, ucxx::UCXXContext::default_feature_flags);
    auto worker = context->createWorker();
    auto listener_ctx = std::make_unique<ListenerContext>(worker);
    auto listener = worker->createListener(listener_port, listener_cb, listener_ctx.get());
    listener_ctx->setListener(listener);
    auto endpoint = worker->createEndpointFromHostname("127.0.0.1", listener_port, true);

    if (progress_mode == PROGRESS_MODE_BLOCKING)
        worker->init_blocking_progress_mode();
    else
        worker->startProgressThread();

    // Block until client connects
    while (listener_ctx->isAvailable())
    {
        if (progress_mode == PROGRESS_MODE_BLOCKING)
            worker->progress_worker_event();
        // Else progress thread is enabled
    }

    std::vector<std::shared_ptr<ucxx::ucxx_request_t>> promises;

    std::vector<int> sendBuffer(5);
    std::iota(std::begin(sendBuffer), std::end(sendBuffer), 0);
    std::vector<int> sendBuffer2(10000);
    std::iota(std::begin(sendBuffer2), std::end(sendBuffer2), 0);

    size_t msgLength = sendBuffer.size() * sizeof(int);
    size_t msgLength2 = sendBuffer2.size() * sizeof(int);

    std::vector<int> recvBuffer(sendBuffer.size());
    std::vector<int> recvBuffer2(sendBuffer2.size());

    // Schedule send and recv messages on different tags and different orders.
    promises.push_back(listener_ctx->getEndpoint()->tag_send(sendBuffer.data(), msgLength, 0));
    promises.push_back(listener_ctx->getEndpoint()->tag_recv(recvBuffer2.data(), msgLength2, 1));
    promises.push_back(endpoint->tag_send(sendBuffer2.data(), msgLength2, 1));
    promises.push_back(endpoint->tag_recv(recvBuffer.data(), msgLength, 0));

    // Wait until all messages are completed
    if (progress_mode == PROGRESS_MODE_BLOCKING)
    {
        for (auto& p : promises)
        {
            auto fut = p->completed_promise.get_future();
            std::future_status status;
            do
            {
                worker->progress_worker_event();
                status = fut.wait_for(std::chrono::nanoseconds::zero());
            } while (status != std::future_status::ready);
        }
    }
    else
    {
        for (auto& p : promises)
            assert_ucs_status(p->completed_promise.get_future().get());
    }

    for (size_t i = 0; i < sendBuffer.size(); i++)
        assert(recvBuffer[i] == sendBuffer[i]);
    for (size_t i = 0; i < sendBuffer2.size(); i++)
        assert(recvBuffer2[i] == sendBuffer2[i]);

    if (progress_mode == PROGRESS_MODE_THREADED)
        worker->stopProgressThread();

    return 0;
}
