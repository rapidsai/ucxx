#include <memory>

#include <ucxx/typedefs.h>

namespace ucxx {

class UCXXAddress;
class UCXXEndpoint;
class UCXXListener;
class UCXXRequest;
class UCXXWorker;

template <class... Args>
std::shared_ptr<UCXXWorker> createWorker(Args&&... args);

std::shared_ptr<UCXXAddress> createAddressFromWorker(std::shared_ptr<ucxx::UCXXWorker> worker);

std::shared_ptr<UCXXAddress> createAddressFromString(std::string addressString);

std::shared_ptr<UCXXEndpoint> createEndpointFromHostname(std::shared_ptr<UCXXWorker> worker,
                                                         std::string ip_address,
                                                         uint16_t port,
                                                         bool endpoint_error_handling);

std::shared_ptr<UCXXEndpoint> createEndpointFromConnRequest(std::shared_ptr<UCXXListener> listener,
                                                            ucp_conn_request_h conn_request,
                                                            bool endpoint_error_handling);

std::shared_ptr<UCXXEndpoint> createEndpointFromWorkerAddress(std::shared_ptr<UCXXWorker> worker,
                                                              std::shared_ptr<UCXXAddress> address,
                                                              bool endpoint_error_handling);

std::shared_ptr<UCXXListener> createListener(std::shared_ptr<UCXXWorker> worker,
                                             uint16_t port,
                                             ucp_listener_conn_callback_t callback,
                                             void* callback_args);

std::shared_ptr<UCXXRequest> createRequest(std::shared_ptr<UCXXEndpoint>& endpoint,
                                           inflight_requests_t inflight_requests,
                                           std::shared_ptr<ucxx_request_t> request);

}  // namespace ucxx
