#include <memory>

#include <ucxx/typedefs.h>

namespace ucxx {

class UCXXAddress;
class UCXXRequestTagMulti;
class UCXXContext;
class UCXXEndpoint;
class UCXXListener;
class UCXXRequest;
class UCXXRequestStream;
class UCXXRequestTag;
class UCXXWorker;

std::shared_ptr<UCXXAddress> createAddressFromWorker(std::shared_ptr<ucxx::UCXXWorker> worker);

std::shared_ptr<UCXXAddress> createAddressFromString(std::string addressString);

std::shared_ptr<UCXXContext> createContext(const UCXXConfigMap ucx_config,
                                           const uint64_t feature_flags);

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

std::shared_ptr<UCXXWorker> createWorker(std::shared_ptr<UCXXContext> context,
                                         const bool enableDelayedNotification);

std::shared_ptr<UCXXRequestTagMulti> tagMultiSend(std::shared_ptr<UCXXEndpoint> endpoint,
                                                  std::vector<void*>& buffer,
                                                  std::vector<size_t>& size,
                                                  std::vector<int>& isCUDA,
                                                  const ucp_tag_t tag);
void tagMultiSendBlocking(std::shared_ptr<UCXXEndpoint> endpoint,
                          std::vector<void*>& buffer,
                          std::vector<size_t>& size,
                          std::vector<int>& isCUDA,
                          ucp_tag_t tag);
std::shared_ptr<UCXXRequestTagMulti> tagMultiRecv(std::shared_ptr<UCXXEndpoint> endpoint,
                                                  const ucp_tag_t tag);
std::vector<std::unique_ptr<UCXXPyBuffer>> tagMultiRecvBlocking(
  std::shared_ptr<UCXXEndpoint> endpoint, ucp_tag_t tag);

std::shared_ptr<UCXXRequestStream> createRequestStream(std::shared_ptr<UCXXEndpoint> endpoint,
                                                       bool send,
                                                       void* buffer,
                                                       size_t length);
std::shared_ptr<UCXXRequestTag> createRequestTag(
  std::shared_ptr<UCXXEndpoint> endpoint,
  bool send,
  void* buffer,
  size_t length,
  ucp_tag_t tag,
  const bool enablePythonFuture,
  std::function<void(std::shared_ptr<void>)> callbackFunction,
  std::shared_ptr<void> callbackData);

#if UCXX_ENABLE_PYTHON
class UCXXNotifier;

template <class... Args>
std::shared_ptr<UCXXNotifier> createNotifier(Args&&... args);
#endif

}  // namespace ucxx
