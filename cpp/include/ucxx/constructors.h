#include <memory>

namespace ucxx
{

class UCXXEndpoint;
class UCXXWorker;

template <class ...Args>
std::shared_ptr<UCXXWorker> createWorker(Args&& ...args);

std::shared_ptr<UCXXEndpoint> createEndpointFromHostname(
        std::shared_ptr<UCXXWorker> worker,
        std::string ip_address,
        uint16_t port,
        bool endpoint_error_handling
);

}  // namespace ucxx
