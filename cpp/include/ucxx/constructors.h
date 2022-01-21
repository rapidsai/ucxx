#include <memory>

namespace ucxx
{

class UCXXWorker;

template <class ...Args>
std::shared_ptr<UCXXWorker> createWorker(Args&& ...args);

}  // namespace ucxx
