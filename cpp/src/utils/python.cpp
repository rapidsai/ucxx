/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include <dlfcn.h>

#include <ucxx/log.h>
#include <ucxx/utils/python.h>

namespace ucxx {

namespace utils {

typedef std::shared_ptr<::ucxx::Future> (*createPythonFutureHandle)(
  std::shared_ptr<::ucxx::Notifier>);
typedef std::shared_ptr<::ucxx::Notifier> (*createPythonNotifierHandle)();

static bool _ucxxPythonLoadChecked                      = false;
static void* _ucxxPythonLib                             = nullptr;
static createPythonFutureHandle _createPythonFuture     = nullptr;
static createPythonNotifierHandle _createPythonNotifier = nullptr;

bool isPythonAvailable()
{
#if UCXX_ENABLE_PYTHON
  if (!_ucxxPythonLoadChecked) {
    _ucxxPythonLoadChecked = true;
    _ucxxPythonLib         = dlopen("libucxx_python.so", RTLD_LAZY);
    if (_ucxxPythonLib == nullptr)
      ucxx_debug("dlopen('libucxx_python.so') failed");
    else
      ucxx_debug("dlopen('libucxx_python.so') loaded at %p", _ucxxPythonLib);
  }
  return _ucxxPythonLib != nullptr;
#else
  return false;
#endif
}

static void* loadSymbol(std::string& symbolName)
{
  if (!isPythonAvailable()) return nullptr;

  return dlsym(_ucxxPythonLib, symbolName.c_str());
}

std::shared_ptr<::ucxx::Future> createPythonFuture(std::shared_ptr<::ucxx::Notifier> notifier)
{
  std::string symbolName{"_ZN4ucxx6python18createPythonFutureESt10shared_ptrINS_8NotifierEE"};

  if (_createPythonFuture == nullptr) {
    void* symbol = loadSymbol(symbolName);
    if (symbol == nullptr) {
      ucxx_error("failed to load %s", symbolName.c_str());
      return nullptr;
    }

    _createPythonFuture = reinterpret_cast<createPythonFutureHandle>(symbol);
  }

  return _createPythonFuture(notifier);
}

std::shared_ptr<::ucxx::Notifier> createPythonNotifier()
{
  std::string symbolName{"_ZN4ucxx6python20createPythonNotifierEv"};

  if (_createPythonNotifier == nullptr) {
    void* symbol = loadSymbol(symbolName);
    if (symbol == nullptr) {
      ucxx_error("failed to load %s", symbolName.c_str());
      return nullptr;
    }

    _createPythonNotifier = reinterpret_cast<createPythonNotifierHandle>(symbol);
  }

  return _createPythonNotifier();
}

}  // namespace utils

}  // namespace ucxx
