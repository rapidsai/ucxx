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

static bool _ucxxPythonLoadChecked = false;
static void* _ucxxPythonLib        = nullptr;

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

}  // namespace utils

}  // namespace ucxx
