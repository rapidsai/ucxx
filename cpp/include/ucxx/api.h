/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#ifndef UCXX_ENABLE_PYTHON
#define UCXX_ENABLE_PYTHON 0
#endif

#ifndef UCXX_ENABLE_RMM
#define UCXX_ENABLE_RMM 0
#endif

#include <ucxx/address.h>
#include <ucxx/context.h>
#include <ucxx/endpoint.h>
#include <ucxx/listener.h>
#include <ucxx/request.h>
#include <ucxx/typedefs.h>
#include <ucxx/worker.h>

#include <ucxx/impl/address.h>
#include <ucxx/impl/buffer_helper.h>
#include <ucxx/impl/component.h>
#include <ucxx/impl/config.h>
#include <ucxx/impl/context.h>
#include <ucxx/impl/delayed_submission.h>
#include <ucxx/impl/endpoint.h>
#include <ucxx/impl/initializer.h>
#include <ucxx/impl/listener.h>
#include <ucxx/impl/log.h>
#include <ucxx/impl/request.h>
#include <ucxx/impl/request_stream.h>
#include <ucxx/impl/request_tag.h>
#include <ucxx/impl/request_tag_multi.h>
#include <ucxx/impl/worker.h>
#include <ucxx/impl/worker_progress_thread.h>
#include <ucxx/utils/impl/file_descriptor.h>
#include <ucxx/utils/impl/sockaddr.h>
#include <ucxx/utils/impl/ucx.h>

#if UCXX_ENABLE_PYTHON
#include <ucxx/python/impl/exception.h>
#include <ucxx/python/impl/future.h>
#include <ucxx/python/impl/notifier.h>
#include <ucxx/python/impl/python_future.h>
#endif
