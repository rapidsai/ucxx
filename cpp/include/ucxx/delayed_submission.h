/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include <ucxx/log.h>

namespace ucxx {

class DelayedSubmission;

typedef std::function<void()> DelayedSubmissionCallbackType;

class DelayedSubmission {
 public:
  bool _send{false};
  void* _buffer{nullptr};
  size_t _length{0};
  ucp_tag_t _tag{0};

  DelayedSubmission() = delete;

  DelayedSubmission(const bool send, void* buffer, const size_t length, const ucp_tag_t tag = 0);
};

class DelayedSubmissionCallback {
 private:
  DelayedSubmissionCallbackType _callback{nullptr};

 public:
  DelayedSubmissionCallback(DelayedSubmissionCallbackType callback);

  DelayedSubmissionCallbackType get();
};

typedef std::shared_ptr<DelayedSubmissionCallback> DelayedSubmissionCallbackPtrType;

class DelayedSubmissionCollection {
 private:
  std::vector<DelayedSubmissionCallbackPtrType> _collection{};
  std::mutex _mutex{};

 public:
  DelayedSubmissionCollection()                                   = default;
  DelayedSubmissionCollection(const DelayedSubmissionCollection&) = delete;
  DelayedSubmissionCollection& operator=(DelayedSubmissionCollection const&) = delete;
  DelayedSubmissionCollection(DelayedSubmissionCollection&& o)               = delete;
  DelayedSubmissionCollection& operator=(DelayedSubmissionCollection&& o) = delete;

  void process();

  void registerRequest(DelayedSubmissionCallbackType callback);
};

}  // namespace ucxx
