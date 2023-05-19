/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>

#include <ucp/api/ucp.h>

#include <ucxx/typedefs.h>

namespace ucxx {

class Buffer;
class InflightRequests;
class RequestAm;
class Request;
class Worker;

namespace internal {

class AmData;

class RecvAmMessage {
 public:
  internal::AmData* _amData{nullptr};  ///< Active messages data
  ucp_ep_h _ep{nullptr};               ///< Handle containing address of the reply endpoint
  std::shared_ptr<RequestAm> _request{
    nullptr};  ///< Request which will later be notified/delivered to user
  std::shared_ptr<Buffer> _buffer{nullptr};  ///< Buffer containing the received data

  /**
   * @brief Constructor of `ucxx::RecvAmMessage`.
   *
   * Construct the object, setting attributes that are later needed by the callback.
   *
   * @param[in] amData  active messages worker data.
   * @param[in] ep      handle containing address of the reply endpoint (i.e., endpoint
   *                    where user is requesting to receive).
   * @param[in] request request to be later notified/delivered to user.
   * @param[in] buffer  buffer containing the received data
   */
  RecvAmMessage(internal::AmData* amData,
                ucp_ep_h ep,
                std::shared_ptr<RequestAm> request,
                std::shared_ptr<Buffer> buffer);

  /**
   * @brief Set the UCP request.
   *
   * Set the underlying UCP request (`_request` attribute) of the `RequestAm`.
   *
   * @param[in] request the UCP request associated to the active message receive operation.
   */
  void setUcpRequest(void* request);

  /**
   * @brief Execute the `ucxx::Request::callback()`.
   *
   * Execute the `ucxx::Request::callback()` method to set the status of the request, the
   * buffer containing the data received and release the reference to this object from
   * `AmData`.
   *
   * @param[in] request the UCP request associated to the active message receive operation.
   * @param[in] status  the completion status of the UCP request.
   */
  void callback(void* request, ucs_status_t status);
};

typedef std::unordered_map<ucp_ep_h, std::queue<std::shared_ptr<RequestAm>>> AmPoolType;
typedef std::unordered_map<RequestAm*, std::shared_ptr<RecvAmMessage>> RecvAmMessageMapType;

class AmData {
 public:
  std::weak_ptr<Worker> _worker{};  ///< The worker to which the Active Message callback belongs
  AmPoolType _recvPool{};  ///< The pool of completed receive requests (waiting for user request)
  AmPoolType _recvWait{};  ///< The pool of user receive requests (waiting for message arrival)
  RecvAmMessageMapType
    _recvAmMessageMap{};  ///< The active messages waiting to be handled by callback
  std::mutex _mutex{};    ///< Mutex to provide access to pools/maps
  std::function<void(std::shared_ptr<Request>)>
    _registerInflightRequest{};  ///< Worker function to register inflight requests with
  std::unordered_map<ucs_memory_type_t, AmAllocatorType>
    _allocators{};  ///< Default and user-defined active message allocators
};

}  // namespace internal

}  // namespace ucxx
