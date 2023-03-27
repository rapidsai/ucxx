/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <memory>

namespace ucxx {

class Component : public std::enable_shared_from_this<Component> {
 protected:
  std::shared_ptr<Component> _parent{nullptr};

 public:
  virtual ~Component();

  // Called from child's constructor
  void setParent(std::shared_ptr<Component> parent);

  std::shared_ptr<Component> getParent() const;
};

}  // namespace ucxx
