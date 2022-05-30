/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
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
