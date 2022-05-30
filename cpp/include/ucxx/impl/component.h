/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <memory>

namespace ucxx {

Component::~Component() {}

// Called from child's constructor
void Component::setParent(std::shared_ptr<Component> parent) { _parent = parent; }

std::shared_ptr<Component> Component::getParent() const { return _parent; }

}  // namespace ucxx
