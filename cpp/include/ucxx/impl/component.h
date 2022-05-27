/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <memory>

namespace ucxx {

UCXXComponent::~UCXXComponent() {}

// Called from child's constructor
void UCXXComponent::setParent(std::shared_ptr<UCXXComponent> parent) { _parent = parent; }

std::shared_ptr<UCXXComponent> UCXXComponent::getParent() const { return _parent; }

}  // namespace ucxx
