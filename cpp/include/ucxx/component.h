/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <memory>


namespace ucxx
{

class UCXXComponent : public std::enable_shared_from_this<UCXXComponent>
{
    protected:
        std::shared_ptr<UCXXComponent> _parent{nullptr};

    public:
        virtual ~UCXXComponent() {}

        // Called from child's constructor
        void setParent(std::shared_ptr<UCXXComponent> parent)
        {
            _parent = parent;
        }

        std::shared_ptr<UCXXComponent> getParent() const
        {
            return _parent;
        }
};

}  // namespace ucxx
