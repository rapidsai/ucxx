/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <memory>


class UCXXComponent : public std::enable_shared_from_this<UCXXComponent>
{
    protected:
        std::shared_ptr<UCXXComponent> _parent;

    public:
        virtual ~UCXXComponent() {}

        void setParent(std::shared_ptr<UCXXComponent> parent)
        {
            _parent = parent;
        }

        std::shared_ptr<UCXXComponent> getParent() const
        {
            return _parent;
        }

        virtual void addChild(std::shared_ptr<UCXXComponent> child) {}
        virtual void removeChild(UCXXComponent* child) {}
};
