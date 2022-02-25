/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include "log.h"

namespace ucxx
{

class UCXXInitializer
{
    public:
    static UCXXInitializer& getInstance() {
        static UCXXInitializer instance;
        return instance;
    }

    private:
    UCXXInitializer()
    {
        parseLogLevel();
    }

    UCXXInitializer(const UCXXInitializer&) = delete;
    UCXXInitializer& operator=(UCXXInitializer const&) = delete;
};

}  // namespace ucxx
