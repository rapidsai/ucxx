/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#pragma once

#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>


namespace ucxx
{

const size_t HeaderFramesSize = 100;

class Header
{
public:
    bool next;
    size_t nframes;
    bool isCUDA[HeaderFramesSize];
    size_t size[HeaderFramesSize];

    Header()
        : next{false}, nframes{0}
    {
        std::fill(isCUDA, isCUDA + HeaderFramesSize, false);
        std::fill(size, size + HeaderFramesSize, 0);
    }

    Header(bool next, size_t nframes, bool isCUDA, size_t size)
        : next{next}, nframes{nframes}
    {
        std::fill(this->isCUDA, this->isCUDA + nframes, isCUDA);
        std::fill(this->size, this->size + nframes, size);
        if (nframes < HeaderFramesSize)
        {
            std::fill(this->isCUDA + nframes, this->isCUDA + HeaderFramesSize, false);
            std::fill(this->size + nframes, this->size + HeaderFramesSize, 0);
        }
    }

    Header(bool next, size_t nframes, bool* isCUDA, size_t* size)
        : next{next}, nframes{nframes}
    {
        std::copy(isCUDA, isCUDA + nframes, this->isCUDA);
        std::copy(size, size + nframes, this->size);
        if (nframes < HeaderFramesSize)
        {
            std::fill(this->isCUDA + nframes, this->isCUDA + HeaderFramesSize, false);
            std::fill(this->size + nframes, this->size + HeaderFramesSize, 0);
        }
    }

    Header(std::string serializedHeader)
    {
        deserialize(serializedHeader);
    }

    static size_t dataSize()
    {
        return sizeof(next) + sizeof(nframes) + sizeof(isCUDA) + sizeof(size);
    }

    std::string serialize()
    {
        std::stringstream ss;

        ss.write((char const*)&next, sizeof(next));
        ss.write((char const*)&nframes, sizeof(nframes));
        for (size_t i = 0; i < HeaderFramesSize; ++i)
            ss.write((char const*)&isCUDA[i], sizeof(isCUDA[i]));
        for (size_t i = 0; i < HeaderFramesSize; ++i)
            ss.write((char const*)&size[i], sizeof(size[i]));

        return ss.str();
    }

    void deserialize(const std::string& serializedHeader)
    {
        std::stringstream ss{serializedHeader};

        ss.read((char*)&next, sizeof(next));
        ss.read((char*)&nframes, sizeof(nframes));
        for (size_t i = 0; i < HeaderFramesSize; ++i)
            ss.read((char*)&isCUDA[i], sizeof(isCUDA[i]));
        for (size_t i = 0; i < HeaderFramesSize; ++i)
            ss.read((char*)&size[i], sizeof(size[i]));
    }

    void print()
    {
        std::cout << next << " " << nframes;
        std::cout << " { ";
        std::copy(isCUDA, isCUDA + HeaderFramesSize,
                  std::ostream_iterator<bool>(std::cout, " "));
        std::cout << "} { ";
        std::copy(size, size+ HeaderFramesSize,
                  std::ostream_iterator<size_t>(std::cout, " "));
        std::cout << "}";
        std::cout << std::endl;
    }
};

typedef void (*UCXXPyBufferDeleter)(void*);

class UCXXPyBuffer
{
protected:
    std::unique_ptr<void, UCXXPyBufferDeleter> _ptr{nullptr, [](void*){}};
    bool _isCUDA{false};
    size_t _size{0};
    bool _isValid{false};

public:
    UCXXPyBuffer(void* ptr, UCXXPyBufferDeleter deleter, const bool isCUDA, const size_t size)
        : _ptr{std::unique_ptr<void, UCXXPyBufferDeleter>(ptr, deleter)}, _isCUDA{isCUDA}, _size{size}, _isValid{true}
    {
    }

    UCXXPyBuffer() = default;

    UCXXPyBuffer(const UCXXPyBuffer&) = delete;
    UCXXPyBuffer& operator=(UCXXPyBuffer const&) = delete;

    UCXXPyBuffer(UCXXPyBuffer&& o) noexcept
        : _ptr{std::exchange(o._ptr, nullptr)},
          _isCUDA{std::exchange(o._isCUDA, false)},
          _size{std::exchange(o._size, 0)},
          _isValid{std::exchange(o._isValid, false)}
    {
    }

    UCXXPyBuffer& operator=(UCXXPyBuffer&& o) noexcept
    {
        this->_ptr = std::exchange(o._ptr, nullptr);
        this->_isCUDA = std::exchange(o._isCUDA, false);
        this->_size = std::exchange(o._size, 0);
        this->_isValid = std::exchange(o._isValid, false);

        return *this;
    }

    bool isValid() const
    {
        return _isValid;
    }

    size_t getSize() const
    {
        return _size;
    }

    bool isCUDA()
    {
        return _isCUDA;
    }

    virtual void* data() = 0;
};

class UCXXPyHostBuffer : public UCXXPyBuffer
{
public:
    UCXXPyHostBuffer(const size_t size)
        : UCXXPyBuffer(malloc(size), UCXXPyHostBuffer::free, false, size)
    {
    }

    std::unique_ptr<void, UCXXPyBufferDeleter> get()
    {
        _isValid = false;
        return std::move(_ptr);
    }

    void* release()
    {
        _isValid = false;
        return _ptr.release();
    }

    void* data()
    {
        return _ptr.get();
    }

    static void free(void* ptr)
    {
        std::cout << "UCXXPyHostBuffer.free(): " << ptr << std::endl;
        free(ptr);
    }
};


class UCXXPyRMMBuffer : public UCXXPyBuffer
{
public:
    UCXXPyRMMBuffer(const size_t size)
        : UCXXPyBuffer(new rmm::device_buffer(size, rmm::cuda_stream_default), UCXXPyRMMBuffer::free, true, size)
    {
    }

    std::unique_ptr<rmm::device_buffer> get()
    {
        _isValid = false;
        return std::unique_ptr<rmm::device_buffer>((rmm::device_buffer*)_ptr.release());
    }

    void* data()
    {
        return ((rmm::device_buffer*)_ptr.get())->data();
    }

    static void free(void* ptr)
    {
        rmm::device_buffer* p = (rmm::device_buffer*)ptr;
        delete p;
    }
};


std::unique_ptr<UCXXPyBuffer> allocateBuffer(const bool isCUDA, const size_t size)
{
    if (isCUDA)
        return std::make_unique<UCXXPyRMMBuffer>(size);
    else
        return std::make_unique<UCXXPyHostBuffer>(size);
}

typedef UCXXPyHostBuffer* UCXXPyHostBufferPtr;
typedef UCXXPyRMMBuffer* UCXXPyRMMBufferPtr;

}  // namespace ucxx
