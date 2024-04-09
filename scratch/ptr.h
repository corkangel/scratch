#pragma once

#include "utils.h"

class sPtrBase
{
public:
    sPtrBase() : _refCount(0) {}
    virtual ~sPtrBase() {}

    virtual void release() = 0;

    void addRef()
    {
        _refCount.fetch_add(1, std::memory_order_relaxed);
    }

    void defRef()
    {
        if (_refCount.fetch_sub(1, std::memory_order_release) == 0)
        {
            std::atomic_thread_fence(std::memory_order_acquire);
            release();
        }
    }

    uint refCount() const
    {
        return _refCount.load();
    }

    std::atomic<uint> _refCount;
};

template <typename T>
class sPtr
{
public:
    sPtr() : _ptr(nullptr) {}

    sPtr(T* ptr) : _ptr(ptr)
    {
        if (_ptr)
        {
            _ptr->addRef();
        }
    }

    sPtr(const sPtr& other) : _ptr(other._ptr)
    {
        if (_ptr)
        {
            _ptr->addRef();
        }
    }

    ~sPtr()
    {
        if (_ptr)
        {
            _ptr->defRef();
            _ptr = nullptr;
        }
    }

    sPtr& operator=(const sPtr& other)
    {
        if (_ptr)
        {
            _ptr->defRef();
        }

        _ptr = other._ptr;

        if (_ptr)
        {
            _ptr->addRef();
        }

        return *this;
    }

    bool operator==(const sPtr& other) const
    {
        return _ptr == other._ptr;
    }

    bool operator!=(const sPtr& other) const
    {
        return _ptr != other._ptr;
    }

    T* operator->() const
    {
        return _ptr;
    }

    T* operator()() const
    {
        return _ptr;
    }

    T& operator*() const
    {
        return *_ptr;
    }

    bool isnull() const
    {
        return _ptr == nullptr;
    }

    void swap(sPtr& other)
    {
        T* temp = _ptr;
        _ptr = other._ptr;
        other._ptr = temp;
    }

    void reset()
    {
        if (_ptr)
        {
            _ptr->defRef();
            _ptr = nullptr;
        }
    }

    sPtr<T> operator+(const sPtr<T>& right) const { return _ptr->operator+(right); }
    sPtr<T> operator-(const sPtr<T>& right) const { return _ptr->operator-(right); }
    sPtr<T> operator*(const sPtr<T>& right) const { return _ptr->operator*(right); }
    sPtr<T> operator/(const sPtr<T>& right) const { return _ptr->operator/(right); }
    sPtr<T> operator+=(const sPtr<T>& right) { return _ptr->operator+=(right); }
    sPtr<T> operator-=(const sPtr<T>& right) { return _ptr->operator-=(right); }
    sPtr<T> operator*=(const sPtr<T>& right) { return _ptr->operator*=(right); }
    sPtr<T> operator/=(const sPtr<T>& right) { return _ptr->operator/=(right); }
    sPtr<T> operator+(const float value) const { return _ptr->operator+(value); }
    sPtr<T> operator-(const float value) const { return _ptr->operator-(value); }
    sPtr<T> operator*(const float value) const { return _ptr->operator*(value); }
    sPtr<T> operator/(const float value) const { return _ptr->operator/(value); }
    sPtr<T> operator+=(const float value) { return _ptr->operator+=(value); }
    sPtr<T> operator-=(const float value) { return _ptr->operator-=(value); }
    sPtr<T> operator*=(const float value) { return _ptr->operator*=(value); }
    sPtr<T> operator/=(const float value) { return _ptr->operator/=(value); }

protected:
    T* _ptr;
};

