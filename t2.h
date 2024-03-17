#pragma once

#include <cassert>
#include <array>

#include "utils.h"

#define sTENSOR_MAX_DIMENSIONS 4


class sTensor
{
    uint _dimensions[sTENSOR_MAX_DIMENSIONS];
    uint _rank;

    float* _storage;
    uint _storageSize;
    bool _storageOwned;

    void Init(const uint* data)
    {
        assert(_rank <= sTENSOR_MAX_DIMENSIONS);
        memcpy(_dimensions, data, _rank * sizeof(uint));

        _storageSize = 1;
        for (uint i = 0; i < _rank; i++)
            _storageSize *= _dimensions[i];

        _storage = new float[_storageSize];
    }

public:

    template <typename... Dimensions, typename std::enable_if<(std::is_same_v<Dimensions, uint> && ...), uint>::type = 0>
    sTensor(Dimensions... dimensions) : _rank(sizeof...(dimensions)), _storageOwned(true)
    {
        static_assert(sizeof...(dimensions) <= sTENSOR_MAX_DIMENSIONS, "Too many dimensions");
        const uint data[] = { dimensions... };
        Init(data);
    }

    template <typename... Dimensions, typename std::enable_if<(std::is_same_v<Dimensions, int> && ...), int>::type = 0>
    sTensor(Dimensions... dimensions) : _rank(sizeof...(dimensions)), _storageOwned(true)
    {
        static_assert(sizeof...(dimensions) <= sTENSOR_MAX_DIMENSIONS, "Too many dimensions");
        const int data[] = { dimensions... };
        Init(reinterpret_cast<const uint*>(data));
    }

    sTensor(uint rank, uint* dimensions) : _rank(rank), _storageOwned(true)
    {
        Init(dimensions);
    }

    ~sTensor()
    {
        if (_storageOwned)
            delete[] _storage;
    }

    // ---------------- static constructors -----------------
    
    template<typename... Dimensions> static sTensor Dims(Dimensions... dims)
    {
        sTensor result(dims...);
        return result;
    }

    template<typename... Dimensions> static sTensor Ones(Dimensions... dims)
    {
        sTensor result(dims...);
        result.ones_();
        return result;
    }

    template<typename... Dimensions> static sTensor Zeros(Dimensions... dims)
    {
        sTensor result(dims...);
        result.zero_();
        return result;
    }

    template<typename... Dimensions> static sTensor Randoms(Dimensions... dims)
    {
        sTensor result(dims...);
        result.random_();
        return result;
    }

    template<typename... Dimensions> static sTensor NormalDistribution(const float mean, const float stddev, Dimensions... dims)
    {
        sTensor result(dims...);
        result.normal_distribution_(mean, stddev);
        return result;
    }

    template<typename... Dimensions> static sTensor Integers(const int start, Dimensions... dims)
    {
        sTensor result(dims...);
        result.integers_(start);
        return result;
    }

    template<typename... Dimensions> static sTensor Linear(const float start, const float step, Dimensions... dims)
    {
        sTensor result(dims...);
        result.linear_(start, step);
        return result;
    }

    // ---------------- accessors -----------------

    uint rank() const
    {
        return _rank;
    }

    uint dim(uint i) const
    {
        assert(i < _rank);
        return _dimensions[i];
    }

    uint size() const
    {
        return _storageSize;
    }

    uint bytes() const
    {
        return _storageSize * sizeof(float);
    }

    float* data()
    {
        return _storage;
    }

    const float* data_const() const
    {
        return _storage;
    }

    // ---------------- in place operations -----------------

    void fill_(float value)
    {
        for (uint i = 0; i < _storageSize; i++)
            _storage[i] = value;
    }

    void zero_()
    {
        fill_(0.0f);
    }

    void ones_()
    {
        fill_(1.0f);
    }

    void gaussian_(float bandwidth)
    {
        for (uint i = 0; i < _storageSize; i++)
            _storage[i] = gaussian(_storage[i], bandwidth);
    }

    void pow_(uint power)
    {
        for (uint i = 0; i < _storageSize; i++)
            _storage[i] = float(pow(_storage[i], power));
    }

    void sqrt_()
    {
        for (uint i = 0; i < _storageSize; i++)
            _storage[i] = sqrt(_storage[i]);
    }

    void exp_()
    {
        for (uint i = 0; i < _storageSize; i++)
            _storage[i] = exp(_storage[i]);
    }

    void log_()
    {
        for (uint i = 0; i < _storageSize; i++)
            _storage[i] = log(_storage[i]);
    }

    void abs_()
    {
        for (uint i = 0; i < _storageSize; i++)
            _storage[i] = abs(_storage[i]);
    }

    void random_()
    {
        for (uint i = 0; i < _storageSize; i++)
            _storage[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    void normal_distribution_(const float mean, const float stddev)
    {
        std::default_random_engine generator;
        std::normal_distribution distribution(mean, stddev);
        for (uint i = 0; i < _storageSize; i++)
            _storage[i] = distribution(generator);
    }

    void integers_(int start)
    {
        for (uint i = 0; i < _storageSize; i++)
            _storage[i] = float(start++);
    }

    void linear_(float start, const float step)
    {
        for (uint i = 0; i < _storageSize; i++)
        {
            _storage[i] = start;
            start += step;
        }
    }

    template<typename... Dimensions>
    void view_(Dimensions... dims)
    {
        const int ds[] = { dims... };
        const int numDims = sizeof...(Dimensions);
        assert(numDims == _rank);

        uint n = 1;
        for (uint i = 0; i < _rank; ++i)
        {
            _dimensions[i] = ds[i];
            n *= ds[i];
        }
        assert(n == _storageSize);
    }

    // ---------------- scalar operations -----------------

    float sum() const
    {
        float result = 0.0f;
        for (uint i = 0; i < _storageSize; i++)
            result += _storage[i];
        return result;
    }

    float mean() const
    {
        return sum() / _storageSize;
    }

    float mse()
    {
        float result = 0.0f;
        for (uint i = 0; i < _storageSize; i++)
            result += float(pow(_storage[i], 2));
        return result / _storageSize;
    }

    float rmse()
    {
        return sqrt(mse());
    }


};


std::ostream& operator<<(std::ostream& os, const sTensor& m)
{
    uint n = 1;
    const float* data = m.data_const();

    for (uint i = 1; i < m.rank(); ++i)
    {
        n *= m.dim(i);
    }

    os << "dMat( dims:[";
    for (uint i = 0; i < m.rank(); ++i)
    {
        os << m.dim(i);
        if (i != m.rank() - 1)
        {
            os << ", ";
        }
    }
    os << "], data:[";
    for (uint i = 0; i < m.size(); ++i)
    {
        if (i >= 20 && i < m.size() - 20)
        {
            if (i == 20)
            {
                os << "\r\n    ...";
            }
            continue;
        }

        if (i != 0 && i % n == 0)
        {
            os << "\r\n    ";
        }

        os << data[i];
        if (i != m.size() - 1)
        {
            os << ", ";
        }

    }
    os << "]";
    os << ")";
    return os;
}