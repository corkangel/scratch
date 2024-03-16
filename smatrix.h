#pragma once

#include "utils.h"

template <typename Storage, uint... Args>
struct sMatrix
{
    static constexpr uint dimensions[] = { Args... };
    static constexpr uint rank = sizeof...(Args);

    sMatrix()
    {
        uint n = 1;
        for (uint i = 0; i < rank; ++i)
        {
            n *= dimensions[i];
        }
        data.resize(n);
    }

    static sMatrix Zeros()
    {
        sMatrix m;
        m.zero_();
        return m;
    }

    static sMatrix Ones()
    {
        sMatrix m;
        m.zero_();
        return m;
    }

    void all_(Storage value)
    {
        for (Storage& d : data)
        {
            d = value;
        }
    }

    void zero_()
    {
        all_(0);
    }

    void ones_()
    {
        all_(1);
    }

    uint size() const
    {
        return uint(data.size());
    }

    uint bytes() const
    {
        return uint(data.size() * sizeof(Storage));
    }

    template<typename... Indices>
    auto& operator()(Indices... indices)
    {
        static_assert(sizeof...(Indices) == rank, "Invalid number of indices");

        const int inds[] = { indices... };
        const uint n = sizeof...(Indices);
        assert(n == rank);

        for (uint i = 0; i < n; ++i)
        {
            assert(inds[i] >= 0 && uint(inds[i]) < dimensions[i]);
        }

        uint index = 0;
        for (uint i = 0; i < n; ++i)
        {
            index = index * dimensions[i] + inds[i];
        }
        return data[index];
    }

    std::vector<Storage> data;
};

using sVec2 = sMatrix<float, 2>;
using sVec3 = sMatrix<float, 3>;


using sMat3x3 = sMatrix<float, 3, 3>;
using sMat4x4 = sMatrix<float, 4, 4>;
