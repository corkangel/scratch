#pragma once

#include "utils.h"

#include <random>

template <typename Storage>
class dMatrixT
{
    template<typename... Dimensions>
    dMatrixT(Dimensions... dims) : rank(sizeof...(Dimensions)), dimensions(rank)
    {
        static_assert((!std::is_same_v<Dimensions, std::initializer_list<Storage>> && ...),
            "Initializer list should use the initializer list constructor");

        const int ds[] = { dims... };
        uint n = 1;
        for (uint i = 0; i < rank; ++i)
        {
            dimensions[i] = ds[i];
            n *= ds[i];
        }
        data.resize(n);
    }

public:

    template<typename... Dimensions>
    static dMatrixT<Storage> Random(Dimensions... dims)
    {
        dMatrixT<Storage> m(dims...);
        for (Storage& d : m.data)
        {
            d = static_cast<Storage>(rand()) / static_cast<Storage>(RAND_MAX);
        }
        return m;
    }

    template<typename... Dimensions>
    static dMatrixT<Storage> NormalDistribution(const Storage& mean, const Storage& stddev, Dimensions... dims)
    {
        dMatrixT<Storage> m(dims...);
        std::default_random_engine generator;
        std::normal_distribution<Storage> distribution(mean, stddev);
        for (Storage& d : m.data)
        {
            d = distribution(generator);
        }
        return m;
    }

    template<typename... Dimensions>
    static dMatrixT<Storage> Dims(Dimensions... dims)
    {
        dMatrixT<Storage> m(dims...);
        m.zero_();
        return m;
    }

    template<typename... Dimensions>
    static dMatrixT<Storage> Zeros(Dimensions... dims)
    {
        dMatrixT<Storage> m(dims...);
        m.zero_();
        return m;
    }

    template<typename... Dimensions>
    static dMatrixT<Storage> Ones(Dimensions... dims)
    {
        dMatrixT<Storage> m(dims...);
        m.ones_();
        return m;
    }


    dMatrixT(const std::vector<Storage>& v) : rank(1), dimensions(1, uint(v.size())), data(v)
    {
    }

    dMatrixT(const std::vector<std::vector<Storage>>& v) : rank(2), dimensions(2, uint(v.size()))
    {
        for (const auto& row : v)
        {
            assert(row.size() == v[0].size());
        }
        for (const auto& row : v)
        {
            data.insert(data.end(), row.begin(), row.end());
        }
    }

    dMatrixT(std::initializer_list<Storage> list) : rank(1)
    {
        dimensions.push_back(uint(list.size()));
        data.resize(list.size());
        std::copy(list.begin(), list.end(), data.begin());
    }

    dMatrixT(std::initializer_list<std::initializer_list<Storage>> list) : rank(2)
    {
        dimensions.push_back(list.size());
        dimensions.push_back(list[0].size());
        data.resize(list.size());

        auto it = list.begin();
        for (std::size_t i = 0; i < list.size(); ++i, ++it) {
            data[i].resize(it->size());
            std::copy(it->begin(), it->end(), data[i].begin());
        }
    }


    uint size() const
    {
        return uint(data.size());
    }

    uint bytes() const
    {
        return uint(data.size() * sizeof(double));
    }

    uint dim(uint i) const
    {
        return dimensions[i];
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

    Storage mean() const
    {
        Storage sum = 0;
        for (const Storage& d : data)
        {
            sum += d;
        }
        return sum / data.size();
    }

    Storage sum() const
    {
        Storage sum = 0;
        for (const Storage& d : data)
        {
            sum += d;
        }
        return sum;
    }

    Storage max() const
    {
        Storage max = data[0];
        for (const Storage& d : data)
        {
            if (d > max)
            {
                max = d;
            }
        }
        return max;
    }

    Storage min() const
    {
        Storage min = data[0];
        for (const Storage& d : data)
        {
            if (d < min)
            {
                min = d;
            }
        }
        return min;
    }

    Storage mse() const
    {
        Storage sum = 0;
        for (const Storage& d : data)
        {
            sum += d * d;
        }
        return sum / data.size();
    }


    template<typename... Dimensions>
    void view_(Dimensions... dims)
    {
        const int ds[] = { dims... };
        const int numDims = sizeof...(Dimensions);
        assert(numDims == rank);

        uint n = 1;
        for (uint i = 0; i < rank; ++i)
        {
            dimensions[i] = ds[i];
            n *= ds[i];
        }
        assert(n == data.size());
    }

    template<typename... Indices>
    Storage& operator()(Indices... indices)
    {
        const int inds[] = { indices... };
        const uint n = sizeof...(Indices);
        assert(n == rank);

        for (uint i = 0; i < n; ++i)
        {
            assert(inds[i] >= 0 && uint(inds[i]) < dimensions[i]);
        }

        int index = 0;
        for (int i = 0; i < n; ++i)
        {
            index = index * dimensions[i] + inds[i];
        }
        return data[index];
    }

    template<typename... Indices>
    const Storage& operator()(Indices... indices) const
    {
        return const_cast<dMatrixT*>(this)->operator()(indices...);
    }

    dMatrixT<Storage>& operator=(const dMatrixT<Storage>& other)
    {
        if (this != &other)
        {
            assert(rank == other.rank);
            assert(dimensions == other.dimensions);
            data = other.data;
        }
        return *this;
    }

    dMatrixT<Storage> operator-(const dMatrixT<Storage>& other) const
    {
        assert(rank == other.rank);
        assert(dimensions == other.dimensions);
        dMatrixT<Storage> result = *this;
        for (uint i = 0; i < size(); ++i)
        {
            result.data[i] -= other.data[i];
        }
        return result;
    }

    dMatrixT<Storage> operator+(const dMatrixT<Storage>& other) const
    {
        assert(rank == other.rank);
        assert(dimensions == other.dimensions);
        dMatrixT<Storage> result = *this;
        for (uint i = 0; i < size(); ++i)
        {
            result.data[i] += other.data[i];
        }
        return result;
    }

    // element wise multiplication
    dMatrixT<Storage> operator*(const dMatrixT<Storage>& other) const
    {
        assert(rank == other.rank);
        assert(dimensions == other.dimensions);
        dMatrixT<Storage> result = *this;
        for (uint i = 0; i < size(); ++i)
        {
            result.data[i] *= other.data[i];
        }
        return result;
    }

    dMatrixT<Storage> operator/(const dMatrixT<Storage>& other) const
    {
        assert(rank == other.rank);
        assert(dimensions == other.dimensions);
        dMatrixT<Storage> result = *this;
        for (uint i = 0; i < size(); ++i)
        {
            result.data[i] /= other.data[i];
        }
        return result;
    }

    dMatrixT<Storage> operator*(const Storage& scalar) const
    {
        dMatrixT<Storage> result = *this;
        for (uint i = 0; i < size(); ++i)
        {
            result.data[i] *= scalar;
        }
        return result;
    }

    template<typename... Indices>
    dMatrixT<Storage> Row(Indices... indices) const
    {
        assert(sizeof...(indices) == rank - 1);
        const int inds[] = { indices... };

        dMatrixT<Storage> result = dMatrixT<Storage>::Dims(int(dimensions[0]));
        uint index = 0;
        for (uint d = 1; d < rank; d++)
        {
            index = inds[d-1] * dimensions[d]; // ugh multi dimensional arrays!
        }

        // copy the row
        for (uint i = 0; i < dimensions[0]; ++i)
        {
            result.data[i] = data[index + i];
        }
        return result;
    }

    const uint rank;
    std::vector<uint> dimensions;
    std::vector<Storage> data;
};

template <typename Storage>
dMatrixT<Storage> MatrixMultiply(const dMatrixT<Storage>& a, const dMatrixT<Storage>& b)
{
    assert(a.rank == 2);
    assert(b.rank == 2);
    assert(a.dimensions[1] == b.dimensions[0]);

    dMatrixT<Storage> result = dMatrixT<Storage>::Dims(int(a.dimensions[0]), int(b.dimensions[1]));
    uint resultIndex = 0;
    for (uint i = 0; i < a.dimensions[0]; ++i)
    {
        for (uint j = 0; j < b.dimensions[1]; ++j)
        {
            Storage sum = 0;
            for (uint k = 0; k < a.dimensions[1]; ++k)
            {
                sum += a(int(i), int(k)) * b(int(k), int(j));
            }
            result.data[resultIndex++] = sum;
        }
    }
    return result;
}

template <typename Storage>
Storage DotProduct(const dMatrixT<Storage>& a, const dMatrixT<Storage>& b)
{
    assert(a.rank == 1);
    assert(b.rank == 1);
    assert(a.size() == b.size());

    Storage result = 0;
    for (uint i = 0; i < a.size(); ++i)
    {
        result += a(int(i)) * b(int(i));
    }
    return result;
}

using  dMat = dMatrixT<float>;
