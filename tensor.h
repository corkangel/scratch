#pragma once

#include "utils.h"

#include <random>

class Tensor
{
    template<typename... Dimensions, typename std::enable_if<(std::is_same_v<Dimensions, uint> && ...), uint>::type = 0>
    Tensor(Dimensions... dims) : rank(sizeof...(Dimensions)), dimensions(rank)
    {
        const uint ds[] = { dims... };
        uint n = 1;
        for (uint i = 0; i < rank; ++i)
        {
            dimensions[i] = ds[i];
            n *= ds[i];
        }
        data.resize(n);
    }

    template<typename... Dimensions, typename std::enable_if<(std::is_same_v<Dimensions, int> && ...), int>::type = 0>
    Tensor(Dimensions... dims) : rank(sizeof...(Dimensions)), dimensions(rank)
    {
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
    static Tensor Random(Dimensions... dims)
    {
        Tensor m(dims...);
        for (float& d : m.data)
        {
            d = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
        return m;
    }

    template<typename... Dimensions>
    static Tensor NormalDistribution(const float& mean, const float& stddev, Dimensions... dims)
    {
        Tensor m(dims...);
        std::default_random_engine generator;
        std::normal_distribution distribution(mean, stddev);
        for (float& d : m.data)
        {
            d = distribution(generator);
        }
        return m;
    }

    template<typename... Dimensions>
    static Tensor Dims(Dimensions... dims)
    {
        Tensor m(dims...);
        m.zero_();
        return m;
    }

    template<typename... Dimensions>
    static Tensor Zeros(Dimensions... dims)
    {
        Tensor m(dims...);
        m.zero_();
        return m;
    }

    template<typename... Dimensions>
    static Tensor Ones(Dimensions... dims)
    {
        Tensor m(dims...);
        m.ones_();
        return m;
    }


    Tensor(const std::vector<float>& v) : rank(1), dimensions(1, uint(v.size())), data(v)
    {
    }

    Tensor(const std::vector<std::vector<float>>& v) : rank(2), dimensions(2, uint(v.size()))
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

    Tensor(std::initializer_list<float> list) : rank(1)
    {
        dimensions.push_back(uint(list.size()));
        data.resize(list.size());
        std::copy(list.begin(), list.end(), data.begin());
    }

    Tensor(std::initializer_list<std::initializer_list<float>> list) : rank(2)
    {
        dimensions.push_back(uint(list.size()));
        dimensions.push_back(uint(list.begin()->size()));
        for (const auto& row : list)
        {
            assert(row.size() == list.begin()->size());
        }
        for (const auto& row : list)
        {
            data.insert(data.end(), row.begin(), row.end());
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

    void all_(float value)
    {
        for (float& d : data)
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

    void squeeze_()
    {
        rank -= 1;
        dimensions.pop_back();
    }
    void unsqueeze_(uint dim)
    {
        rank += 1;
        dimensions.insert(dimensions.begin() + dim, 1);
    }

    void pow_(uint p)
    {
        for (float& d : data)
        {
            d = float(std::pow(d, p));
        }
    }

    void sqrt_()
    {
        for (float& d : data)
        {
            d = std::sqrt(d);
        }
    }

    void gaussian_(float bw)
    {
        for (float& d : data)
        {
            d = gaussian(d, bw);
        }
    }

    Tensor sum1()
    {
        const uint nrows = dim(0);
        Tensor result = Dims(nrows, uint(1));

        for (uint r = 0; r < nrows; r++)
        {
            float sum = 0;
            for (uint c = 0; c < dim(1); c++)
            {
                sum += operator()(r, c);
            }
            result(r, uint(0)) = sum;
        }
        return result;
    }

    float mean() const
    {
        float sum = 0;
        for (const float& d : data)
        {
            sum += d;
        }
        return sum / data.size();
    }

    float sum() const
    {
        float sum = 0;
        for (const float& d : data)
        {
            sum += d;
        }
        return sum;
    }

    float max() const
    {
        float max = data[0];
        for (const float& d : data)
        {
            if (d > max)
            {
                max = d;
            }
        }
        return max;
    }

    float min() const
    {
        float min = data[0];
        for (const float& d : data)
        {
            if (d < min)
            {
                min = d;
            }
        }
        return min;
    }

    float mse() const
    {
        float sum = 0;
        for (const float& d : data)
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

    void cat0_(const Tensor& other)
    {
        assert(rank == 2);
        assert(rank == other.rank);
        assert(dim(1) == other.dim(1));
        dimensions[0] += other.dim(0);

        data.insert(data.end(), other.data.begin(), other.data.end());
    }

    template<typename... Indices, typename std::enable_if<(std::is_same_v<Indices, uint> && ...), uint>::type = 0>
    float& operator()(Indices... indices)
    {
        const uint inds[] = { indices... };
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

    template<typename... Indices, typename std::enable_if<(std::is_same_v<Indices, int> && ...), int>::type = 0>
    float& operator()(Indices... indices)
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

    template<typename... Indices, typename std::enable_if<(std::is_same_v<Indices, uint> && ...), uint>::type = 0>
    const float& operator()(Indices... indices) const
    {
        return const_cast<Tensor*>(this)->operator()(indices...);
    }

    template<typename... Indices, typename std::enable_if<(std::is_same_v<Indices, int> && ...), int>::type = 0>
    const float& operator()(Indices... indices) const
    {
        return const_cast<Tensor*>(this)->operator()(indices...);
    }

    Tensor& operator=(const Tensor& other)
    {
        if (this != &other)
        {
            assert(rank == other.rank);
            assert(dimensions == other.dimensions);
            data = other.data;
        }
        return *this;
    }

    Tensor operator-(const Tensor& other) const
    {
        assert(rank == other.rank);
        assert(dimensions == other.dimensions);
        Tensor result = *this;
        for (uint i = 0; i < size(); ++i)
        {
            result.data[i] -= other.data[i];
        }
        return result;
    }

    Tensor operator+(const Tensor& other) const
    {
        assert(rank == other.rank);
        assert(dimensions == other.dimensions);
        Tensor result = *this;
        for (uint i = 0; i < size(); ++i)
        {
            result.data[i] += other.data[i];
        }
        return result;
    }

    // element wise multiplication
    Tensor operator*(const Tensor& other) const
    {
        assert(rank == other.rank);
        assert(dimensions == other.dimensions);
        Tensor result = *this;
        for (uint i = 0; i < size(); ++i)
        {
            result.data[i] *= other.data[i];
        }
        return result;
    }

    Tensor operator/(const Tensor& other) const
    {
        assert(rank == other.rank);
        assert(dimensions == other.dimensions);
        Tensor result = *this;
        for (uint i = 0; i < size(); ++i)
        {
            result.data[i] /= other.data[i];
        }
        return result;
    }

    Tensor operator*(const float& scalar) const
    {
        Tensor result = *this;
        for (uint i = 0; i < size(); ++i)
        {
            result.data[i] *= scalar;
        }
        return result;
    }

    Tensor& operator+=(const Tensor& other)
    {
        assert(rank == other.rank);
        assert(dimensions == other.dimensions);
        for (uint i = 0; i < size(); ++i)
        {
            data[i] += other.data[i];
        }
        return *this;
    }

    Tensor& operator-=(const Tensor& other)
    {
        assert(rank == other.rank);
        assert(dimensions == other.dimensions);
        for (uint i = 0; i < size(); ++i)
        {
            data[i] -= other.data[i];
        }
        return *this;
    }

    Tensor& operator*=(const Tensor& other)
    {
        assert(rank == other.rank);
        assert(dimensions == other.dimensions);
        for (uint i = 0; i < size(); ++i)
        {
            data[i] *= other.data[i];
        }
        return *this;
    }

    Tensor& operator/=(const Tensor& other)
    {
        assert(rank == other.rank);
        assert(dimensions == other.dimensions);
        for (uint i = 0; i < size(); ++i)
        {
            data[i] /= other.data[i];
        }
        return *this;
    }



    // this is a copy!! bad
    static Tensor Broadcast0(const Tensor& m, const uint num)
    {
        assert(m.dim(0) == 1);
        Tensor result = Tensor::Dims(num, m.dim(1));

        uint index = 0;
        for (uint i = 0; i < num; ++i)
        {
            for (uint j = 0; j < m.dim(1); ++j)
            {
                result.data[index++] = m.data[j];
            }
        }
        return result;
    }

    static Tensor Broadcast1(const Tensor& m, const uint num)
    {
        assert(m.dim(1) == 1);
        Tensor result = Tensor::Dims(m.dim(0), num);

        uint index = 0;
        const uint nrows = m.dim(0);
        for (uint i = 0; i < nrows; ++i)
        {
            for (uint j = 0; j < num; ++j)
            {
                result.data[index++] = m.data[i];
            }
        }

        return result;
    }

    template<typename... Indices, typename std::enable_if<(std::is_same_v<Indices, int> && ...), int>::type = 0>
    Tensor Row(Indices... indices) const
    {
        assert(sizeof...(indices) == rank - 1);
        const int inds[] = { indices... };
        const uint rowWidth = dimensions[rank - 1];

        Tensor result = Tensor::Dims(rowWidth);
        uint index = 0;
        for (uint d = 1; d < rank; d++)
        {
            index = inds[d - 1] * dimensions[d]; // ugh multi dimensional arrays!
        }

        // copy the row
        for (uint i = 0; i < rowWidth; ++i)
        {
            result.data[i] = data[index + i];
        }
        return result;
    }

    template<typename... Indices, typename std::enable_if<(std::is_same_v<Indices, uint> && ...), uint>::type = 0>
    Tensor Row(Indices... indices) const
    {
        assert(sizeof...(indices) == rank - 1);
        const uint inds[] = { indices... };
        const uint rowWidth = dimensions[rank - 1];

        Tensor result = Tensor::Dims(rowWidth);
        uint index = 0;
        for (uint d = 1; d < rank; d++)
        {
            index = inds[d - 1] * dimensions[d]; // ugh multi dimensional arrays!
        }

        // copy the row
        for (uint i = 0; i < rowWidth; ++i)
        {
            result.data[i] = data[index + i];
        }
        return result;
    }

    //void SetRow(const Tensor& row, uint index)
    //{
    //    assert(row.rank == rank-1);
    //}

    uint rank;
    std::vector<uint> dimensions;
    std::vector<float> data;
};

Tensor MatrixMultiply(const Tensor& a, const Tensor& b)
{
    assert(a.rank == 2);
    assert(b.rank == 2);
    assert(a.dimensions[0] == b.dimensions[0]);

    Tensor result = Tensor::Dims(a.dimensions[0], b.dimensions[1]);
    uint resultIndex = 0;
    for (uint i = 0; i < a.dimensions[0]; ++i)
    {
        for (uint j = 0; j < b.dimensions[1]; ++j)
        {
            float sum = 0;
            for (uint k = 0; k < a.dimensions[1]; ++k)
            {
                sum += a(i, k) * b(k, j);
            }
            result.data[resultIndex++] = sum;
        }
    }
    return result;
}

float DotProduct(const Tensor& a, const Tensor& b)
{
    assert(a.rank == 1);
    assert(b.rank == 1);
    assert(a.size() == b.size());

    float result = 0;
    for (uint i = 0; i < a.size(); ++i)
    {
        result += a(i) * b(i);
    }
    return result;
}

std::ostream& operator<<(std::ostream& os, const Tensor& m)
{
    uint n = 1;
    for (uint i = 1; i < m.rank; ++i)
    {
        n *= m.dim(i);
    }

    os << "dMat( dims:[";
    for (uint i = 0; i < m.rank; ++i)
    {
        os << m.dim(i);
        if (i != m.rank - 1)
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

        os << m.data[i];
        if (i != m.size() - 1)
        {
            os << ", ";
        }

    }
    os << "]";
    os << ")";
    return os;
}