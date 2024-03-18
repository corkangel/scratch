#pragma once

#include <cassert>
#include <array>
#include <iostream>
#include <random>
#include <sstream>
#include <cstdarg>

#include "utils.h"

#define sTENSOR_MAX_DIMENSIONS 4

class sTensor;

class sTensorCellIterator
{
    sTensor& _tensor;
    uint _pos;

public:
    sTensorCellIterator(sTensor& tensor, bool end = false);

    bool operator!=(const sTensorCellIterator& other) const;

    float& operator*();

    // prefix increment
    sTensorCellIterator operator++();

    // postfix increment
    sTensorCellIterator operator++(int);
};

class sTensorRowIterator
{
    sTensor& _tensor;
    uint _row;

public:
    sTensorRowIterator(sTensor& tensor, uint row = 0);

    bool operator!=(const sTensorRowIterator& other) const;

    sTensor operator*();

    // prefix increment
    sTensorRowIterator operator++();

    // postfix increment
    sTensorRowIterator operator++(int);

    uint row() const
    {
        return _row;
    }
};


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
    sTensor(Dimensions... dimensions) :
        _rank(sizeof...(dimensions)), _storageOwned(true)
    {
        static_assert(sizeof...(dimensions) <= sTENSOR_MAX_DIMENSIONS, "Too many dimensions");
        const uint data[] = { dimensions... };
        Init(data);
    }

    template <typename... Dimensions, typename std::enable_if<(std::is_same_v<Dimensions, int> && ...), int>::type = 0>
    sTensor(Dimensions... dimensions) : 
        _rank(sizeof...(dimensions)), _storageOwned(true)
    {
        static_assert(sizeof...(dimensions) <= sTENSOR_MAX_DIMENSIONS, "Too many dimensions");
        const int data[] = { dimensions... };
        Init(reinterpret_cast<const uint*>(data));
    }

    sTensor(const uint rank, const uint* dimensions) : 
        _rank(rank), _storageOwned(true)
    {
        Init(dimensions);
    }

    sTensor(const sTensor& other)
        : _rank(other._rank), _storageSize(other._storageSize), _storageOwned(true)
    {
        Init(other._dimensions);

        memcpy(_storage, other._storage, _storageSize * sizeof(float));
    }

    ~sTensor()
    {
        if (_storageOwned)
        {
            assert(_storage != nullptr);
            delete[] _storage;
            _storage = nullptr;
        }
    }

    // ---------------- static constructors -----------------
    
    template<typename... Dimensions> static sTensor Dims(Dimensions... dims)
    {
        sTensor result(dims...);
        return result;
    }

    template<typename... Dimensions> static sTensor Fill(const float value, Dimensions... dims)
    {
        sTensor result(dims...);
        result.fill_(value);
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

    sTensor& fill_(float value)
    {
        for (uint i = 0; i < _storageSize; i++)
            _storage[i] = value;
        return *this;
    }

    sTensor& zero_()
    {
        fill_(0.0f);
        return *this;
    }

    sTensor& ones_()
    {
        fill_(1.0f);
        return *this;
    }

    sTensor& gaussian_(float bandwidth)
    {
        for (uint i = 0; i < _storageSize; i++)
            _storage[i] = gaussian(_storage[i], bandwidth);
        return *this;
    }

    sTensor& pow_(uint power)
    {
        for (uint i = 0; i < _storageSize; i++)
            _storage[i] = float(pow(_storage[i], power));
        return *this;
    }

    sTensor& sqrt_()
    {
        for (uint i = 0; i < _storageSize; i++)
            _storage[i] = sqrt(_storage[i]);
        return *this;
    }

    sTensor& exp_()
    {
        for (uint i = 0; i < _storageSize; i++)
            _storage[i] = exp(_storage[i]);
        return *this;
    }

    sTensor& log_()
    {
        for (uint i = 0; i < _storageSize; i++)
            _storage[i] = log(_storage[i]);
        return *this;
    }

    sTensor& abs_()
    {
        for (uint i = 0; i < _storageSize; i++)
            _storage[i] = abs(_storage[i]);
        return *this;
    }

    sTensor& add_(const float value)
    {
        for (uint i = 0; i < _storageSize; i++)
            _storage[i] += value;
        return *this;
    }

    sTensor& subtract_(const float value)
    {
        for (uint i = 0; i < _storageSize; i++)
            _storage[i] += value;
        return *this;
    }

    sTensor& multiply_(const float value)
    {
        for (uint i = 0; i < _storageSize; i++)
            _storage[i] *= value;
        return *this;
    }

    sTensor& divide_(const float value)
    {
        for (uint i = 0; i < _storageSize; i++)
            _storage[i] /= value;
        return *this;
    }

    sTensor& random_()
    {
        for (uint i = 0; i < _storageSize; i++)
            _storage[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        return *this;
    }

    sTensor& normal_distribution_(const float mean, const float stddev)
    {
        std::default_random_engine generator;
        std::normal_distribution distribution(mean, stddev);
        for (uint i = 0; i < _storageSize; i++)
            _storage[i] = distribution(generator);
        return *this;
    }

    sTensor& integers_(int start)
    {
        for (uint i = 0; i < _storageSize; i++)
            _storage[i] = float(start++);
        return *this;
    }

    sTensor& linear_(float start, const float step)
    {
        for (uint i = 0; i < _storageSize; i++)
        {
            _storage[i] = start;
            start += step;
        }
        return *this;
    }

    template<typename... Dimensions>
    sTensor& view_(Dimensions... dims)
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
        return *this;
    }

    // removes all dimensions of size 1
    sTensor& squeeze_()
    {
        uint newRank = 0;
        for (uint i = 0; i < _rank; i++)
        {
            if (_dimensions[i] > 1)
            {
                _dimensions[newRank++] = _dimensions[i];
            }
        }
        _rank = newRank;
        return *this;
    }

    // adds a dimension of size 1 at the specified position
    sTensor& unsqueeze_(uint dim)
    {
        assert(dim < _rank);
        for (uint i = _rank; i > dim; i--)
        {
            _dimensions[i] = _dimensions[i - 1];
        }
        _dimensions[dim] = 1;
        _rank++;
        return *this;
    }

    sTensor& cat0_(const sTensor& other)
    {
        assert(_storageOwned);

        assert(_rank == other._rank);
        assert(dim(1) == other.dim(1));
        _dimensions[0] += other.dim(0);

        float* newStorage = new float[_storageSize + other._storageSize];
        memcpy(newStorage, _storage, _storageSize * sizeof(float));
        memcpy(newStorage + _storageSize, other._storage, other._storageSize * sizeof(float));
        delete[] _storage;

        _storage = newStorage;
        _storageSize += other._storageSize;
        return *this;
    }

    sTensor& set_row_(uint row, const sTensor& other)
    {
        assert(_rank-1 == other.rank());

        uint n = 1;
        for (uint i = 1; i < _rank; i++)
        {
            n *= _dimensions[i];
        }
        uint start = row * n;

        for (uint i = 0; i < other.size(); i++)
        {
            _storage[start + i] = other._storage[i];
        }
        return *this;
    }

    sTensor& transpose_()
    {
        assert(_rank == 2);
        float* newStorage = new float[_storageSize];
        for (uint r = 0; r < dim(0); r++)
        {
            for (uint c = 0; c < dim(1); c++)
            {
                newStorage[c * dim(0) + r] = operator()(r, c);
            }
        }
        delete[] _storage;
        _storage = newStorage;

        uint temp = _dimensions[0];
        _dimensions[0] = _dimensions[1];
        _dimensions[1] = temp;
        return *this;
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

    float mse() const
    {
        float result = 0.0f;
        for (uint i = 0; i < _storageSize; i++)
            result += float(pow(_storage[i], 2));
        return result / _storageSize;
    }

    float rmse() const
    {
        return sqrt(mse());
    }

    float min() const
    {
        float result = _storage[0];
        for (uint i = 1; i < _storageSize; i++)
            result = std::min(result, _storage[i]);
        return result;
    }

    float max() const
    {
        float result = _storage[0];
        for (uint i = 1; i < _storageSize; i++)
            result = std::max(result, _storage[i]);
        return result;
    }

    // ---------------- operators -----------------

    bool operator==(const sTensor& other)
    {
        bool result = (_rank == other._rank);

        for (uint i = 0; i < _rank; i++)
            result &= (_dimensions[i] == other._dimensions[i]); 

        for (uint i = 0; i < _storageSize; i++)
            result &= (_storage[i] == other._storage[i]);

        return result;
    }

    bool operator!=(const sTensor& other)
    {
        return !operator==(other);
    }

    sTensor& operator=(const sTensor& other)
    {
        assert(_rank == other._rank);
        for (uint i = 0; i < _rank; i++)
            assert(_dimensions[i] == other._dimensions[i]);

        memcpy(_storage, other._storage, _storageSize * sizeof(float));
        return *this;
    }


    template<typename... Indices, typename std::enable_if<(std::is_same_v<Indices, int> && ...), int>::type = 0>
    float& operator()(Indices... indices)
    {
        const int inds[] = { indices... };
        const uint n = sizeof...(Indices);
        assert(n == _rank);

        for (uint i = 0; i < n; ++i)
        {
            assert(inds[i] >= 0 && uint(inds[i]) < _dimensions[i]);
        }

        int index = 0;
        for (int i = 0; i < n; ++i)
        {
            index = index * _dimensions[i] + inds[i];
        }
        return _storage[index];
    }

    template<typename... Indices, typename std::enable_if<(std::is_same_v<Indices, uint> && ...), uint>::type = 0>
    float& operator()(Indices... indices)
    {
        const uint inds[] = { indices... };
        const uint n = sizeof...(Indices);
        assert(n == _rank);

        for (uint i = 0; i < n; ++i)
        {
            assert(inds[i] >= 0 && uint(inds[i]) < _dimensions[i]);
        }

        int index = 0;
        for (int i = 0; i < n; ++i)
        {
            index = index * _dimensions[i] + inds[i];
        }
        return _storage[index];
    }

    template<typename... Indices, typename std::enable_if<(std::is_same_v<Indices, uint> && ...), uint>::type = 0>
    const float& operator()(Indices... indices) const
    {
        return const_cast<sTensor*>(this)->operator()(indices...);
    }

    template<typename... Indices, typename std::enable_if<(std::is_same_v<Indices, int> && ...), int>::type = 0>
    const float& operator()(Indices... indices) const
    {
        return const_cast<sTensor*>(this)->operator()(indices...);
    }

    // ---------------- tensor math operators -----------------

    using sTensorOp = void(*)(float&, const float);

    void apply_dimension(const sTensor& other, sTensorOp f, uint dim, uint& index1, uint& index2)
    {
        if (dim >= _rank)
            return;

        bool broadcast = (_dimensions[dim] != other._dimensions[dim] && other._dimensions[dim] == 1);
        if (broadcast)
        {
            int debug = 1;
        }
        for (uint i = 0; i < _dimensions[dim]; i++)
        {
            if (dim == _rank - 1)
            {
                f(_storage[index1], other._storage[index2]);
                index1++; 
                if (!broadcast)
                {
                    index2++; // broadcasted dimension
                }
            }
            else
            {
                apply_dimension(other, f, dim + 1, index1, index2);
                if (broadcast)
                {
                    uint n = 1;
                    for (uint i = dim + 1; i < _rank; i++)
                        n *= _dimensions[i];
                    index2 -= n; // broadcasted dimension
                }
            }
        }

        if (dim == _rank - 1 && broadcast)
        {
            index2 += other._dimensions[dim];
        }

    }

    void apply_(const sTensor& other, sTensorOp f)
    {
        assert(_rank == other._rank);

        uint index1 = 0;
        uint index2 = 0;
        apply_dimension(other, f, 0, index1, index2);
        assert(index1 == _storageSize);
        assert(index2 == other._storageSize || index2 == 0);

        //for (uint i = 0; i < _storageSize; i++)
       //  f(_storage[i], other._storage[i]);
    }

    sTensor operator+(const sTensor& other) const
    {
        sTensor result = *this;
        result.apply_(other, [](float& a, const float b) { a += b; });
        return result;
    }

    sTensor operator-(const sTensor& other) const
    {
        sTensor result = *this;
        result.apply_(other, [](float& a, const float b) { a -= b; });
        return result;
    }

    sTensor operator/(const sTensor& other) const
    {
        sTensor result = *this;
        result.apply_(other, [](float& a, const float b) { a /= b; });
        return result;
    }

    sTensor operator*(const sTensor& other) const
    {
        sTensor result = *this;
        result.apply_(other, [](float& a, const float b) { a *= b; });
        return result;
    }

    // sum of all elements in each row
    sTensor sum_rows()
    {
        assert(_rank == 2);
        const uint ncols = dim(1);

        sTensor result = Dims(uint(1), ncols);

        for (uint c = 0; c < ncols; c++)
        {
            float sum = 0;
            for (uint r = 0; r < dim(0); r++)
            {
                sum += operator()(r, c);
            }
            result(uint(0), c) = sum;
        }
        return result;
    }

    // sum of all elements in each column
    sTensor sum_columns()
    {
        assert(_rank == 2);
        const uint nrows = dim(0);
        sTensor result = Dims(nrows, uint(1));

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

    // ---------------- tensor scalar operators -----------------

    sTensor operator+(const float value) const
    {
        sTensor result = *this;
        for (uint i = 0; i < _storageSize; i++)
            result._storage[i] += value;
        return result;
    }

    sTensor operator-(const float value) const
    {
        sTensor result = *this;
        for (uint i = 0; i < _storageSize; i++)
            result._storage[i] -= value;
        return result;
    }

    sTensor operator*(const float value) const
    {
        sTensor result = *this;
        for (uint i = 0; i < _storageSize; i++)
            result._storage[i] *= value;
        return result;
    }

    sTensor operator/(const float value) const
    {
        sTensor result = *this;
        for (uint i = 0; i < _storageSize; i++)
            result._storage[i] /= value;
        return result;
    }

    // ---------------- utils -----------------

    sTensor MatMult(const sTensor& other)
    {
        assert(_rank == 2);
        assert(other._rank == 2);
        for (uint i = 0; i < _rank; i++) assert(_dimensions[i] == other._dimensions[i]);

        const uint nrows = dim(0);
        const uint ncols = dim(1);
        assert(nrows == ncols);

        sTensor result = Dims(nrows, ncols);
        for (uint r = 0; r < nrows; r++)
        {
            for (uint c = 0; c < ncols; c++)
            {
                float sum = 0;
                for (uint i = 0; i < ncols; i++)
                {
                    sum += (_storage[r * ncols + i] * other._storage[i * ncols + c]);
                    //sum += operator()(r, i) * other(i, c); - equivalent
                }
                result(r, c) = sum;
            }
        }
        return result;
    }

    float DotProduct(const sTensor& other)
    {
        assert(_rank == 1);
        assert(other._rank == 1);
        assert(size() == other.size());

        float result = 0;
        for (uint i = 0; i < size(); ++i)
        {
            result += _storage[i] * other._storage[i];
        }
        return result;
    }

    sTensor Transpose() const
    {
        assert(_rank == 2);
        sTensor result = Dims(dim(1), dim(0));
        for (uint r = 0; r < dim(0); r++)
        {
            for (uint c = 0; c < dim(1); c++)
            {
                result(c, r) = operator()(r, c);
            }
        }
        return result;
    }

    sTensor clone() const
    {
        sTensor result(_rank, _dimensions);
        memcpy(result._storage, _storage, _storageSize * sizeof(float));
        return result;
    }

    sTensor clone_empty() const
    {
        uint dims[sTENSOR_MAX_DIMENSIONS];
        memcpy(dims, _dimensions, _rank * sizeof(uint));
        dims[_rank-1] = 0; // last dimension is 0

        sTensor result(_rank, dims);
        return result;
    }

    sTensor row(const uint row)
    {
        assert(_rank == 2);
        assert(row < dim(0));

        sTensor result = Dims(uint(1), dim(1));
        for (uint c = 0; c < dim(1); c++)
        {
            result(uint(0), c) = operator()(row, c);
        }
        return result;
    }

    sTensor column(const uint col) const
    {
        assert(_rank == 2);
        assert(col < dim(1));

        sTensor result = Dims(dim(0), uint(1));
        for (uint r = 0; r < dim(0); r++)
        {
            result(r, uint(0)) = operator()(r, col);
        }
        return result;
    }

    sTensor random_sample_rows(const float p)
    {
        assert(_rank == 2);
        assert(p >= 0.0f && p <= 1.0f);
        const uint count = uint(p * dim(0));

        sTensor result = Dims(count, dim(1));
        for (uint i = 0; i < count; i++)
        {
            const float fmax = float(RAND_MAX) + 100; // +100 to avoid 100%
            const uint row = uint((rand() / fmax) * dim(0));
            const uint index = row * dim(1);
            _memccpy(result._storage + i * dim(1), _storage + index, dim(1), sizeof(float));
        }
        return result;
    }

    sTensor slice_rows(const uint start, const uint end) const
    {
        assert(start < end);
        assert(end <= dim(0));

        uint n = 1;
        for (uint i = 1; i < _rank; i++)
        {
            n *= _dimensions[i];
        }

        sTensor result = Dims(end - start, dim(1));
        for (uint i = start; i < end; i++)
        {
            const uint index = i * dim(1);
            _memccpy(result._storage + (i - start) * dim(1), _storage + index, dim(1), sizeof(float));
        }
        return result;
    }
    
    // ---------------- iterators -----------------

    friend class sTensorCellIterator;
    friend class sTensorRowIterator;

    sTensorCellIterator begin_cells()
    {
        return sTensorCellIterator(*this);
    }

    sTensorCellIterator end_cells()
    {
        return sTensorCellIterator(*this, true);
    }

    sTensorRowIterator begin_rows()
    {
        return sTensorRowIterator(*this);
    }

    sTensorRowIterator end_rows()
    {
        return sTensorRowIterator(*this, _dimensions[0]);
    }
};


std::ostream& operator<<(std::ostream& os, const sTensor& m);

void slog(const std::string& msg);
void slog(const char* msg, const sTensor& m);
void slog(const std::stringstream& ss);
void slog(const char* format, ...);
const std::vector<std::string>& get_logs();
