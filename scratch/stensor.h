#pragma once

#include <algorithm>
#include <cassert>
#include <array>
#include <iostream>
#include <random>
#include <sstream>
#include <cstdarg>
#include <iomanip>
#include <memory>

#include "utils.h"
#include "matmul.ch"
#include "slog.h"

#define sTENSOR_MAX_DIMENSIONS max_tensor_dimensions

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

    uint pos() const
    {
        return _pos;
    }
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
    static uint idCounter;

    uint _id;
    uint _dimensions[sTENSOR_MAX_DIMENSIONS] = {};
    uint _rank;

    float* _storage;
    uint _storageSize;
    bool _storageOwned;
    const char* _label = nullptr;

    std::shared_ptr<sTensor> _grad;

    void Init(const uint* data)
    {
        assert(_rank <= sTENSOR_MAX_DIMENSIONS);
        memcpy(_dimensions, data, _rank * sizeof(uint));

        _storageSize = 1;
        for (uint i = 0; i < _rank; i++)
            _storageSize *= _dimensions[i];

        _storage = new float[_storageSize];

#if _DEBUG
        const float STUFF = 99.99f;
        for (uint i = 0; i < _storageSize; i++) _storage[i] = STUFF;
#endif
    }

    const timepoint now() const
    {
        if (!sTensor::enableAutoLog) return timepoint();

        return std::chrono::steady_clock::now();
    }

    sTensor& autolog(const char* label, const timepoint begin)
    {
        if (!sTensor::enableAutoLog) return *this;

        std::stringstream ss;
        ss << _id << ":";
        ss << std::fixed << std::setprecision(2) ;
        ss << (_label ? _label : "") << " / " << label << ": [";
        for (uint i = 0; i < _rank; i++)
        {
            ss << _dimensions[i];
            if (i < _rank - 1) ss << ", ";
        }
        ss << "] {";

        if (_rank == 0)
            ss << _storage[0];
        else
        {
            for (uint i = 0; i < 4; i++)
            {
                ss << _storage[i];
                if (i < _storageSize - 1) ss << ", ";
            }

            if (_storageSize > 4) ss << "...";

            for (uint i = 0; i < std::min(uint(4),_storageSize); i++)
            {
                uint pos = _storageSize - i - 1;
                ss << _storage[pos];
                if (pos < _storageSize - 1) ss << ", ";
            }
            ss << "}";
        }
        slog(ss.str());

        log_tensor_info(info(label, begin));
        return *this;
    }

public:

    static bool enableAutoLog;
    static sTensor null;

    template <typename... Dimensions, typename std::enable_if<(std::is_same_v<Dimensions, uint> && ...), uint>::type = 0>
    sTensor(Dimensions... dimensions) :
        _rank(sizeof...(dimensions)), _storageOwned(true)
    {
        static_assert(sizeof...(dimensions) <= sTENSOR_MAX_DIMENSIONS, "Too many dimensions");
        const uint data[] = { dimensions... };
        Init(data);
        _id = idCounter++;
    }

    template <typename... Dimensions, typename std::enable_if<(std::is_same_v<Dimensions, int> && ...), int>::type = 0>
    sTensor(Dimensions... dimensions) : 
        _rank(sizeof...(dimensions)), _storageOwned(true)
    {
        static_assert(sizeof...(dimensions) <= sTENSOR_MAX_DIMENSIONS, "Too many dimensions");
        const int data[] = { dimensions... };
        Init(reinterpret_cast<const uint*>(data));
        _id = idCounter++;
    }

    sTensor(const uint rank, const uint* dimensions, const uint id = 0, const char* label = nullptr) : 
        _rank(rank), _storageOwned(true)
    {
        Init(dimensions);
        if (id != 0) _id = id; else _id = idCounter++;
        if (label) _label = label;
    }

    sTensor(const sTensor& other)
        : _rank(other._rank), _storageSize(other._storageSize), _storageOwned(true), _label(other._label), _id(other._id)
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

    sTensor& set_label(const char* label)
    {
        _label = label;
        return *this;
    }

    void set_grad(const sTensor& grad)
    {
        _grad = std::make_shared<sTensor>(grad);
    }

    sTensor* grad() const
    {
        return _grad.get();
    }

    void zero_grad()
    {
        if (_grad.get() != nullptr)
            _grad->zero_();
    }

    // ---------------- static constructors -----------------
    
    static sTensor Empty()
    {
        constexpr uint dims[sTENSOR_MAX_DIMENSIONS] = { 0 };
        sTensor result(0, dims);
        return result;
    }

    static sTensor Rank(const uint rank)
    {
        constexpr uint dims[sTENSOR_MAX_DIMENSIONS] = { 0 };
        sTensor result(rank, dims);
        return result;
    }

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
        timepoint begin = now();
        fill_(0.0f);
        return autolog("zero_", begin);
    }

    sTensor& ones_()
    {
        timepoint begin = now();
        fill_(1.0f);
        return autolog("ones_", begin);
    }

    sTensor& gaussian_(float bandwidth)
    {
        timepoint begin = now();
        for (uint i = 0; i < _storageSize; i++)
            _storage[i] = gaussian(_storage[i], bandwidth);
        return autolog("gaussian_", begin);
    }

    sTensor& pow_(uint power)
    {
        timepoint begin = now();
        for (uint i = 0; i < _storageSize; i++)
            _storage[i] = float(pow(_storage[i], power));
        return autolog("pow_", begin);
    }

    sTensor& sqrt_()
    {
        timepoint begin = now();
        for (uint i = 0; i < _storageSize; i++)
            _storage[i] = sqrt(_storage[i]);
        return autolog("sqrt_", begin);
    }

    sTensor& exp_()
    {
        timepoint begin = now();
        for (uint i = 0; i < _storageSize; i++)
            _storage[i] = std::exp(_storage[i]);
        return autolog("exp_", begin);
    }

    sTensor& log_()
    {
        timepoint begin = now();
        for (uint i = 0; i < _storageSize; i++)
            _storage[i] = log(_storage[i]);
        return autolog("log_", begin);
    }

    sTensor& abs_()
    {
        timepoint begin = now();
        for (uint i = 0; i < _storageSize; i++)
            _storage[i] = abs(_storage[i]);
        return autolog("abs_", begin);
    }

    sTensor& add_(const float value)
    {
        timepoint begin = now();
        for (uint i = 0; i < _storageSize; i++)
            _storage[i] += value;
        return autolog("add_", begin);
    }

    sTensor& subtract_(const float value)
    {
        timepoint begin = now();
        for (uint i = 0; i < _storageSize; i++)
            _storage[i] += value;
        return autolog("subtract_", begin);
    }

    sTensor& multiply_(const float value)
    {
        timepoint begin = now();
        for (uint i = 0; i < _storageSize; i++)
            _storage[i] *= value;
        return autolog("multiply_", begin);
    }

    sTensor& divide_(const float value)
    {
        timepoint begin = now();
        for (uint i = 0; i < _storageSize; i++)
            _storage[i] /= value;
        return autolog("divide_", begin);
    }

    sTensor& random_()
    {
        timepoint begin = now();
        for (uint i = 0; i < _storageSize; i++)
            _storage[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        return autolog("random_", begin);
    }

    sTensor& normal_distribution_(const float mean, const float stddev)
    {
        timepoint begin = now();
        std::default_random_engine generator;
        std::normal_distribution distribution(mean, stddev);
        for (uint i = 0; i < _storageSize; i++)
            _storage[i] = distribution(generator);
        return autolog("normal_distribution_", begin);
    }

    sTensor& integers_(int start)
    {
        timepoint begin = now();
        for (uint i = 0; i < _storageSize; i++)
            _storage[i] = float(start++);
        return autolog("integers_", begin);
    }

    sTensor& linear_(float start, const float step)
    {
        timepoint begin = now();
        for (uint i = 0; i < _storageSize; i++)
        {
            _storage[i] = start;
            start += step;
        }
        return autolog("linear_", begin);
    }

    template<typename... Dimensions>
    sTensor& view_(Dimensions... dims)
    {
        timepoint begin = now();
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
        return autolog("view_", begin);
    }

    sTensor index_select(const sTensor& indices) const
    {
        timepoint begin = now();
        assert(_rank == 2 && indices.rank() == 1);

        const uint n = indices.size();
        sTensor result = sTensor::Dims(n, uint(1));
        for (uint i = 0; i < n; i++)
        {
            result.set2d(i, 0, get2d(i, uint(indices.get1d(i))));
        }
        return result.autolog("index_select", begin);
    }

    sTensor argmax() const
    {
        timepoint begin = now();

        const uint nrows = dim(0);
        const uint ncols = dim(1);
        sTensor result = sTensor::Dims(nrows, uint(1));

        for (uint i = 0; i < nrows; i++)
        {
            float max = get2d(i, 0);
            uint maxIndex = 0;
            for (uint j = 1; j < ncols; j++)
            {
                if (get2d(i, j) > max)
                {
                    max = get2d(i, j);
                    maxIndex = j;
                }
            }
            result.set2d(i, 0, float(maxIndex));
        }
        return result.autolog("argmax", begin);
    }

    // removes all dimensions of size 1

    sTensor equal(const sTensor& other) const
    {
        assert(_rank == other._rank);
        assert(_storageSize == other._storageSize);

        timepoint begin = now();
        sTensor result = clone();
        for (uint i = 0; i < _storageSize; i++)
            result._storage[i] = _storage[i] == other._storage[i] ? 1.0f : 0.0f;
        return result.autolog("equal", begin);
    }

    sTensor squeeze() const
    {
        sTensor result = clone_shallow();
        return result.squeeze_();
    }

    sTensor& squeeze_()
    {
        timepoint begin = now();
        uint newRank = 0;
        for (uint i = 0; i < _rank; i++)
        {
            if (_dimensions[i] > 1)
            {
                _dimensions[newRank++] = _dimensions[i];
            }
        }
        for (uint i = newRank; i < _rank; i++)
        {
            _dimensions[i] = 0;
        }
        _rank = newRank;
        return autolog("squeeze_", begin);
    }

    // adds a dimension of size 1 at the specified position

    sTensor unsqueeze(uint dim) const
    {
        sTensor result = clone_shallow();
        return result.unsqueeze_(dim);
    }

    sTensor& unsqueeze_(uint dim)
    {
        timepoint begin = now();
        assert(dim <= _rank);
        for (uint i = _rank; i > dim; i--)
        {
            _dimensions[i] = _dimensions[i - 1];
        }
        _dimensions[dim] = 1;
        _rank++;
        return autolog("unsqueeze_", begin);
    }

    sTensor& cat0_(const sTensor& other)
    {
        timepoint begin = now();
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
        return autolog("cat0_", begin);
    }

    sTensor& set_row_(uint row, const sTensor& other)
    {
        timepoint begin = now();
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
        return autolog("set_row_", begin);
    }

    sTensor& transpose_()
    {
        timepoint begin = now();
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
        return autolog("transpose_", begin);
    }

    sTensor& clamp_min_(const float v)
    {
        timepoint begin = now();
        for (uint i = 0; i < _storageSize; i++)
            _storage[i] = std::max(_storage[i], v);
        return autolog("clamp_min", begin);
    }

    sTensor& clamp_max_(const float v)
    {
        timepoint begin = now();
        for (uint i = 0; i < _storageSize; i++)
            _storage[i] = std::min(_storage[i], v);
        return autolog("clamp_max", begin);
    }

    // ---------------- scalar operations -----------------

    sTensor exp() const
    {
        sTensor result = clone();
        return result.exp_();
    }

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

    float std() const
    {
        float m = mean();
        float result = 0.0f;
        for (uint i = 0; i < _storageSize; i++)
            result += float(pow(_storage[i] - m, 2));
        return sqrt(result / _storageSize);
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
        timepoint begin = now();
        assert(_rank == other._rank || _rank == 0);

        if (_rank == 0)
        {
            _rank = other._rank;
            Init(other._dimensions);
        }
        for (uint i = 0; i < _rank; i++)
            assert(_dimensions[i] == other._dimensions[i]);

        memcpy(_storage, other._storage, _storageSize * sizeof(float));
        _label = other._label;
        _id = other._id;
        return autolog("operator=", begin);
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

    using sTensorOp = void(*)(float&, const float, const float);

private:
    sTensor apply_rank1(sTensor& result, const sTensor& other, sTensorOp f) const
    {
        for (uint i = 0; i < _dimensions[0]; i++)
            f(result(i), operator()(i), other(i));

        return result;
    }

    sTensor apply_rank2(sTensor& result, const sTensor& other, sTensorOp f) const
    {
        const uint maxDim0 = std::max(_dimensions[0], other._dimensions[0]);
        const uint maxDim1 = std::max(_dimensions[1], other._dimensions[1]);

        for (uint i = 0; i < maxDim0; i++)
        {
            for (uint j = 0; j < maxDim1; j++)
            {
                const uint left_i = (i >= _dimensions[0]) ? 0 : i;
                const uint left_j = (j >= _dimensions[1]) ? 0 : j;
                const uint right_i = (i >= other._dimensions[0]) ? 0 : i;
                const uint right_j = (j >= other._dimensions[1]) ? 0 : j;

                f(result(i, j), operator()(left_i, left_j), other(right_i, right_j));
            }
        }
        return result;
    }

    sTensor apply_rank3(sTensor& result, const sTensor& other, sTensorOp f) const
    {
        const uint maxDim0 = std::max(_dimensions[0], other._dimensions[0]);
        const uint maxDim1 = std::max(_dimensions[1], other._dimensions[1]);
        const uint maxDim2 = std::max(_dimensions[2], other._dimensions[2]);

        for (uint i = 0; i < maxDim0; i++)
        {
            for (uint j = 0; j < maxDim1; j++)
            {
                for (uint k = 0; k < maxDim2; k++)
                {
                    const uint left_i = (i >= _dimensions[0]) ? 0 : i;
                    const uint left_j = (j >= _dimensions[1]) ? 0 : j;
                    const uint left_k = (k >= _dimensions[2]) ? 0 : k;
                    const uint right_i = (i >= other._dimensions[0]) ? 0 : i;
                    const uint right_j = (j >= other._dimensions[1]) ? 0 : j;
                    const uint right_k = (k >= other._dimensions[2]) ? 0 : k;

                    f(result(i, j, k), operator()(left_i, left_j, left_k), other(right_i, right_j, right_k));
                }
            }
        }
        return result;
    }

    sTensor apply_rank4(sTensor& result, const sTensor& other, sTensorOp f) const
    {
        const uint maxDim0 = std::max(_dimensions[0], other._dimensions[0]);
        const uint maxDim1 = std::max(_dimensions[1], other._dimensions[1]);
        const uint maxDim2 = std::max(_dimensions[2], other._dimensions[2]);
        const uint maxDim3 = std::max(_dimensions[3], other._dimensions[3]);

        for (uint i = 0; i < maxDim0; i++)
        {
            for (uint j = 0; j < maxDim1; j++)
            {
                for (uint k = 0; k < maxDim2; k++)
                {
                    for (uint l = 0; l < maxDim3; l++)
                    {
                        const uint left_i = (i >= _dimensions[0]) ? 0 : i;
                        const uint left_j = (j >= _dimensions[1]) ? 0 : j;
                        const uint left_k = (k >= _dimensions[2]) ? 0 : k;
                        const uint left_l = (l >= _dimensions[3]) ? 0 : l;
                        const uint right_i = (i >= other._dimensions[0]) ? 0 : i;
                        const uint right_j = (j >= other._dimensions[1]) ? 0 : j;
                        const uint right_k = (k >= other._dimensions[2]) ? 0 : k;
                        const uint right_l = (l >= other._dimensions[3]) ? 0 : l;

                        f(result(i, j, k, l), operator()(left_i, left_j, left_k, left_l), other(right_i, right_j, right_k, right_l));
                    }
                }
            }
        }
        return result;
    }

 public:
    sTensor apply_(const sTensor& other, sTensorOp f) const
    {
        assert(_rank == other._rank);

        // broadcasting rules
        uint new_dims[sTENSOR_MAX_DIMENSIONS];
        for (uint d = 0; d < _rank; d++)
        {
            if (_dimensions[d] == 1 && other._dimensions[d] != 1)
            {
                new_dims[d] = other._dimensions[d];
            }
            else if (other._dimensions[d] == 1 && _dimensions[d] != 1)
            {
                new_dims[d] = _dimensions[d];
            }
            else
            {
                new_dims[d] = _dimensions[d];
                assert(_dimensions[d] == other._dimensions[d]);
            }
        }

        sTensor result(_rank, new_dims, _id, _label);
        result.set_label(other._label);

        if (_rank == 1) return apply_rank1(result, other, f);
        if (_rank == 2) return apply_rank2(result, other, f);
        if (_rank == 3) return apply_rank3(result, other, f);
        if (_rank == 4) return apply_rank4(result, other, f);
        assert(false);

        return result;
    }

    sTensor operator+(const sTensor& other) const
    {
        timepoint begin = now();
        return apply_(other, [](float& o, const float a, const float b) { o = a + b; }).autolog("operator+", begin);
    }

    sTensor operator-(const sTensor& other) const
    {
        timepoint begin = now();
        return apply_(other, [](float& o, const float a, const float b) { o = a - b; }).autolog("operator-", begin);
    }

    sTensor operator/(const sTensor& other) const
    {
        timepoint begin = now();
        return apply_(other, [](float& o, const float a, const float b) { if (b == 0.f || isnan(b) || isinf(b)) o = 0.f; else o = a / b; }).autolog("operator/", begin);
    }

    sTensor operator*(const sTensor& other) const
    {
        timepoint begin = now();
        return apply_(other, [](float& o, const float a, const float b) { o = a * b; }).autolog("operator*", begin);
    }

    // sum of all elements in each row - only works for 2x2 matrices
    sTensor sum_rows()
    {
        timepoint begin = now();
        assert(_rank == 2);
        const uint ncols = dim(1);

        sTensor result = Dims(uint(1), ncols);
        result.set_label(_label);

        for (uint c = 0; c < ncols; c++)
        {
            float sum = 0;
            for (uint r = 0; r < dim(0); r++)
            {
                sum += get2d(r, c);
            }
            result(uint(0), c) = sum;
        }
        return result.squeeze_().autolog("sum_rows", begin);
    }

    // sum of all elements in each column - only works for 2x2 matrices
    sTensor sum_columns()
    {
        timepoint begin = now();
        assert(_rank == 2);
        const uint nrows = dim(0);
        sTensor result = Dims(nrows, uint(1));
        result.set_label(_label);

        for (uint r = 0; r < nrows; r++)
        {
            float sum = 0;
            for (uint c = 0; c < dim(1); c++)
            {
                sum += operator()(r, c);
            }
            result(r, uint(0)) = sum;
        }
        return result.squeeze_().autolog("sum_columns", begin);
    }

    float get1d(const uint n) const
    {
        assert(_rank == 1);
        return _storage[n];
    }

    void set1d(const uint n, const float value)
    {
        assert(_rank == 1);
        _storage[n] = value;
    }

    void add1d(const uint n, const float value)
    {
        assert(_rank == 1);
        _storage[n] += value;
    }

    float get2d(const uint row, const uint col) const
    {
        assert(_rank == 2);
        return _storage[row * _dimensions[1] + col];
    }

    void set2d(const uint row, const uint col, const float value)
    {
        assert(_rank == 2);
        _storage[row * _dimensions[1] + col] = value;
    }

    void add2d(const uint row, const uint col, const float value)
    {
        assert(_rank == 2);
        _storage[row * _dimensions[1] + col] += value;
    }

 private:

    sTensor sum_rank2(sTensor& result, const uint dim)
    {
        uint resultIndices[1] = {};

        for (uint i = 0; i < _dimensions[0]; i++)
        {
            for (uint j = 0; j < _dimensions[1]; j++)
            {
                if (dim == 0)
                {
                    resultIndices[0] = j;
                }
                else if (dim == 1)
                {
                    resultIndices[0] = i;
                }
                result(resultIndices[0]) += operator()(i, j);
            }
        }
        return result;
    }

    sTensor sum_rank3(sTensor& result, const uint dim)
    {
        uint resultIndices[2] = {};

        for (uint i = 0; i < _dimensions[0]; i++)
        {
            for (uint j = 0; j < _dimensions[1]; j++)
            {
                for (uint k = 0; k < _dimensions[2]; k++)
                {
                    if (dim == 0)
                    {
                        resultIndices[0] = j;
                        resultIndices[1] = k;
                    }
                    else if (dim == 1)
                    {
                        resultIndices[0] = i;
                        resultIndices[1] = k;
                    }
                    else if (dim == 2)
                    {
                        resultIndices[0] = i;
                        resultIndices[1] = j;
                    }
                    result(resultIndices[0], resultIndices[1]) += operator()(i, j, k);
                }
            }
        }
        return result;
    }

    sTensor sum_rank4(sTensor& result, const uint dim)
    {
        uint resultIndices[3] = {};

        for (uint i = 0; i < _dimensions[0]; i++)
        {
            for (uint j = 0; j < _dimensions[1]; j++)
            {
                for (uint k = 0; k < _dimensions[2]; k++)
                {
                    for (uint l = 0; l < _dimensions[3]; l++)
                    {
                        if (dim == 0)
                        {
                            resultIndices[0] = j;
                            resultIndices[1] = k;
                            resultIndices[2] = l;
                        }
                        else if (dim == 1)
                        {
                            resultIndices[0] = i;
                            resultIndices[1] = k;
                            resultIndices[2] = l;
                        }
                        else if (dim == 2)
                        {
                            resultIndices[0] = i;
                            resultIndices[1] = j;
                            resultIndices[2] = l;
                        }
                        else if (dim == 3)
                        {
                            resultIndices[0] = i;
                            resultIndices[1] = j;
                            resultIndices[2] = k;
                        }
                        result(resultIndices[0], resultIndices[1], resultIndices[2]) += operator()(i, j, k, l);
                    }
                }
            }
        }
        return result;
    }

public:
    sTensor sum(const uint dim)
    {
        timepoint begin = now();
        assert(dim < _rank);
        uint new_dims[sTENSOR_MAX_DIMENSIONS];

        uint pos = 0;
        for (uint i = 0; i < _rank; i++)
        {
            if (i == dim) continue;
            new_dims[pos++] = _dimensions[i];
        }

        const uint dim_size = _dimensions[dim];
        sTensor result = sTensor::Zeros(_rank - 1, new_dims);
        result.set_label(_label);

        assert(_rank > 1);
        
        if (_rank == 2) return sum_rank2(result, dim).autolog("sum_rank2", begin);;
        if (_rank == 3) return sum_rank3(result, dim).autolog("sum_rank3", begin);
        if (_rank == 4) return sum_rank4(result, dim).autolog("sum_rank4", begin);
        
        return result.autolog("sum_dim", begin);
    }

    // should this squeeze the final dimension?. yes!
    sTensor sum_final_dimension()
    {
        timepoint begin = now();
        uint dim = _rank - 1;
        uint new_dims[sTENSOR_MAX_DIMENSIONS];
        memcpy(new_dims, _dimensions, _rank * sizeof(uint));
        new_dims[dim] = 1;

        sTensor result(_rank, new_dims);
        result.set_label(_label);

        const uint finalDimSize = _dimensions[dim];
        const uint nItems = _storageSize / finalDimSize;
        for (uint r = 0; r < nItems; r++)
        {
            float sum = 0;
            for (uint c = 0; c < finalDimSize; c++)
            {
                sum += _storage[r + c];
            }
            result._storage[r] = sum;
        }
        result.squeeze_();
        return result.autolog("sum_final_dimension", begin);
    }

    sTensor greater_than(const float value)
    {
        timepoint begin = now();
        sTensor result = clone();
        for (uint i = 0; i < _storageSize; i++)
            result._storage[i] = _storage[i] > value ? 1.0f : 0.0f;
        return result.autolog("greater_than", begin);
    }
    sTensor less_than(const float value)
    {
        timepoint begin = now();
        sTensor result = clone();
        for (uint i = 0; i < _storageSize; i++)
            result._storage[i] = _storage[i] < value ? 1.0f : 0.0f;
        return result.autolog("less_than", begin);
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

    void FromHost(CudaTensor& t) const
    {
        memcpy(&t._dimensions[0], &_dimensions[0], _rank * sizeof(uint));
        t._storage = _storage;
        t._storageSize = _storageSize;
    }

    void ToHost(const CudaTensor& t)
    {
        memcpy(_storage, t._storage, _storageSize * sizeof(float));
    }

    sTensor MatMult(const sTensor& other) const
    {
        timepoint begin = now();
        assert(_rank == 2);
        assert(other._rank == 2);

        const uint nrows = dim(0);
        const uint ncols = dim(1);
        const uint other_nrows = other.dim(0);
        const uint other_ncols = other.dim(1);
        assert(ncols == other_nrows);

        sTensor result = Zeros(nrows, other_ncols);
        result.set_label(_label);

        //CudaTensor cpuLeft; this->FromHost(cpuLeft);
        //CudaTensor cpuRight; other.FromHost(cpuRight);
        //CudaTensor cpuResult; result.FromHost(cpuResult);

        //cpuMmatmul(cpuLeft, cpuRight, cpuResult);
        //result.ToHost(cpuResult);

        for (uint i = 0; i < nrows; i++)
        {
            for (uint j = 0; j < other_ncols; j++)
            {
                for (uint k = 0; k < ncols; k++)
                {
                    result.add2d(i, j, get2d(i, k) * other.get2d(k, j));
                }
            }
        }

        return result.autolog("matmul", begin);
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
        result.set_label(_label);

        for (uint r = 0; r < dim(0); r++)
        {
            for (uint c = 0; c < dim(1); c++)
            {
                result.set2d(c, r, get2d(r, c));
            }
        }
        return result;
    }

    sTensor clone() const
    {
        sTensor result(_rank, _dimensions);
        memcpy(result._storage, _storage, _storageSize * sizeof(float));
        return result.set_label(_label);
    }

    sTensor clone_empty() const
    {
        uint dims[sTENSOR_MAX_DIMENSIONS];
        memcpy(dims, _dimensions, _rank * sizeof(uint));
        dims[_rank-1] = 0; // last dimension is 0

        sTensor result(_rank, dims);
        return result.set_label(_label);
    }

    sTensor clone_shallow() const
    {
        sTensor result(_rank, _dimensions);
        result._storage = _storage;
        result._storageOwned = false;
        result._label = _label;
        return result;
    }

    sTensor& ref_shallow_(const sTensor& other)
    {
        _rank = other._rank;
        memcpy(_dimensions, other._dimensions, _rank * sizeof(uint));
        _storage = other._storage;
        _storageSize = other._storageSize;
        _storageOwned = false;
        return *this;
    }

    sTensor row(const uint row)
    {
        assert(_rank == 2);
        assert(row < dim(0));

        sTensor result = Dims(uint(1), dim(1));
        result.set_label(_label);

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
        result.set_label(_label);

        for (uint r = 0; r < dim(0); r++)
        {
            result(r, uint(0)) = operator()(r, col);
        }
        return result;
    }

    sTensor random_sample_rows(const float p)
    {
        timepoint begin = now();
        assert(_rank == 2);
        assert(p >= 0.0f && p <= 1.0f);
        const uint nrows = dim(0);
        const uint count = uint(p * nrows);

        std::vector<uint> ids(nrows);
        for (uint i = 0; i < nrows; i++)
        {
            ids[i] = i;
        }
        std::shuffle(ids.begin(), ids.end(), std::mt19937{ std::random_device{}() });

        sTensor result = Dims(count, dim(1));
        result.set_label(_label);

        for (uint i = 0; i < count; i++)
        {
            const uint row = ids[i];
            for (uint c = 0; c < dim(1); c++)
            {
                result(i, c) = operator()(row, c);
            }
        }
        return result.autolog("random_sample", begin);
    }

    sTensor slice_rows(const uint start, const uint end) const
    {
        timepoint begin = now();
        assert(start < end);
        assert(end <= dim(0));

        uint n = 1;
        for (uint i = 1; i < _rank; i++)
        {
            n *= _dimensions[i];
        }

        sTensor result = Dims(end - start, dim(1));
        result.set_label(_label);

        const uint index = start * n;
        memcpy(result._storage, _storage + index, (end - start) * n * sizeof(float));
        return result.autolog("slice_rows", begin);
    }

    void put_rows(const uint start, const sTensor& other)
    {
        assert(_rank == other.rank());
        assert(start + other.dim(0) < dim(0));

        uint n = 1;
        for (uint i = 1; i < _rank; i++)
        {
            n *= _dimensions[i];
        }
        const uint index = start * n;
        memcpy(_storage + index, other._storage, other.size() * sizeof(float));
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

    sTensorInfo info(const char* operation, const timepoint begin) const;
};

std::ostream& operator<<(std::ostream& os, const sTensor& m);




