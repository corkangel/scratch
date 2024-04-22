#pragma once

#include <array>
#include <cstdarg>

#include <stb_image.h>

#include "utils.h"
#include "ptr.h"
#include "matmul.ch"
#include "slog.h"

constexpr uint sTENSOR_MAX_DIMENSIONS = max_tensor_dimensions;

class sTensor;
using pTensor = sPtr<sTensor>;

// alias to make natvis convenient
class sStorage : public std::shared_ptr<float>
{
public:
    sStorage() : std::shared_ptr<float>(nullptr) {}
    sStorage(float* ptr) : std::shared_ptr<float>(ptr) {}
    sStorage(const sStorage& other) : std::shared_ptr<float>(other) {}
    sStorage(const std::shared_ptr<float>& other) : std::shared_ptr<float>(other) {}
};

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
    pTensor _tensor;
    uint _row;

public:
    sTensorRowIterator(pTensor tensor, uint row = 0);

    bool operator!=(const sTensorRowIterator& other) const;

    pTensor operator*();

    // prefix increment
    sTensorRowIterator operator++();

    // postfix increment
    sTensorRowIterator operator++(int);

    uint row() const
    {
        return _row;
    }
};

class sTensor : public sPtrBase
{
public:

    // ---------------- constructors -----------------

    template <typename... Dimensions, typename std::enable_if<(std::is_same_v<Dimensions, uint> && ...), uint>::type = 0>
    sTensor(Dimensions... dimensions) :
        _rank(sizeof...(dimensions))
    {
        static_assert(sizeof...(dimensions) <= sTENSOR_MAX_DIMENSIONS, "Too many dimensions");
        const uint dims[] = { dimensions... };
        Allocate(dims);
        _id = idCounter++;
    }

    template <typename... Dimensions, typename std::enable_if<(std::is_same_v<Dimensions, int> && ...), int>::type = 0>
    sTensor(Dimensions... dimensions) : 
        _rank(sizeof...(dimensions))
    {
        static_assert(sizeof...(dimensions) <= sTENSOR_MAX_DIMENSIONS, "Too many dimensions");
        const int dims[] = { dimensions... };
        Allocate(reinterpret_cast<const uint*>(dims));
        _id = idCounter++;
    }

    sTensor(const uint rank, const uint* dimensions, const uint id = 0, const char* label = nullptr);
    sTensor(const sTensor& other);
    ~sTensor();

    // ---------------- static constructors -----------------
    
    static sTensor EmptyNoPtr();
    static pTensor Empty();
    static pTensor Rank(const uint rank);

    template<typename... Dimensions> static pTensor Dims(Dimensions... dims)
    {
        return pTensor( new sTensor(dims...));
    }

    template<typename... Dimensions> static pTensor Fill(const float value, Dimensions... dims)
    {
        pTensor result = pTensor(new sTensor(dims...));
        result->fill_(value);
        return result;
    }

    template<typename... Dimensions> static pTensor Ones(Dimensions... dims)
    {
        pTensor result = pTensor(new sTensor(dims...));
        result->ones_();
        return result;
    }

    template<typename... Dimensions> static pTensor Zeros(Dimensions... dims)
    {
        pTensor result = pTensor(new sTensor(dims...));
        result->zero_();
        return result;
    }

    template<typename... Dimensions> static pTensor Randoms(Dimensions... dims)
    {
        pTensor result = pTensor(new sTensor(dims...));
        result->random_();
        return result;
    }

    template<typename... Dimensions> static pTensor NormalDistribution(const float mean, const float stddev, Dimensions... dims)
    {
        pTensor result = pTensor(new sTensor(dims...));
        result->normal_distribution_(mean, stddev);
        return result;
    }

    template<typename... Dimensions> static pTensor Integers(const int start, Dimensions... dims)
    {
        pTensor result = pTensor(new sTensor(dims...));
        result->integers_(start);
        return result;
    }

    template<typename... Dimensions> static pTensor Linear(const float start, const float step, Dimensions... dims)
    {
        pTensor result = pTensor(new sTensor(dims...));
        result->linear_(start, step);
        return result;
    }

    // ---------------- accessors -----------------
    uint rank() const;
    uint dim(uint i) const;
    uint dim_unsafe(uint i) const;
    uint size() const;
    uint size_dims(const uint dims) const; // the size up to the specified dimension
    uint bytes() const;
    float* data();
    const float* data_const() const;
    float at(const uint n) const;
    pTensor ptr();

    // ---------------- setters -----------------
    pTensor set_label(const char* label);
    void set_grad(const pTensor& grad);
    pTensor grad() const;
    void zero_grad();

    // ---------------- in place operations -----------------
    pTensor fill_(float value);
    pTensor zero_();
    pTensor ones_();
    pTensor gaussian_(float bandwidth);
    pTensor pow_(const float power);
    pTensor sqrt_();
    pTensor exp_();
    pTensor log_();
    pTensor abs_();
    pTensor add_(const float value);
    pTensor subtract_(const float value);
    pTensor multiply_(const float value);
    pTensor divide_(const float value);
    pTensor random_();
    pTensor normal_distribution_(const float mean, const float stddev);
    pTensor integers_(int start);
    pTensor linear_(float start, const float step);
    pTensor cat0_(const pTensor& other);
    pTensor set_row_(uint row, const pTensor& other);
    pTensor transpose_();
    pTensor clamp_min_(const float v);
    pTensor clamp_max_(const float v);

    pTensor index_select(const pTensor& indices) const;
    pTensor argmax() const;

    // fills with 0 or 1 based on the comparison
    pTensor equal(const pTensor& other) const;

    // remove a specified dimension
    pTensor squeeze(const uint dim) const;
    pTensor squeeze_(const uint dim);

    // removes all dimensions of size 1
    pTensor squeeze() const;
    pTensor squeeze_();

    // adds a dimension of size 1 at the specified position
    pTensor unsqueeze(uint dim) const;
    pTensor unsqueeze_(uint dim);

    pTensor flatten_();
    pTensor reshapeto_(const pTensor& other);

    template<typename... Dimensions, typename std::enable_if<(std::is_same_v<Dimensions, uint> && ...), uint>::type=0>
    pTensor reshape_(Dimensions... dims)
    {
        timepoint begin = now();
        const uint ds[] = { dims... };
        const uint numDims = sizeof...(Dimensions);
        return reshape_internal(numDims, ds);
    }

    template<typename... Dimensions, typename std::enable_if<(std::is_same_v<Dimensions, int> && ...), int>::type=0>
    pTensor reshape_(Dimensions... dims)
    {
        timepoint begin = now();
        const int ds[] = { dims... };
        const int numDims = sizeof...(Dimensions);
        return reshape_internal(numDims, (const uint*)ds);
    }

    template<typename... Dimensions, typename std::enable_if<(std::is_same_v<Dimensions, uint> && ...), uint>::type=0>
    pTensor view_(Dimensions... dims)
    {
        timepoint begin = now();
        const uint ds[] = { dims... };
        const uint numDims = sizeof...(Dimensions);
        return view_internal(numDims, ds);
    }

    template<typename... Dimensions, typename std::enable_if<(std::is_same_v<Dimensions, int> && ...), int>::type=0>
    pTensor view_(Dimensions... dims)
    {
        timepoint begin = now();
        const int ds[] = { dims... };
        const int numDims = sizeof...(Dimensions);
        return view_internal(numDims, (const uint*)ds);
    }

    // ---------------- scalar operations -----------------
    pTensor exp() const;
    float sum() const;
    float mean() const;
    float std() const;
    float mse() const;
    float rmse() const;
    float min() const;
    float max() const;

    // ---------------- operators -----------------
    bool operator==(const pTensor& other);
    bool operator!=(const pTensor& other);
    pTensor operator=(const pTensor& other);

    // ---------------- tensor math operators -----------------
    pTensor operator+(const pTensor& other) const;
    pTensor operator-(const pTensor& other) const;
    pTensor operator/(const pTensor& other) const;
    pTensor operator*(const pTensor& other) const;

    pTensor operator+=(const pTensor& other);
    pTensor operator-=(const pTensor& other);
    pTensor operator/=(const pTensor& other);
    pTensor operator*=(const pTensor& other);

    // sum of all elements in each row - only works for 2x2 matrices
    pTensor sum_rows();

    // sum of all elements in each column - only works for 2x2 matrices
    pTensor sum_columns();

    float getAt(const uint n) const;
    float get1d(const uint n) const;
    void set1d(const uint n, const float value);
    void add1d(const uint n, const float value);
    float get2d(const uint row, const uint col) const;
    void set2d(const uint row, const uint col, const float value);
    void add2d(const uint row, const uint col, const float value);
    float get3d(const uint d1, const uint d2, const uint d3) const;
    void set3d(const uint d1, const uint d2, const uint d3, const float value);
    void add3d(const uint d1, const uint d2, const uint d3, const float value);
    float get4d(const uint d1, const uint d2, const uint d3, const uint d4) const;
    void set4d(const uint d1, const uint d2, const uint d3, const uint d4, const float value);
    void add4d(const uint d1, const uint d2, const uint d3, const uint d4, const float value);


    pTensor sum(const uint dim);
    pTensor sum_final_dimension();
    pTensor greater_than(const float value);
    pTensor less_than(const float value);

    // pads both ends, so pad=1 adds 1 row and 1 column to each side
    pTensor pad2d(const uint pad) const;

    // format is (batch, channels, rows, columns)
    pTensor pad_images(const uint padding, const bool dilation = false) const;

    // ---------------- tensor scalar operators -----------------

    pTensor operator+(const float value) const;
    pTensor operator-(const float value) const;
    pTensor operator*(const float value) const;
    pTensor operator/(const float value) const;
    pTensor operator+=(const float value);
    pTensor operator-=(const float value);
    pTensor operator*=(const float value);
    pTensor operator/=(const float value);

    // ---------------- utils -----------------

    void FromHost(CudaTensor& t) const;
    void ToHost(const CudaTensor& t);

    pTensor MatMult(const pTensor& other) const;
    float DotProduct(const pTensor& other);
    pTensor Transpose() const;

    pTensor clone() const;
    pTensor clone_shallow() const;

    // returns a new tensor with the same dimensions but a single 'row'
    pTensor select1d(const uint index0) const;
    pTensor select2d(const uint index0, const uint index1) const;


    pTensor row2d(const uint row) const;
    pTensor column2d(const uint col) const;

    pTensor random_sample_rows(const float p);
    pTensor slice_rows(const uint start, const uint end) const;
    void put_rows(const uint start, const pTensor& other);
    
    pTensor slice2d(const uint rowStart, const uint rowEnd, const uint colStart, const uint colEnd) const;
    pTensor slice3d(const uint d1Start, const uint d1End, const uint d2Start, const uint d2End, const uint d3Start, const uint d3End) const;
    pTensor slice4d(const uint d1Start, const uint d1End, const uint d2Start, const uint d2End, const uint d3Start, const uint d3End, const uint d4Start, const uint d4End);

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
        return sTensorRowIterator(ptr());
    }

    sTensorRowIterator end_rows()
    {
        return sTensorRowIterator(ptr(), _dimensions[0]);
    }

    sTensorInfo info(const char* operation, const timepoint begin) const;

    static bool enableAutoLog;
    static const sTensor null;
    static const pTensor nullPtr;

    // for sPtrBase
    void release() override;

private:

    using sTensorOp = void(*)(float&, const float, const float);

    sTensor& apply_rank1(pTensor& result_, const sTensor& other, sTensorOp f) const;
    sTensor& apply_rank2(pTensor& result_, const sTensor& other, sTensorOp f) const;
    sTensor& apply_rank3(pTensor& result_, const sTensor& other, sTensorOp f) const;
    sTensor& apply_rank4(pTensor& _result, const sTensor& other, sTensorOp f) const;
    pTensor apply_(const pTensor& other, sTensorOp f) const;

    pTensor sum_rank2(pTensor& result, const uint dim);
    pTensor sum_rank3(pTensor& result, const uint dim);
    pTensor sum_rank4(pTensor& result, const uint dim);
    pTensor apply_inplace_(const pTensor& other, sTensorOp f);

    pTensor reshape_internal(const uint rank, const uint* dims);
    pTensor view_internal(const uint rank, const uint* dims);

    static uint idCounter;

    uint _id;
    uint _dimensions[sTENSOR_MAX_DIMENSIONS] = {};
    uint _rank;

    sStorage _storage;
    uint _storageSize;
    const char* _label = nullptr;

    pTensor _grad;

    void Allocate(const uint* dims);

    const timepoint now() const;

    pTensor autolog(const char* label, const timepoint begin);
};

std::ostream& operator<<(std::ostream& os, const sTensor& m);

