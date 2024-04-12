#include "stensor.h"
#include <chrono>

// ----------------- cell iterator -----------------

sTensorCellIterator::sTensorCellIterator(sTensor& tensor, bool end) : _tensor(tensor), _pos(0)
{
    if (end) _pos = tensor.size();
}

bool sTensorCellIterator::operator!=(const sTensorCellIterator& other) const
{
    return _pos != other._pos;
}

float& sTensorCellIterator::operator*()
{
    return _tensor.data()[_pos];
}

// prefix increment
sTensorCellIterator sTensorCellIterator::operator++()
{
    _pos++;
    if (_pos >= _tensor.size())
    {
        _pos = _tensor.size();
    }
    return *this;
}

// postfix increment
sTensorCellIterator sTensorCellIterator::operator++(int)
{
    sTensorCellIterator copy(*this);
    _pos++;
    if (_pos >= _tensor.size())
    {
        _pos = _tensor.size();
    }
    return copy;
}

// ----------------- row iterator -----------------

sTensorRowIterator::sTensorRowIterator(pTensor tensor, uint row) : _tensor(tensor), _row(row)
{
}

sTensorRowIterator sTensorRowIterator::operator++()
{
    _row++;
    if (_row >= _tensor->dim(0))
    {
        _row = _tensor->dim(0);
    }
    return *this;
}

sTensorRowIterator sTensorRowIterator::operator++(int)
{
    sTensorRowIterator copy(*this);
    _row++;
    if (_row >= _tensor->dim(0))
    {
        _row = _tensor->dim(0);
    }
    return copy;
}

bool sTensorRowIterator::operator!=(const sTensorRowIterator& other) const
{
    return _row != other._row;
}

pTensor sTensorRowIterator::operator*()
{
    sTensor* row = new sTensor(_tensor->rank()-1, &_tensor->_dimensions[1]);
    row->Allocate(&_tensor->_dimensions[1]);
    uint rowSize = 1;
    for (uint i = 1; i < _tensor->rank(); ++i)
    {
        rowSize *= _tensor->dim(i);
    }
    memcpy(row->data(), _tensor->data() + _row * rowSize, rowSize * sizeof(float));
    return pTensor(row);
}


// ----------------- utils -----------------

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
        if (i >= 6 && i < m.size() - 6)
        {
            if (i == 6)
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

bool sTensor::enableAutoLog = false;
uint sTensor::idCounter = 100;
const sTensor sTensor::null = sTensor::EmptyNoPtr();
const pTensor sTensor::nullPtr = sTensor::Empty();


sTensorInfo sTensor::info(const char* operation, const std::chrono::steady_clock::time_point begin) const
{
    sTensorInfo result;
    result.rank = _rank;
    memcpy(result.dimensions, _dimensions, _rank * sizeof(uint));
    result.label = _label;
    result.operation = operation;
    result.id = _id;

    auto end = std::chrono::high_resolution_clock::now();
    result.time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

    memcpy(result.data_front, _storage.get(), std::min(_storageSize, sInfoDataSize) * sizeof(float));
    memcpy(result.data_back, _storage.get() + _storageSize - sInfoDataSize, std::min(_storageSize, sInfoDataSize) * sizeof(float));
    return result;
}



void sTensor::Allocate(const uint* dims)
{
    assert(_rank <= sTENSOR_MAX_DIMENSIONS);
    memcpy(_dimensions, dims, _rank * sizeof(uint));

    _storageSize = 1;
    for (uint i = 0; i < _rank; i++)
        _storageSize *= _dimensions[i];

    _storage.reset(new float[_storageSize]);

#if _DEBUG
    const float STUFF = 9999.7777f;
    float* s = _storage.get();
    for (uint i = 0; i < _storageSize; i++) s[i] = STUFF;
#endif
}

const timepoint sTensor::now() const
{
    if (!sTensor::enableAutoLog) return timepoint();

    return std::chrono::steady_clock::now();
}

pTensor sTensor::autolog(const char* label, const timepoint begin)
{
    if (!sTensor::enableAutoLog) return ptr();

    std::stringstream ss;
    ss << _id << ":";
    ss << std::fixed << std::setprecision(2);
    ss << (_label ? _label : "") << " / " << label << ": [";
    for (uint i = 0; i < _rank; i++)
    {
        ss << _dimensions[i];
        if (i < _rank - 1) ss << ", ";
    }
    ss << "] {";

    if (_rank == 0)
        ss << _storage.get()[0];
    else
    {
        for (uint i = 0; i < 4; i++)
        {
            ss << _storage.get()[i];
            if (i < _storageSize - 1) ss << ", ";
        }

        if (_storageSize > 4) ss << "...";

        for (uint i = 0; i < std::min(uint(4), _storageSize); i++)
        {
            uint pos = _storageSize - i - 1;
            ss << _storage.get()[pos];
            if (pos < _storageSize - 1) ss << ", ";
        }
        ss << "}";
    }
    slog(ss.str());

    log_tensor_info(info(label, begin));
    return ptr();
}

pTensor sTensor::ptr()
{
    return pTensor(this);
}

void sTensor::release()
{
    delete this;
}

sTensor::sTensor(const uint rank, const uint* dimensions, const uint id, const char* label) :
    _rank(rank)
{
    Allocate(dimensions);
    if (id != 0) _id = id; else _id = idCounter++;
    if (label) _label = label;
}


sTensor::sTensor(const sTensor& other)
    : _rank(other._rank), _storage(other._storage), _storageSize(other._storageSize), _label(other._label), _id(other._id)
{
    memcpy(_dimensions, other._dimensions, _rank * sizeof(uint));
}

sTensor::~sTensor()
{
    if (_storage.get() != nullptr)
    {
        _storage.reset();
    }
}

pTensor sTensor::set_label(const char* label)
{
    _label = label;
    return ptr();
}

void sTensor::set_grad(const pTensor& grad)
{
    _grad = grad;
}

pTensor sTensor::grad() const
{
    return _grad;
}

void sTensor::zero_grad()
{
    if (_grad())
        _grad->zero_();
}

// ---------------- constructors -----------------

sTensor sTensor::EmptyNoPtr()
{
    constexpr uint dims[sTENSOR_MAX_DIMENSIONS] = { 0 };
    sTensor result(0, dims);
    return result;
}

pTensor sTensor::Empty()
{
    constexpr uint dims[sTENSOR_MAX_DIMENSIONS] = { 0 };
    sTensor* result = new sTensor(0, dims);
    return pTensor(result);
}

pTensor sTensor::Rank(const uint rank)
{
    constexpr uint dims[sTENSOR_MAX_DIMENSIONS] = { 0 };
    sTensor* result = new sTensor(rank, dims);
    return pTensor(result);
}

// ---------------- accessors -----------------

uint sTensor::rank() const
{
    return _rank;
}

uint sTensor::dim(uint i) const
{
    assert(i < _rank);
    return _dimensions[i];
}

uint sTensor::dim_unsafe(uint i) const
{
    return _dimensions[i];
}

uint sTensor::size() const
{
    return _storageSize;
}

uint sTensor::size_dims(const uint dims) const
{
    uint n = 1;
    for (uint i = 0; i < dims; i++)
    {
        n *= _dimensions[i];
    }
    return n;
}

uint sTensor::bytes() const
{
    return _storageSize * sizeof(float);
}

float* sTensor::data()
{
    return _storage.get();
}

const float* sTensor::data_const() const
{
    return _storage.get();
}

float sTensor::at(uint i) const
{
    assert(i < _storageSize);
    return _storage.get()[i];
}

// ---------------- in place operations -----------------

pTensor sTensor::fill_(float value)
{
    float* s = _storage.get();
    for (uint i = 0; i < _storageSize; i++)
        s[i] = value;
    return ptr();
}

pTensor sTensor::zero_()
{
    timepoint begin = now();
    fill_(0.0f);
    return autolog("zero_", begin);
}

pTensor sTensor::ones_()
{
    timepoint begin = now();
    fill_(1.0f);
    return autolog("ones_", begin);
}

pTensor sTensor::gaussian_(float bandwidth)
{
    timepoint begin = now();
    float* s = _storage.get();
    for (uint i = 0; i < _storageSize; i++)
        s[i] = gaussian(s[i], bandwidth);
    return autolog("gaussian_", begin);
}

pTensor sTensor::pow_(uint power)
{
    timepoint begin = now();
    float* s = _storage.get();
    for (uint i = 0; i < _storageSize; i++)
        s[i] = float(pow(s[i], power));
    return autolog("pow_", begin);
}

pTensor sTensor::sqrt_()
{
    timepoint begin = now();
    float* s = _storage.get();
    for (uint i = 0; i < _storageSize; i++)
        s[i] = sqrt(s[i]);
    return autolog("sqrt_", begin);
}

pTensor sTensor::exp_()
{
    timepoint begin = now();
    float* s = _storage.get();
    for (uint i = 0; i < _storageSize; i++)
        s[i] = std::exp(s[i]);
    return autolog("exp_", begin);
}

pTensor sTensor::log_()
{
    timepoint begin = now();
    float* s = _storage.get();
    for (uint i = 0; i < _storageSize; i++)
        s[i] = log(s[i]);
    return autolog("log_", begin);
}

pTensor sTensor::abs_()
{
    timepoint begin = now();
    float* s = _storage.get();
    for (uint i = 0; i < _storageSize; i++)
        s[i] = abs(s[i]);
    return autolog("abs_", begin);
}

pTensor sTensor::add_(const float value)
{
    timepoint begin = now();
    float* s = _storage.get();
    for (uint i = 0; i < _storageSize; i++)
        s[i] += value;
    return autolog("add_", begin);
}

pTensor sTensor::subtract_(const float value)
{
    timepoint begin = now();
    float* s = _storage.get();
    for (uint i = 0; i < _storageSize; i++)
        s[i] += value;
    return autolog("subtract_", begin);
}

pTensor sTensor::multiply_(const float value)
{
    timepoint begin = now();
    float* s = _storage.get();
    for (uint i = 0; i < _storageSize; i++)
        s[i] *= value;
    return autolog("multiply_", begin);
}

pTensor sTensor::divide_(const float value)
{
    timepoint begin = now();
    float* s = _storage.get();
    for (uint i = 0; i < _storageSize; i++)
        s[i] /= value;
    return autolog("divide_", begin);
}

pTensor sTensor::random_()
{
    timepoint begin = now();
    float* s = _storage.get();
    for (uint i = 0; i < _storageSize; i++)
        s[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    return autolog("random_", begin);
}

pTensor sTensor::normal_distribution_(const float mean, const float stddev)
{
    timepoint begin = now();
    std::default_random_engine generator;
    std::normal_distribution distribution(mean, stddev);
    float* s = _storage.get();
    for (uint i = 0; i < _storageSize; i++)
        s[i] = distribution(generator);
    return autolog("normal_distribution_", begin);
}

pTensor sTensor::integers_(int start)
{
    timepoint begin = now();
    float* s = _storage.get();
    for (uint i = 0; i < _storageSize; i++)
        s[i] = float(start++);
    return autolog("integers_", begin);
}

pTensor sTensor::linear_(float start, const float step)
{
    timepoint begin = now();
    float* s = _storage.get();
    for (uint i = 0; i < _storageSize; i++)
    {
        s[i] = start;
        start += step;
    }
    return autolog("linear_", begin);
}


pTensor sTensor::index_select(const pTensor& indices) const
{
    timepoint begin = now();
    assert(_rank == 2 && indices->rank() == 1);

    const uint n = indices->size();
    pTensor result = sTensor::Dims(n, uint(1));
    for (uint i = 0; i < n; i++)
    {
        result->set2d(i, 0, get2d(i, uint(indices->get1d(i))));
    }
    return result->autolog("index_select", begin);
}

pTensor sTensor::argmax() const
{
    timepoint begin = now();

    const uint nrows = dim(0);
    const uint ncols = dim(1);
    pTensor result = sTensor::Dims(nrows, uint(1));

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
        result->set2d(i, 0, float(maxIndex));
    }
    return result->autolog("argmax", begin);
}

// fills with 0 or 1 based on the comparison
pTensor sTensor::equal(const pTensor& other) const
{
    assert(_rank == other->_rank);
    assert(_storageSize == other->_storageSize);

    timepoint begin = now();
    pTensor result = clone();
    const float* s = _storage.get();
    for (uint i = 0; i < _storageSize; i++)
        result->_storage.get()[i] = s[i] == other->_storage.get()[i] ? 1.0f : 0.0f;
    return result->autolog("equal", begin);
}

// remove a specified dimension
pTensor sTensor::squeeze(const uint dim) const
{
    pTensor result = clone_shallow();
    return result->squeeze_(dim);
}

pTensor sTensor::squeeze_(const uint dim)
{
    assert(dim < _rank);
    assert(_dimensions[dim] == 1);
    timepoint begin = now();
    uint newRank = 0;
    for (uint i = 0; i < _rank; i++)
    {
        if (i != dim)
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

// removes all dimensions of size 1

pTensor sTensor::squeeze() const
{
    pTensor result = clone_shallow();
    return result->squeeze_();
}

pTensor sTensor::squeeze_()
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

pTensor sTensor::unsqueeze(uint dim) const
{
    pTensor result = clone_shallow();
    return result->unsqueeze_(dim);
}

pTensor sTensor::unsqueeze_(uint dim)
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

pTensor sTensor::cat0_(const pTensor& other)
{
    timepoint begin = now();

    assert(_rank == other->_rank);
    assert(dim(1) == other->dim(1));
    _dimensions[0] += other->dim(0);

    float* newStorage = new float[_storageSize + other->_storageSize];
    memcpy(newStorage, _storage.get(), _storageSize * sizeof(float));
    memcpy(newStorage + _storageSize, other->_storage.get(), other->_storageSize * sizeof(float));

    _storage.reset(newStorage);
    _storageSize += other->_storageSize;
    return autolog("cat0_", begin);
}

pTensor sTensor::set_row_(uint row, const pTensor& other)
{
    timepoint begin = now();
    assert(_rank - 1 == other->rank());

    uint n = 1;
    for (uint i = 1; i < _rank; i++)
    {
        n *= _dimensions[i];
    }
    uint start = row * n;

    float* s = _storage.get();
    const float* o = other->_storage.get();
    for (uint i = 0; i < other->size(); i++)
    {
        s[start + i] = o[i];
    }
    return autolog("set_row_", begin);
}

pTensor sTensor::transpose_()
{
    timepoint begin = now();
    assert(_rank == 2);
    const float *s = _storage.get();
    float* newStorage = new float[_storageSize];
    for (uint r = 0; r < dim(0); r++)
    {
        for (uint c = 0; c < dim(1); c++)
        {
            newStorage[c * dim(0) + r] = s[r * dim(1) + c];
            //newStorage[c * dim(0) + r] = operator()(r, c);
        }
    }
    _storage.reset(newStorage);

    uint temp = _dimensions[0];
    _dimensions[0] = _dimensions[1];
    _dimensions[1] = temp;
    return autolog("transpose_", begin);
}

pTensor sTensor::clamp_min_(const float v)
{
    timepoint begin = now();
    float* s = _storage.get();
    for (uint i = 0; i < _storageSize; i++)
        s[i] = std::max(s[i], v);
    return autolog("clamp_min", begin);
}

pTensor sTensor::clamp_max_(const float v)
{
    timepoint begin = now();
    float* s = _storage.get();
    for (uint i = 0; i < _storageSize; i++)
        s[i] = std::min(s[i], v);
    return autolog("clamp_max", begin);
}

// ---------------- scalar operations -----------------

pTensor sTensor::exp() const
{
    pTensor result = clone();
    return result->exp_();
}

float sTensor::sum() const
{
    float result = 0.0f;
    const float* s = _storage.get();
    for (uint i = 0; i < _storageSize; i++)
        result += s[i];
    return result;
}

float sTensor::mean() const
{
    return sum() / _storageSize;
}

float sTensor::std() const
{
    float m = mean();
    float result = 0.0f;
    const float* s = _storage.get();
    for (uint i = 0; i < _storageSize; i++)
        result += float(pow(s[i] - m, 2));
    return sqrt(result / _storageSize);
}

float sTensor::mse() const
{
    float result = 0.0f;
    const float* s = _storage.get();
    for (uint i = 0; i < _storageSize; i++)
        result += float(pow(s[i], 2));
    return result / _storageSize;
}

float sTensor::rmse() const
{
    return sqrt(mse());
}

float sTensor::min() const
{
    const float* s = _storage.get();
    float result = s[0];
    for (uint i = 1; i < _storageSize; i++)
        result = std::min(result, s[i]);
    return result;
}

float sTensor::max() const
{
    const float* s = _storage.get();
    float result = s[0];
    for (uint i = 1; i < _storageSize; i++)
        result = std::max(result, s[i]);
    return result;
}

// ---------------- operators -----------------

bool sTensor::operator==(const pTensor& other)
{
    bool result = (_rank == other->_rank);

    for (uint i = 0; i < _rank; i++)
        result &= (_dimensions[i] == other->_dimensions[i]);

    const float* s = _storage.get();
    const float* o = other->_storage.get();
    for (uint i = 0; i < _storageSize; i++)
        result &= (s[i] == o[i]);

    return result;
}

bool sTensor::operator!=(const pTensor& other)
{
    return !operator==(other);
}

pTensor sTensor::operator=(const pTensor& other)
{
    timepoint begin = now();
    for (uint i = 0; i < _rank; i++)
        assert(_dimensions[i] == other->_dimensions[i]);

    _rank = other->_rank;
    _storage = other->_storage;
    _storageSize = other->_storageSize;

    _label = other->_label;
    _id = other->_id;
    _grad = other->_grad;

    return autolog("operator=", begin);
}


// ---------------- tensor math operators -----------------

sTensor& sTensor::apply_rank1(pTensor& result_, const sTensor& other, sTensorOp f) const
{
    sTensor& result = *result_;
    float *r = result._storage.get();

    float *s = _storage.get();
    float *o = other._storage.get();
    for (uint i = 0; i < _dimensions[0]; i++)
    {
        const uint left_i = (i >= _dimensions[0]) ? 0 : i;
        const uint right_i = (i >= other._dimensions[0]) ? 0 : i;
        f(r[i], s[left_i], o[right_i]);
    }

     //f(r[i], operator()(i), other(i));

    return result;
}

sTensor& sTensor::apply_rank2(pTensor& result_, const sTensor& other, sTensorOp f) const
{
    sTensor& result = *result_;
    float* r = result._storage.get();

    const uint maxDim0 = std::max(_dimensions[0], other._dimensions[0]);
    const uint maxDim1 = std::max(_dimensions[1], other._dimensions[1]);

    const float* s = _storage.get();
    float *o = other._storage.get();

    for (uint i = 0; i < maxDim0; i++)
    {
        for (uint j = 0; j < maxDim1; j++)
        {
            const uint left_i = (i >= _dimensions[0]) ? 0 : i;
            const uint left_j = (j >= _dimensions[1]) ? 0 : j;
            const uint right_i = (i >= other._dimensions[0]) ? 0 : i;
            const uint right_j = (j >= other._dimensions[1]) ? 0 : j;

            f(r[i * maxDim1 + j], s[left_i * _dimensions[1] + left_j], o[right_i * other._dimensions[1] + right_j]);
            
            //f(r[i * maxDim1 + j], operator()(left_i, left_j), other(right_i, right_j));
            //f(r[], operator()(left_i, left_j), other(right_i, right_j));
        }
    }
    return result;
}

sTensor& sTensor::apply_rank3(pTensor& result_, const sTensor& other, sTensorOp f) const
{
    sTensor& result = *result_;
    float* r = result._storage.get();

    const uint maxDim0 = std::max(_dimensions[0], other._dimensions[0]);
    const uint maxDim1 = std::max(_dimensions[1], other._dimensions[1]);
    const uint maxDim2 = std::max(_dimensions[2], other._dimensions[2]);

    const float* s = _storage.get();
    const float* o = other._storage.get();

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

                f(r[i * maxDim1 * maxDim2 + j * maxDim2 + k], s[left_i * _dimensions[1] * _dimensions[2] + left_j * _dimensions[2] + left_k], o[right_i * other._dimensions[1] * other._dimensions[2] + right_j * other._dimensions[2] + right_k]);

                //f(r[i * maxDim1 * maxDim2 + j * maxDim2 + k], operator()(left_i, left_j, left_k), other(right_i, right_j, right_k));
                //f(result(i, j, k), operator()(left_i, left_j, left_k), other(right_i, right_j, right_k));
            }
        }
    }
    return result;
}

sTensor& sTensor::apply_rank4(pTensor& _result, const sTensor& other, sTensorOp f) const
{
    sTensor& result = *_result;
    float* r = result._storage.get();

    const uint maxDim0 = std::max(_dimensions[0], other._dimensions[0]);
    const uint maxDim1 = std::max(_dimensions[1], other._dimensions[1]);
    const uint maxDim2 = std::max(_dimensions[2], other._dimensions[2]);
    const uint maxDim3 = std::max(_dimensions[3], other._dimensions[3]);

    const float* s = _storage.get();
    const float* o = other._storage.get();

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

                    f(r[i * maxDim1 * maxDim2 * maxDim3 + j * maxDim2 * maxDim3 + k * maxDim3 + l], 
                        s[left_i * _dimensions[1] * _dimensions[2] * _dimensions[3] + left_j * _dimensions[2] * _dimensions[3] + left_k * _dimensions[3] + left_l], 
                        o[right_i * other._dimensions[1] * other._dimensions[2] * other._dimensions[3] + right_j * other._dimensions[2] * other._dimensions[3] + right_k * other._dimensions[3] + right_l]);

                    //f(result(i, j, k, l), operator()(left_i, left_j, left_k, left_l), other(right_i, right_j, right_k, right_l));
                }
            }
        }
    }
    return result;
}


pTensor sTensor::apply_(const pTensor& other, sTensorOp f) const
{
    assert(_rank == other->_rank);

    // broadcasting rules
    uint new_dims[sTENSOR_MAX_DIMENSIONS] = {};
    for (uint d = 0; d < _rank; d++)
    {
        if (_dimensions[d] == 1 && other->_dimensions[d] != 1)
        {
            new_dims[d] = other->_dimensions[d];
        }
        else if (other->_dimensions[d] == 1 && _dimensions[d] != 1)
        {
            new_dims[d] = _dimensions[d];
        }
        else
        {
            new_dims[d] = _dimensions[d];
            assert(_dimensions[d] == other->_dimensions[d]);
        }
    }

    pTensor result = pTensor(new sTensor(_rank, new_dims, _id, _label));
    result->set_label(other->_label);
    result->_grad = _grad;

    if (_rank == 1) return apply_rank1(result, *other, f).ptr();
    if (_rank == 2) return apply_rank2(result, *other, f).ptr();
    if (_rank == 3) return apply_rank3(result, *other, f).ptr();
    if (_rank == 4) return apply_rank4(result, *other, f).ptr();
    assert(false);

    return result;
}

pTensor sTensor::operator+(const pTensor& other) const
{
    timepoint begin = now();
    return apply_(other, [](float& o, const float a, const float b) { o = a + b; })->autolog("operator+", begin);
}

pTensor sTensor::operator-(const pTensor& other) const
{
    timepoint begin = now();
    return apply_(other, [](float& o, const float a, const float b) { o = a - b; })->autolog("operator-", begin);
}

pTensor sTensor::operator/(const pTensor& other) const
{
    timepoint begin = now();
    return apply_(other, [](float& o, const float a, const float b) { if (b == 0.f || isnan(b) || isinf(b)) o = 0.f; else o = a / b; })->autolog("operator/", begin);
}

pTensor sTensor::operator*(const pTensor& other) const
{
    timepoint begin = now();
    return apply_(other, [](float& o, const float a, const float b) { o = a * b; })->autolog("operator*", begin);
}

pTensor sTensor::apply_inplace_(const pTensor& other, sTensorOp f)
{
    assert(_rank == other->_rank);

    // broadcasting rules
    uint new_dims[sTENSOR_MAX_DIMENSIONS];
    for (uint d = 0; d < _rank; d++)
    {
        if (_dimensions[d] == 1 && other->_dimensions[d] != 1)
        {
            new_dims[d] = other->_dimensions[d];
        }
        else if (other->_dimensions[d] == 1 && _dimensions[d] != 1)
        {
            new_dims[d] = _dimensions[d];
        }
        else
        {
            new_dims[d] = _dimensions[d];
            assert(_dimensions[d] == other->_dimensions[d]);
        }
    }

    if (_rank == 1) return apply_rank1(ptr(), *other, f).ptr();
    if (_rank == 2) return apply_rank2(ptr(), *other, f).ptr();
    if (_rank == 3) return apply_rank3(ptr(), *other, f).ptr();
    if (_rank == 4) return apply_rank4(ptr(), *other, f).ptr();
    assert(false);

    return ptr();
}

pTensor sTensor::operator+=(const pTensor& other)
{
    timepoint begin = now();
    return apply_inplace_(other, [](float& o, const float a, const float b) { o = a + b; })->autolog("operator+", begin);
}

pTensor sTensor::operator-=(const pTensor& other)
{
    timepoint begin = now();
    return apply_inplace_(other, [](float& o, const float a, const float b) { o = a - b; })->autolog("operator-", begin);
}

pTensor sTensor::operator/=(const pTensor& other)
{
    timepoint begin = now();
    return apply_inplace_(other, [](float& o, const float a, const float b) { if (b == 0.f || isnan(b) || isinf(b)) o = 0.f; else o = a / b; })->autolog("operator/", begin);
}

pTensor sTensor::operator*=(const pTensor& other)
{
    timepoint begin = now();
    return apply_inplace_(other, [](float& o, const float a, const float b) { o = a * b; })->autolog("operator*", begin);
}

// sum of all elements in each row - only works for 2x2 matrices
pTensor sTensor::sum_rows()
{
    timepoint begin = now();
    assert(_rank == 2);
    const uint ncols = dim(1);

    pTensor result = Dims(uint(1), ncols);
    result->set_label(_label);

    float *r = result->_storage.get();
    for (uint c = 0; c < ncols; c++)
    {
        float sum = 0;
        for (uint r = 0; r < dim(0); r++)
        {
            sum += get2d(r, c);
        }
        r[c] = sum;
        //(*result)(uint(0), c) = sum;
    }
    return result->squeeze_()->autolog("sum_rows", begin);
}

// sum of all elements in each column - only works for 2x2 matrices
pTensor sTensor::sum_columns()
{
    timepoint begin = now();
    assert(_rank == 2);
    const uint nrows = dim(0);
    pTensor result = Dims(nrows, uint(1));
    result->set_label(_label);

    float *s = _storage.get();
    float *rr = result->_storage.get();
    for (uint r = 0; r < nrows; r++)
    {
        float sum = 0;
        for (uint c = 0; c < dim(1); c++)
        {
            sum += s[r * dim(1) + c];
        }
        rr[r] = sum;
        //(*result)(r, uint(0)) = sum;
    }
    return result->squeeze_()->autolog("sum_columns", begin);
}

float sTensor::getAt(const uint n) const
{
    assert(n < _storageSize);
    return _storage.get()[n];
}

float sTensor::get1d(const uint n) const
{
    assert(_rank == 1);
    return _storage.get()[n];
}

void sTensor::set1d(const uint n, const float value)
{
    assert(_rank == 1);
    _storage.get()[n] = value;
}

void sTensor::add1d(const uint n, const float value)
{
    assert(_rank == 1);
    _storage.get()[n] += value;
}

float sTensor::get2d(const uint row, const uint col) const
{
    assert(_rank == 2);
    return _storage.get()[row * _dimensions[1] + col];
}

void sTensor::set2d(const uint row, const uint col, const float value)
{
    assert(_rank == 2);
    _storage.get()[row * _dimensions[1] + col] = value;
}

void sTensor::add2d(const uint row, const uint col, const float value)
{
    assert(_rank == 2);
    _storage.get()[row * _dimensions[1] + col] += value;
}

float sTensor::get3d(const uint i, const uint j, const uint k) const
{
    assert(_rank == 3);
    return _storage.get()[i * _dimensions[1] * _dimensions[2] + j * _dimensions[2] + k];
}

void sTensor::set3d(const uint i, const uint j, const uint k, const float value)
{
    assert(_rank == 3);
    _storage.get()[i * _dimensions[1] * _dimensions[2] + j * _dimensions[2] + k] = value;
}

void sTensor::add3d(const uint i, const uint j, const uint k, const float value)
{
    assert(_rank == 3);
    _storage.get()[i * _dimensions[1] * _dimensions[2] + j * _dimensions[2] + k] += value;
}

pTensor sTensor::sum_rank2(pTensor& result, const uint dim)
{
    uint resultIndices[1] = {};

    float *s = _storage.get();
    float *r = result->_storage.get();
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
            r[resultIndices[0]] += s[i * _dimensions[1] + j];
            //(*result)(resultIndices[0]) += operator()(i, j);
        }
    }
    return result;
}

pTensor sTensor::sum_rank3(pTensor& result, const uint dim)
{
    uint resultIndices[2] = {};

    const float* s = _storage.get();
    float* r = result->_storage.get();
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
                r[resultIndices[0] * result->_dimensions[1] + resultIndices[1]] += s[i * _dimensions[1] * _dimensions[2] + j * _dimensions[2] + k];
                //(*result)(resultIndices[0], resultIndices[1]) += operator()(i, j, k);
            }
        }
    }
    return result;
}

pTensor sTensor::sum_rank4(pTensor& result, const uint dim)
{
    uint resultIndices[3] = {};

    const float* s = _storage.get();
    float* r = result->_storage.get();
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
                    r[resultIndices[0] * result->_dimensions[1] * result->_dimensions[2] + resultIndices[1] * result->_dimensions[2] + resultIndices[2]] += s[i * _dimensions[1] * _dimensions[2] * _dimensions[3] + j * _dimensions[2] * _dimensions[3] + k * _dimensions[3] + l];
                    //(*result)(resultIndices[0], resultIndices[1], resultIndices[2]) += operator()(i, j, k, l);
                }
            }
        }
    }
    return result;
}


pTensor sTensor::sum(const uint dim)
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
    pTensor result = sTensor::Zeros(_rank - 1, new_dims);
    result->set_label(_label);

    assert(_rank > 1);

    if (_rank == 2) return sum_rank2(result, dim)->autolog("sum_rank2", begin);;
    if (_rank == 3) return sum_rank3(result, dim)->autolog("sum_rank3", begin);
    if (_rank == 4) return sum_rank4(result, dim)->autolog("sum_rank4", begin);

    return result->autolog("sum_dim", begin);
}

// should this squeeze the final dimension?. yes!
pTensor sTensor::sum_final_dimension()
{
    timepoint begin = now();
    uint dim = _rank - 1;
    uint new_dims[sTENSOR_MAX_DIMENSIONS];
    memcpy(new_dims, _dimensions, _rank * sizeof(uint));
    new_dims[dim] = 1;

    sTensor result(_rank, new_dims);
    result.set_label(_label);
    result._grad = _grad;

    const float* s = _storage.get();
    float* o = result._storage.get();
    const uint finalDimSize = _dimensions[dim];
    const uint nItems = _storageSize / finalDimSize;
    for (uint r = 0; r < nItems; r++)
    {
        float sum = 0;
        for (uint c = 0; c < finalDimSize; c++)
        {
            sum += s[r + c];
        }
        o[r] = sum;
    }
    result.squeeze_();
    return result.autolog("sum_final_dimension", begin);
}

pTensor sTensor::greater_than(const float value)
{
    timepoint begin = now();
    pTensor result = clone();
    const float* s = _storage.get();
    float* o = result->_storage.get();
    for (uint i = 0; i < _storageSize; i++)
        o[i] = s[i] > value ? 1.0f : 0.0f;
    return result->autolog("greater_than", begin);
}

pTensor sTensor::less_than(const float value)
{
    timepoint begin = now();
    pTensor result = clone();
    const float* s = _storage.get();
    float* o = result->_storage.get();
    for (uint i = 0; i < _storageSize; i++)
        o[i] = s[i] < value ? 1.0f : 0.0f;
    return result->autolog("less_than", begin);
}

pTensor sTensor::pad2d(const uint pad) const
{
    timepoint begin = now();
    assert(_rank == 2);
    const uint nrows = dim(0);
    const uint ncols = dim(1);
    const uint new_nrows = nrows + 2 * pad;
    const uint new_ncols = ncols + 2 * pad;

    pTensor result = sTensor::Zeros(new_nrows, new_ncols);
    result->set_label(_label);

    const float* s = _storage.get();
    float* o = result->_storage.get();
    for (uint r = 0; r < nrows; r++)
    {
        for (uint c = 0; c < ncols; c++)
        {
            o[(r + pad) * new_ncols + c + pad] = s[r * ncols + c];
        }
    }
    return result->autolog("pad2d", begin);
}

pTensor sTensor::pad3d(const uint amount) const
{
    timepoint begin = now();
    assert(_rank == 3);

    const uint n = dim(0);
    const uint h = dim(1);
    const uint w = dim(2);

    pTensor result = sTensor::Zeros(n, h + 2 * amount, w + 2 * amount);

    // Copy each image into the center of the corresponding padded image
    const float* s = _storage.get();
    float* r = result->_storage.get();
    for (uint i = 0; i < n; i++)
    {
        for (uint j = 0; j < h; j++)
        {
            for (uint k = 0; k < w; k++)
            {
                r[i * (h + 2 * amount) * (w + 2 * amount) + (j + amount) * (w + 2 * amount) + k + amount] = s[i * h * w + j * w + k];
                //result->set3d(i, j + amount, k + amount, s[i * h * w + j * w + k]);
            }
        }
    }
    return result;
}

// ---------------- tensor scalar operators -----------------

pTensor sTensor::operator+(const float value) const
{
    pTensor result = clone();
    float* o = result->_storage.get();
    for (uint i = 0; i < _storageSize; i++)
        o[i] += value;
    return result;
}

pTensor sTensor::operator-(const float value) const
{
    pTensor result = clone();
    float* o = result->_storage.get();
    for (uint i = 0; i < _storageSize; i++)
        o[i] -= value;
    return result;
}

pTensor sTensor::operator*(const float value) const
{
    pTensor result = clone();
    float* o = result->_storage.get();
    for (uint i = 0; i < _storageSize; i++)
        o[i] *= value;
    return result;
}

pTensor sTensor::operator/(const float value) const
{
    pTensor result = clone();
    float* o = result->_storage.get();
    for (uint i = 0; i < _storageSize; i++)
        o[i] /= value;
    return result;
}

pTensor sTensor::operator+=(const float value)
{
    timepoint begin = now();
    float* s = _storage.get();
    for (uint i = 0; i < _storageSize; i++)
        s[i] += value;
    return autolog("operator+=", begin);
}

pTensor sTensor::operator-=(const float value)
{
    timepoint begin = now();
    float* s = _storage.get();
    for (uint i = 0; i < _storageSize; i++)
        s[i] -= value;
    return autolog("operator-=", begin);
}

pTensor sTensor::operator*=(const float value)
{
    timepoint begin = now();
    float* s = _storage.get();
    for (uint i = 0; i < _storageSize; i++)
        s[i] *= value;
    return autolog("operator*=", begin);
}

pTensor sTensor::operator/=(const float value)
{
    timepoint begin = now();
    float* s = _storage.get();
    for (uint i = 0; i < _storageSize; i++)
        s[i] /= value;
    return autolog("operator/=", begin);
}

// ---------------- utils -----------------

void sTensor::FromHost(CudaTensor& t) const
{
    memcpy(&t._dimensions[0], &_dimensions[0], _rank * sizeof(uint));
    t._storage = _storage.get();
    t._storageSize = _storageSize;
}

void sTensor::ToHost(const CudaTensor& t)
{
    memcpy(_storage.get(), t._storage, _storageSize * sizeof(float));
}

pTensor sTensor::MatMult(const pTensor& other) const
{
    timepoint begin = now();
    assert(_rank == 2);
    assert(other->_rank == 2);

    const uint nrows = dim(0);
    const uint ncols = dim(1);
    const uint other_nrows = other->dim(0);
    const uint other_ncols = other->dim(1);
    assert(ncols == other_nrows);

    pTensor result = Dims(nrows, other_ncols);
    result->set_label(_label);

    const float* s = _storage.get();
    const float* o = other->_storage.get();
    float* r = result->_storage.get();

    for (uint i = 0; i < nrows; i++)
    {
        for (uint j = 0; j < other_ncols; j++)
        {
            const uint offset = i * other_ncols + j;
            r[offset] = 0;
            for (uint k = 0; k < ncols; k++)
            {
                r[offset] += s[i * ncols + k] * o[k * other_ncols + j];
                //result->add2d(i, j, get2d(i, k) * other->get2d(k, j));
            }
        }
    }

    return result->autolog("matmul", begin);
}

float sTensor::DotProduct(const pTensor& other)
{
    assert(_rank == 1);
    assert(other->_rank == 1);
    assert(size() == other->size());

    float result = 0;
    const float* s = _storage.get();
    const float* o = other->_storage.get();
    for (uint i = 0; i < size(); ++i)
    {
        result += s[i] * o[i];
    }
    return result;
}

pTensor sTensor::Transpose() const
{
    assert(_rank == 2);
    pTensor result = Dims(dim(1), dim(0));
    result->set_label(_label);

    for (uint r = 0; r < dim(0); r++)
    {
        for (uint c = 0; c < dim(1); c++)
        {
            result->set2d(c, r, get2d(r, c));
        }
    }
    return result;
}

pTensor sTensor::clone() const
{
    pTensor result = pTensor(new sTensor(_rank, _dimensions));
    memcpy(result->_storage.get(), _storage.get(), _storageSize * sizeof(float));
    result->set_label(_label);
    result->_grad = _grad;
    return result;
}

pTensor sTensor::clone_shallow() const
{
    pTensor result = pTensor(new sTensor(_rank, _dimensions));
    result->_storage = _storage;
    result->set_label(_label);
    result->_grad = _grad;
    return result;
}

pTensor sTensor::select(const uint dim, const uint index) const
{
    timepoint begin = now();
    assert(dim < _rank);
    assert(index < _dimensions[dim]);

    uint new_dims[sTENSOR_MAX_DIMENSIONS];
    uint pos = 0;
    for (uint i = 0; i < _rank; i++)
    {
        if (i == dim)
        {
            new_dims[pos++] = 1;
        }
        else
        {
            new_dims[pos++] = _dimensions[i];
        }
    }

    pTensor result = sTensor::Dims(_rank, new_dims);
    result->set_label(_label);

    if (_rank == 1)
    {
        result->set1d(0, get1d(index));
    }
    else if (_rank == 2)
    {
        if (dim == 0)
        {
            for (uint c = 0; c < _dimensions[1]; c++)
            {
                result->set2d(0, c, get2d(index, c));
            }
        }
        else if (dim == 1)
        {
            for (uint r = 0; r < _dimensions[0]; r++)
            {
                result->set2d(r, 0, get2d(r, index));
            }
        }
    }
    else if (_rank == 3)
    {
        if (dim == 0)
        {
            for (uint j = 0; j < _dimensions[1]; j++)
            {
                for (uint k = 0; k < _dimensions[2]; k++)
                {
                    result->set3d(0, j, k, get3d(index, j, k));
                }
            }
        }
        else if (dim == 1)
        {
            for (uint i = 0; i < _dimensions[0]; i++)
            {
                for (uint k = 0; k < _dimensions[2]; k++)
                {
                    result->set3d(i, 0, k, get3d(i, index, k));
                }
            }
        }
        else if (dim == 2)
        {
            for (uint i = 0; i < _dimensions[0]; i++)
            {
                for (uint j = 0; j < _dimensions[1]; j++)
                {
                    result->set3d(i, j, 0, get3d(i, j, index));
                }
            }
        }
    }
    return result->autolog("select", begin);
}

pTensor sTensor::row2d(const uint row) const
{
    assert(_rank == 2);
    assert(row < dim(0));

    pTensor result = Dims(uint(1), dim(1));
    result->set_label(_label);

    const float* s = _storage.get();
    float* r = result->_storage.get();
    for (uint c = 0; c < dim(1); c++)
    {
        r[c] = s[row * dim(1) + c];
        //(*result)(uint(0), c) = operator()(row, c);
    }
    return result;
}

//pTensor sTensor::index_select(const uint row)

pTensor sTensor::column2d(const uint col) const
{
    assert(_rank == 2);
    assert(col < dim(1));

    pTensor result = Dims(dim(0), uint(1));
    result->set_label(_label);

    const float* s = _storage.get();
    float* rr = result->_storage.get();
    for (uint r = 0; r < dim(0); r++)
    {
        rr[r] = s[r * dim(1) + col];
        //(*result)(r, uint(0)) = operator()(r, col);
    }
    return result;
}

pTensor sTensor::random_sample_rows(const float p)
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

    pTensor result = Dims(count, dim(1));
    result->set_label(_label);

    sTensor& r = *result;
    for (uint i = 0; i < count; i++)
    {
        const uint row = ids[i];
        for (uint c = 0; c < dim(1); c++)
        {
            r.set2d(i, c, get2d(row, c));
        }
    }
    return result->autolog("random_sample", begin);
}

pTensor sTensor::slice_rows(const uint start, const uint end) const
{
    timepoint begin = now();
    assert(start < end);
    assert(end <= dim(0));

    uint batchSize = end - start;

    uint n = 1;
    for (uint i = 1; i < _rank; i++)
    {
        n *= _dimensions[i];
    }

    uint new_dims[sTENSOR_MAX_DIMENSIONS] = {};
    memcpy(new_dims, _dimensions, _rank * sizeof(uint));
    new_dims[0] = batchSize;

    pTensor result = pTensor(new sTensor(_rank, new_dims));
    result->set_label(_label);

    const uint index = start * n;
    memcpy(result->_storage.get(), _storage.get() + index, batchSize * n * sizeof(float));
    return result->autolog("slice_rows", begin);
}

pTensor sTensor::slice2d(const uint rowStart, const uint rowEnd, const uint colStart, const uint colEnd) const
{
    timepoint begin = now();
    assert(rowStart < rowEnd);
    assert(colStart < colEnd);
    assert(rowEnd <= dim(0));
    assert(colEnd <= dim(1));

    pTensor result = Dims(rowEnd - rowStart, colEnd - colStart);

    const uint stride = dim(1);
    const float* s = _storage.get();
    float* rr = result->_storage.get();
    for (uint r = rowStart; r < rowEnd; r++)
    {
        for (uint c = colStart; c < colEnd; c++)
        {
            //rr[(r - rowStart) * (colEnd - colStart) + (c - colStart)] = s[r * stride + c];
            result->set2d(r - rowStart, c - colStart, get2d(r, c));
        }
    }
    return result->autolog("slice2d", begin);
}


void sTensor::put_rows(const uint start, const pTensor& other)
{
    assert(_rank == other->rank());
    assert(start + other->dim(0) < dim(0));

    uint n = 1;
    for (uint i = 1; i < _rank; i++)
    {
        n *= _dimensions[i];
    }
    const uint index = start * n;
    memcpy(_storage.get() + index, other->_storage.get(), other->size() * sizeof(float));
}

