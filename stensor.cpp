#include "stensor.h"

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

sTensorRowIterator::sTensorRowIterator(sTensor& tensor, uint row) : _tensor(tensor), _row(row)
{
}

sTensorRowIterator sTensorRowIterator::operator++()
{
    _row++;
    if (_row >= _tensor.dim(0))
    {
        _row = _tensor.dim(0);
    }
    return *this;
}

sTensorRowIterator sTensorRowIterator::operator++(int)
{
    sTensorRowIterator copy(*this);
    _row++;
    if (_row >= _tensor.dim(0))
    {
        _row = _tensor.dim(0);
    }
    return copy;
}

bool sTensorRowIterator::operator!=(const sTensorRowIterator& other) const
{
    return _row != other._row;
}

sTensor sTensorRowIterator::operator*()
{
    sTensor row(_tensor.rank()-1, &_tensor._dimensions[1]);
    row._storage = _tensor.data() + _row * _tensor.dim(1);
    row._storageOwned = false;
    return row;
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
