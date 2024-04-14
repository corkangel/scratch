#include "tests.h"


const char* codeRed = "\033[31m";
const char* codeGreen = "\033[32m";
const char* codeYellow = "\033[33m";
const char* codeBlue = "\033[34m";
const char* codeReset = "\033[0m";

bool _fail = false;
const char* _name = nullptr;

const char* _file = nullptr;
int _line = 0;

sTest::sTest(const char* name, void (*test)()) : name(name), test(test)
{
    _fail = false;
    _name = name;
    test();
    const char* code = _fail ? codeRed : codeGreen;
    const char* text  = _fail ? "FAIL" : "PASS";
    std::cout << code << text << ": " << name << codeReset << std::endl;
}

void expect_eq_int_(const uint a, const uint b, const char* file, const int line)
{
    _file = file;
    _line = line;
    if (a != b) {
        std::cout << "[" << _file << ":" << _line << ":" << _name << "] Expected int " << a << " to equal " << b << std::endl;
        //assert(false);
        _fail |= true;
    }
}

void expect_eq_float_(const float a, const float b, const char* file, const int line)
{
    _file = file;
    _line = line;
    constexpr float epsilon = 0.0001f;
    if (std::abs(a-b) > epsilon) {
        std::cout << "[" << _file << ":" << _line << ":" << _name << "] Expected float " << a << " to equal " << b << std::endl;
        //assert(false);
        _fail |= true;
    }
}

pTensor create_tensor_ones(const uint dims) 
{
    return pTensor(sTensor::Ones(dims));
}

pTensor create_tensor_zeros(const uint dims)
{
    return pTensor(sTensor::Zeros(dims));
}

pTensor create_tensor_linear(const uint dims)
{
    return pTensor(sTensor::Linear(0, 1, dims));
}

pTensor create_tensor_linear2d(const uint d0, const uint d1)
{
    return pTensor(sTensor::Linear(0, 1, d0, d1));
}
