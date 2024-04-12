//tests


#include "scratch/stensor.h"

const char* codeRed = "\033[31m";
const char* codeGreen = "\033[32m";
const char* codeYellow = "\033[33m";
const char* codeBlue = "\033[34m";
const char* codeReset = "\033[0m";

bool _fail = false;

struct sTest
{
    sTest(const char* name, void (*test)()) : name(name), test(test)
    {
        _fail = false;
        test();
        const char* code = _fail ? codeRed : codeGreen;
        const char* text  = _fail ? "FAIL" : "PASS";
        std::cout << code << text << ": " << name << codeReset << std::endl;
    }
    const char* name;
    void (*test)();
};

#define sTEST(name) { sTest x(#name,t_##name); };

void expect_eq_int(const uint a, const uint b) {
    if (a != b) {
        std::cout << "Expected int " << a << " to equal " << b << std::endl;
        //assert(false);
        _fail |= true;
    }
}

void expect_eq_float(const float a, const float b) {
    constexpr float epsilon = 0.0001f;
    if (std::abs(a-b) > epsilon) {
        std::cout << "Expected float " << a << " to equal " << b << std::endl;
        //assert(false);
        _fail |= true;
    }
}

void expect_tensor_eq(const pTensor& a, const pTensor& b) {
    if (a != b) {
        std::cout << "Expected tensors equal" << std::endl;
        //assert(false);
        _fail |= true;
    }
}

void expect_tensor_size(const pTensor& a, const int size) {
    if (a->size() != size) {
        std::cout << "Expected tensor size " << a->size() << " to equal " << size << std::endl;
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
