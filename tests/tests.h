//tests

#pragma once

#include "scratch/stensor.h"

struct sTest
{
    sTest(const char* name, void (*test)());
    const char* name;
    void (*test)();
};

#define sTEST(name) { sTest x(#name,t_##name); };

void expect_eq_int_(const uint a, const uint b, const char* file, const int line);
#define expect_eq_int(a, b) expect_eq_int_(a, b, __FILE__, __LINE__)

void expect_eq_float_(const float a, const float b, const char* file, const int line);
#define expect_eq_float(a, b) expect_eq_float_(a, b, __FILE__, __LINE__)

pTensor create_tensor_ones(const uint dims);
pTensor create_tensor_zeros(const uint dims);
pTensor create_tensor_linear(const uint dims);
pTensor create_tensor_linear2d(const uint d0, const uint d1);
