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

void expect_eq_int(const uint a, const uint b);
void expect_eq_float(const float a, const float b);
void expect_tensor_eq(const pTensor& a, const pTensor& b);
void expect_tensor_size(const pTensor& a, const int size);
pTensor create_tensor_ones(const uint dims);
pTensor create_tensor_zeros(const uint dims);
pTensor create_tensor_linear(const uint dims);
pTensor create_tensor_linear2d(const uint d0, const uint d1);
