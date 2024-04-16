//tests


#include "tests.h"

void _init_dml();
void _test_dml(float* dataPtr, const uint* dims, const uint ndims, const float v);
void _close_dml();


void t_dml_multiply()
{
    pTensor t = sTensor::Ones(4, 4);
    const uint dims[] = { 4, 4 };

    _test_dml(t->data(), dims, 2, 7.7f);

    expect_eq_float(t->data()[0], 7.7f);
}

void test_dml()
{
    _init_dml();

    sTEST(dml_multiply);

    _close_dml();
}