//tests


#include "tests.h"

void _init_dml();
void _test_dml(const float v);
void _close_dml();

void test_dml()
{
    _init_dml();
    _test_dml(3.3f);
    _test_dml(8.8f);
    _close_dml();
}