//tests

#include "stensor.h"

void test_tensors()
{
    {
        // raw array init
        uint dims[] = { 2, 2 };
        sTensor t1(2U, dims);

        // initializer list
        sTensor t2({ 2, 2 });
    }

    {
        sTensor ones = sTensor::Ones(2, 3);
        sTensor randoms = sTensor::Randoms(2, 3);
        sTensor nd = sTensor::NormalDistribution(0.f, 1.f, 3, 8);
        sTensor dd = sTensor::Dims(3, 2);

        sTensor wide = sTensor::Linear(0.f, 0.1f, 2, 4, 10);
        //std::cout << wide << std::endl;

        wide.view_(4, 2, 10);
        //std::cout << wide << std::endl;

        assert(wide == wide);
        assert(wide != ones);

        ones = randoms;
        assert(ones == randoms);

        ones.add_(99.f);
        std::cout << "ones: " << ones << std::endl;

        {
            sTensor o1 = sTensor::Ones(3, 3);
            sTensor o2 = sTensor::Integers(1, 3, 3);

            sTensor add = o1 + o2;
            assert(add(1, 1) == 6.f);

            sTensor mult = o1 * o2;
            assert(mult(1, 1) == 5.f);

            sTensor sub = o1 - o2;
            assert(sub(1, 1) == -4.f);
        }
    }

    {
        sTensor a = sTensor::Randoms(3, 1);
        sTensor b = sTensor::Integers(10, 3, 1);
        sTensor c = a * b;
        c = c + 22.f;

        //std::cout << "A " << a << std::endl;
        //std::cout << "B " << b << std::endl;
        //std::cout << "C " << c << std::endl;
    }

    {
        sTensor mm = sTensor::Integers(10, 4, 2);
        sTensor rows = mm.sum_rows();
        sTensor sum0 = mm.sum(0);
        assert(rows(1) == sum0(1));

        sTensor cols = mm.sum_columns();
        sTensor sum1 = mm.sum(1);
        assert(cols(1) == sum1(1));


        slog("--------------- matrix sums over dimensions --------------- ");
        slog("mm", mm);
        slog("rows", rows);
        slog("sum0", sum0);
        slog("cols", cols);
        slog("sum1", sum1);
    }

    {
        // matrix multiplication - matching dimensions
        sTensor mm = sTensor::Integers(1, 3, 3);
        sTensor mm2 = sTensor::Integers(2, 3, 3);
        sTensor mm3 = mm.MatMult(mm2);
        assert(mm3(1, 1) == 96.f);
    }

    {
        // matrix multiplication - non matching dimensions
        sTensor mm1 = sTensor::Integers(1, 4, 2);
        sTensor mm2 = sTensor::Integers(2, 2, 3);
        sTensor mm3 = mm1.MatMult(mm2);
        assert(mm3.dim(0) == 4);
        assert(mm3.dim(1) == 3);

        assert(mm3(0, 0) == 12.f);
        assert(mm3(1, 1) == 33.f);
        assert(mm3(2, 2) == 62.f);

        slog("--------------- matrix multiplication - non matching dimensions --------------- ");
        slog("m1.4x2", mm1);
        slog("m2.2x3", mm2);
        slog("mm3.4x3", mm3);
    }


    {
        sTensor m1 = sTensor::Fill(3.f, 1, 6);
        sTensor m2 = sTensor::Integers(1, 6, 1);

        sTensor result_mult = m1 * m2;
        //assert(result_mult(1, 1) == 6.6f);

        sTensor result_add = m1 + m2;
        //assert(result_add(1, 1) == 5.3f);

        slog("--------------- matrix add with broadcast ---------------");
        slog("add1", m1);
        slog("add2", m2);
        slog("result_add", result_add);
    }

    {
        sTensor mm = sTensor::Randoms(3);
        sTensor mm2 = sTensor::Ones(3);
        float dp = mm.DotProduct(mm2);
        std::cout << "dp: " << dp << std::endl;
    }
}



