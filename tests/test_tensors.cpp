//tests

#include "scratch/stensor.h"
#include "tests.h"

void t_tensor_ones()
{
    pTensor a = create_tensor_ones(3);
    expect_tensor_size(a, 3);
    expect_eq_float(a->at(0), 1.0);
    expect_eq_float(a->at(1), 1.0);
    expect_eq_float(a->at(2), 1.0);
}

void t_tensor_zeros()
{
    pTensor a = create_tensor_zeros(3);
    expect_tensor_size(a, 3);
    expect_eq_float(a->at(0), 0.0);
    expect_eq_float(a->at(1), 0.0);
    expect_eq_float(a->at(2), 0.0);
}

void t_tensor_add()
{
    pTensor a = create_tensor_ones(3);
    pTensor b = create_tensor_ones(3);
    pTensor c = a + b;
    expect_tensor_size(c, 3);
    expect_eq_float(c->at(0), 2.0);
    expect_eq_float(c->at(1), 2.0);
    expect_eq_float(c->at(2), 2.0);
}

void t_tensor_multiply()
{
    pTensor a = create_tensor_ones(3);
    pTensor b = create_tensor_ones(3);
    pTensor c = a * b;
    expect_tensor_size(c, 3);
    expect_eq_float(c->at(0), 1.0);
    expect_eq_float(c->at(1), 1.0);
    expect_eq_float(c->at(2), 1.0);
}

void t_tensor_multiply2()
{
    pTensor a = create_tensor_linear2d(2, 2);
    pTensor b = create_tensor_linear2d(2, 2);
    pTensor c = a * b;
    expect_tensor_size(c, 4);
    expect_eq_float(c->at(0), 0.0);
    expect_eq_float(c->at(1), 1.0);
    expect_eq_float(c->at(2), 4.0);
    expect_eq_float(c->at(3), 9.0);
}

void t_tensor_subtract()
{
    pTensor a = create_tensor_ones(3);
    pTensor b = create_tensor_ones(3);
    pTensor c = a - b;
    expect_tensor_size(c, 3);
    expect_eq_float(c->at(0), 0.0);
    expect_eq_float(c->at(1), 0.0);
    expect_eq_float(c->at(2), 0.0);
}

void t_tensor_subtract2()
{
    pTensor a = create_tensor_linear2d(2, 2);
    pTensor c = a - a;
    expect_tensor_size(c, 4);
    expect_eq_float(c->at(0), 0.0);
    expect_eq_float(c->at(1), 0.0);
    expect_eq_float(c->at(2), 0.0);
    expect_eq_float(c->at(3), 0.0);
}

void t_tensor_divide()
{
    pTensor a = create_tensor_ones(3);
    pTensor b = create_tensor_ones(3);
    pTensor c = a / b;
    expect_tensor_size(c, 3);
    expect_eq_float(c->at(0), 1.0);
    expect_eq_float(c->at(1), 1.0);
    expect_eq_float(c->at(2), 1.0);
}

void t_tensor_dot()
{
    pTensor a = create_tensor_linear(3);
    pTensor b = create_tensor_linear(3);
    float c = a->DotProduct(b);
    expect_eq_float(c, 5.0);
}

void t_tensor_transpose()
{
    pTensor a = create_tensor_linear2d(2,1);
    pTensor b = a->Transpose();
    expect_tensor_size(b, 2);
    expect_eq_int(b->dim(0), 1);
    expect_eq_int(b->dim(1), 2);
}

void t_tensor_matmul()
{
    pTensor a = create_tensor_linear2d(2,2);
    pTensor b = create_tensor_linear2d(2,2);
    pTensor c = a->MatMult(b);
    expect_tensor_size(c, 4);
    expect_eq_int(c->dim(0), 2);
    expect_eq_int(c->dim(1), 2);
    expect_eq_float(c->at(0), 2.0);
    expect_eq_float(c->at(1), 3.0);
    expect_eq_float(c->at(2), 6.0);
    expect_eq_float(c->at(3), 11.0);
}

void t_tensor_mean()
{
    pTensor a = create_tensor_linear2d(2, 2);
    expect_eq_float(a->mean(), 1.5f);
    expect_eq_float(a->std(), 1.11803f); // NOTE "unbiased" std. 1.29099 is the "biased" std.
}

void t_tensor_math()
{
    pTensor a = create_tensor_ones(2);
    a->set1d(0, 4.0f);
    a->set1d(1, 8.0f);

    expect_eq_float(a->sum(), 12.0f);
    expect_eq_float(a->min(), 4.0f);
    expect_eq_float(a->max(), 8.0f);
    expect_eq_float(a->mean(), 6.0f);
    expect_eq_float(a->mse(), 40.0f);
    expect_eq_float(a->rmse(), 6.32456f);
}

void t_tensor_slice2d()
{
    pTensor a = create_tensor_linear2d(2, 2);
    pTensor b = a->slice_rows(0, 1);
    expect_tensor_size(b, 2);
    expect_eq_float(b->at(0), 0.0);
    expect_eq_float(b->at(1), 1.0);
}

void t_tensor_slice_rows()
{
    pTensor a = sTensor::Ones(4, 4);
    pTensor b = a->slice_rows(0, 1);
    expect_tensor_size(b, 4);
    expect_eq_float(b->at(0), 1.0);
    expect_eq_float(b->at(1), 1.0);

}

void t_tensor_operators()
{
    pTensor a = create_tensor_ones(3);
    pTensor b = create_tensor_ones(3);
    pTensor c = a + b;
    expect_tensor_size(c, 3);
    expect_eq_float(c->at(0), 2.0);
    expect_eq_float(c->at(1), 2.0);
    expect_eq_float(c->at(2), 2.0);

    c = a - b;
    expect_tensor_size(c, 3);
    expect_eq_float(c->at(0), 0.0);
    expect_eq_float(c->at(1), 0.0);
    expect_eq_float(c->at(2), 0.0);

    c = a * b;
    expect_tensor_size(c, 3);
    expect_eq_float(c->at(0), 1.0);
    expect_eq_float(c->at(1), 1.0);
    expect_eq_float(c->at(2), 1.0);

    c = a / b;
    expect_tensor_size(c, 3);
    expect_eq_float(c->at(0), 1.0);
    expect_eq_float(c->at(1), 1.0);
    expect_eq_float(c->at(2), 1.0);
}

void t_tensor_clone()
{
    pTensor a = create_tensor_ones(3);
    pTensor b = a->clone();
    expect_eq_int(a->size(), b->size());

    expect_eq_float(a->at(0), b->at(0));
    expect_eq_float(a->at(1), b->at(1));

    b->set1d(0, 4.0f);

    expect_eq_float(a->at(0), 1.0f);
    expect_eq_float(b->at(0), 4.0f);
}


void t_tensor_clone_shallow()
{
    pTensor a = create_tensor_ones(3);
    pTensor b = a->clone_shallow();
    expect_eq_int(a->size(), b->size());

    expect_eq_float(a->at(0), b->at(0));
    expect_eq_float(a->at(1), b->at(1));

    b->set1d(0, 4.0f);

    expect_eq_float(a->at(0), 4.0f);
    expect_eq_float(b->at(0), 4.0f);
}

void t_tensor_broadcast1d()
{
    pTensor a = sTensor::Ones(4);
    pTensor b = sTensor::Fill(3, 1);
    pTensor c = a * b;
    expect_tensor_size(c, 4);
    expect_eq_float(c->at(0), 3.0);
    expect_eq_float(c->at(1), 3.0);
    expect_eq_float(c->at(2), 3.0);
}

void t_tensor_broadcast2d()
{
    pTensor a = sTensor::Ones(2, 2);    // 2x2
    pTensor b = sTensor::Fill(4, 2, 1); // 2x1
    pTensor c = a * b;
    expect_tensor_size(c, 4);
    expect_eq_float(c->at(0), 4.0);
    expect_eq_float(c->at(1), 4.0);
    expect_eq_float(c->at(2), 4.0);
    expect_eq_float(c->at(3), 4.0);

    // reverse order
    pTensor d = b * a;
    expect_eq_float(d->at(0), 4.0);
    expect_eq_float(d->at(1), 4.0);
    expect_eq_float(d->at(2), 4.0);

}

void t_tensor_broadcast3d()
{
    pTensor a = sTensor::Ones(2, 2, 2);    // 2x2x2
    pTensor b = sTensor::Fill(4, 2, 1, 1); // 2x1x1
    pTensor c = a * b;
    expect_tensor_size(c, 8);
    expect_eq_float(c->at(0), 4.0);
    expect_eq_float(c->at(1), 4.0);
    expect_eq_float(c->at(2), 4.0);
    expect_eq_float(c->at(3), 4.0);
    expect_eq_float(c->at(4), 4.0);
    expect_eq_float(c->at(5), 4.0);
    expect_eq_float(c->at(6), 4.0);
    expect_eq_float(c->at(7), 4.0);

    // reverse order
    pTensor d = b * a;
    expect_eq_float(d->at(0), 4.0);
    expect_eq_float(d->at(1), 4.0);
    expect_eq_float(d->at(2), 4.0);
    expect_eq_float(d->at(3), 4.0);
    expect_eq_float(d->at(4), 4.0);
    expect_eq_float(d->at(5), 4.0);
    expect_eq_float(d->at(6), 4.0);
    expect_eq_float(d->at(7), 4.0);

    // broadcast two dimensions
    pTensor mid = sTensor::Fill(4, 1, 2, 1); // 1x2x1
    pTensor e = a * mid;
    expect_tensor_size(e, 8);

    // broadcast all dimensions
    pTensor all = sTensor::Fill(4, 1, 1, 1); // 1x2x1
    pTensor f = a * all;
    expect_tensor_size(f, 8);
}

void t_tensor_compare()
{
    pTensor five = sTensor::Fill(4.0f, 5);

    pTensor lt = five->less_than(6.0f);
    expect_eq_float(lt->at(0), 1.0f);

    pTensor gt = five->greater_than(6.0f);
    expect_eq_float(gt->at(0), 0.0f);
}

void t_tensor_sum()
{
    pTensor a = sTensor::Linear(4, 1, 4);
    expect_eq_float(a->sum(), 22.0f);

    pTensor b = sTensor::Ones(5, 3);
    expect_eq_float(b->sum_rows()->at(0), 5.f);
    expect_eq_float(b->sum_columns()->at(0), 3.f);
}

void t_tensor_squeeze()
{
    pTensor a = sTensor::Ones(1, 2, 1, 3);

    // remove all dimensions of size 1
    pTensor b = a->squeeze();
    expect_eq_int(b->dim(0), 2);
    expect_eq_int(b->dim(1), 3);
    expect_eq_int(a->size(), b->size());

    // remove a specific dimension at start
    pTensor c = a->squeeze(0);
    expect_eq_int(a->size(), c->size());
    expect_eq_int(c->dim(0), 2);
    expect_eq_int(c->dim(1), 1);
    expect_eq_int(c->dim(2), 3);

    // remove a specific dimension in middle
    pTensor d = a->squeeze(2);
    expect_eq_int(a->size(), d->size());
    expect_eq_int(d->dim(0), 1);
    expect_eq_int(d->dim(1), 2);
    expect_eq_int(d->dim(2), 3);
}

void t_tensor_unsqueeze()
{
    pTensor a = sTensor::Ones(2, 1, 3);

    // add a dimension of size 1 at start
    pTensor b = a->unsqueeze(0);
    expect_eq_int(a->size(), b->size());
    expect_eq_int(b->dim(0), 1);
    expect_eq_int(b->dim(1), 2);
    expect_eq_int(b->dim(2), 1);
    expect_eq_int(b->dim(3), 3);

    // add a dimension of size 1 in middle
    pTensor c = a->unsqueeze(2);
    expect_eq_int(a->size(), c->size());
    expect_eq_int(c->dim(0), 2);
    expect_eq_int(c->dim(1), 1);
    expect_eq_int(c->dim(2), 1);
    expect_eq_int(c->dim(3), 3);
}

void t_tensor_view()
{
    pTensor a = sTensor::Ones(2, 3);
    pTensor b = a->view_(1, 6);
    expect_eq_int(a->size(), b->size()); // actually the same object

    //a->view_(1, 2, 3); // would assert, view() cannot change rank

    // reshape() can change rank
    pTensor c = a->reshape_(1, 6); 
    expect_eq_int(a->size(), c->size());
}

void t_tensor_pad()
{
    pTensor image = sTensor::Ones(14, 14);
    pTensor padded1 = image->pad2d(1);
    expect_eq_int(padded1->dim(0), 16);
    expect_eq_int(padded1->dim(1), 16);
    expect_eq_float(padded1->at(0), 0.0f);

    pTensor images = sTensor::Ones(6, 1, 14, 14);
    pTensor padded2 = images->pad_images(1);
    expect_eq_int(padded2->dim(0), 6);
    expect_eq_int(padded2->dim(1), 1);
    expect_eq_int(padded2->dim(2), 16);
    expect_eq_int(padded2->dim(3), 16);
    expect_eq_float(padded2->at(0), 0.0f);

    pTensor channels = sTensor::Ones(4, 3, 14, 14);
    pTensor padchan = channels->pad_images(1);
    expect_eq_int(padchan->dim(0), 4);
    expect_eq_int(padchan->dim(1), 3);
    expect_eq_int(padchan->dim(2), 16);
    expect_eq_int(padchan->dim(3), 16);
    expect_eq_float(padchan->at(0), 0.0f);
}

void t_tensor_select()
{
    pTensor a = sTensor::Ones(2, 3, 4);

    pTensor result1 = a->select(0, 0);
    expect_eq_int(result1->rank(), 3);
    expect_eq_int(result1->dim(0), 1);
    expect_eq_int(result1->dim(1), 3);
    expect_eq_int(result1->dim(2), 4);

    pTensor result2 = a->select(1, 0);
    expect_eq_int(result2->rank(), 2);
    expect_eq_int(result2->dim(0), 1);
    expect_eq_int(result2->dim(1), 4);
}

void t_tensor_get_set()
{
    pTensor a = sTensor::Ones(4);
    a->set1d(0, 4.0f);
    expect_eq_float(a->get1d(0), 4.0f);
    a->add1d(0, 4.0f);
    expect_eq_float(a->get1d(0), 8.0f);

    pTensor b = sTensor::Ones(2, 2);
    b->set2d(0, 1, 4.0f);
    expect_eq_float(b->get2d(0, 1), 4.0f);
    b->add2d(0, 1, 4.0f);
    expect_eq_float(b->get2d(0, 1), 8.0f);

    pTensor c = sTensor::Ones(2, 2, 2);
    c->set3d(0, 1, 1, 4.0f);
    expect_eq_float(c->get3d(0, 1, 1), 4.0f);
    c->add3d(0, 1, 1, 4.0f);
    expect_eq_float(c->get3d(0, 1, 1), 8.0f);

    pTensor d = sTensor::Ones(2, 2, 2, 2);
    d->set4d(0, 1, 1, 1, 4.0f);
    expect_eq_float(d->get4d(0, 1, 1, 1), 4.0f);
    d->add4d(0, 1, 1, 1, 4.0f);
    expect_eq_float(d->get4d(0, 1, 1, 1), 8.0f);
}

void test_tensors()
{
    sTEST(tensor_ones);
    sTEST(tensor_zeros);
    sTEST(tensor_add);
    sTEST(tensor_multiply);
    sTEST(tensor_multiply2);
    sTEST(tensor_subtract);
    sTEST(tensor_subtract2);
    sTEST(tensor_divide);
    sTEST(tensor_dot);
    sTEST(tensor_transpose);
    sTEST(tensor_matmul);
    sTEST(tensor_mean);
    sTEST(tensor_math);
    sTEST(tensor_slice2d);
    sTEST(tensor_slice_rows);
    sTEST(tensor_operators);
    sTEST(tensor_clone);
    sTEST(tensor_clone_shallow);
    sTEST(tensor_broadcast1d);
    sTEST(tensor_broadcast2d);
    sTEST(tensor_broadcast3d);
    sTEST(tensor_compare);
    sTEST(tensor_sum);
    sTEST(tensor_squeeze);
    sTEST(tensor_unsqueeze);
    sTEST(tensor_view);
    sTEST(tensor_pad);
    sTEST(tensor_select);
    sTEST(tensor_get_set);

}