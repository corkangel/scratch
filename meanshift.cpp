#include <chrono>

#include "stensor.h"

const uint numCentroids = 6;
const uint numSample = 250;
const float spread = 5.f;

struct meandata
{
    sTensor centroids = sTensor::Dims(numCentroids, uint(2));
    sTensor samples = sTensor::Dims(0,2);
};

meandata _data;

sTensor& meanshift_centroids()
{
    return _data.centroids;
}

sTensor& meanshift_samples()
{
    return _data.samples;
}

void meanshift_init()
{
    _data.centroids = sTensor::Randoms(numCentroids, uint(2)).multiply_(70).add_(-35.f);
    _data.samples = sTensor::Dims(0, 2);

    slog("centroids", _data.centroids);
    slog("");

    // generate samples
    for (sTensorRowIterator riter = _data.centroids.begin_rows(); riter != _data.centroids.end_rows(); ++riter)
    {
        sTensor centroid = *riter; // 2 matrix
        centroid.unsqueeze_(0);    // 1x2 matrix

        sTensor batch = sTensor::NormalDistribution(0.f, spread, numSample, uint(2)); // 250x2 matrix
        batch = batch + centroid; // 250x2 matrix broadcasted with 1x2 matrix
        _data.samples.cat0_(batch);
    }

    slog("samples", _data.samples);
}

void meanshift_iterate_rows()
{

    // process a batch of samples once
    for (sTensorRowIterator riter = _data.samples.begin_rows(); riter != _data.samples.end_rows(); ++riter)
    {
        sTensor sample = *riter; // 2 matrix
        sample.unsqueeze_(0);    // 1x2 matrix
        //if (riter.row() == 0) slog("sample", sample);

        //sTensor diffs = (samples - sample);
        ////if (riter.row() == 0) slog("diffs", diffs);

        //diffs.pow_(2);
        ////if (riter.row() == 0) slog("diffs^2", diffs);

        //sTensor sum_columns = diffs.sum_columns();
        ////if (riter.row() == 0) slog("sum_columns", sum_columns);

        //sTensor sum_columns_sqrt = sum_columns.sqrt_();
        ////if (riter.row() == 0) slog("sum_columns_sqrt", sum_columns_sqrt);

        //sTensor weights = sum_columns_sqrt.gaussian_(2.5f);
        ////if (riter.row() == 0) slog("weights", weights);

        //sTensor weighted_samples = samples * weights;
        ////if (riter.row() == 0) slog("weighted_samples", weighted_samples);

        //sTensor sum_weights = weighted_samples.sum_rows();
        ////if (riter.row() == 0) slog("sum_weights", sum_weights);

        //const float sum = weights.sum();
        //sTensor sample_new = sum_weights / sum; // 1x2 matrix

        ////if (riter.row() == 0) slog("new", sample_new);

        // only compare against N% of the samples
        sTensor random_samples = _data.samples.random_sample_rows(0.2f);

        // all the above steps in two lines
        sTensor weights2 = (random_samples - sample).pow_(2).sum(1).sqrt_().gaussian_(2.5f).unsqueeze_(1);
        sTensor all = (random_samples * weights2).sum(0) / weights2.sum();
        //if (riter.row() == 0) slog("all", all);
        //assert(all(0, 0) == sample_new(0, 0));

        _data.samples.set_row_(riter.row(), all.squeeze_());
    }
    //slog("new_samples", _data.samples);
}


sTensor dist(sTensor& a, sTensor& b)
{
    return (a-b).pow_(2).sum(2).sqrt_();
}

void meanshift_step()
{
    auto start = std::chrono::high_resolution_clock::now();

    {
        sTensor tmp = _data.samples.clone();
        tmp.unsqueeze_(0);
        slog("samples", tmp);

        sTensor batch = _data.samples.clone();
        batch.unsqueeze_(1);
        slog("batch", batch);

        sTensor added = tmp - batch;
        slog("added", added);
        //for (auto it = added.begin_rows(); it != added.end_rows(); ++it)
        //{
        //    slog("added_row", *it);
        //}

        sTensor ddist = dist(tmp, batch);
        slog("dist", ddist);

        sTensor weights = ddist.gaussian_(2.5f);
        slog("weights", weights);

        //for (auto it = weights.begin_rows(); it != weights.end_rows(); ++it)
        //{
        //    slog("weight_row", *it);
        //}

        sTensor div = weights.sum(1).unsqueeze_(1);
        slog("div", div);

        sTensor matmul = weights.MatMult(_data.samples);
        slog("matmul", matmul);

        //sTensor num = matmul.sum(1);
        //slog("num", num);

        sTensor new_batch = matmul / div;
        slog("new_batch", new_batch);

        _data.samples = new_batch;

        //sTensor sum = tmp - batch;
        //slog("sum", sum);

        //sum.pow_(2);
        //slog("sum_pow2", sum);

        //sTensor sim_dum = sum.sum_final_dimension();
        //slog("sum_dim", sim_dum);

        //sim_dum.sqrt_();
        //slog("sum_dim_sqrt", sim_dum);

        //sTensor weights = sim_dum.gaussian_(2.5f); // 5x1500 matrix
        //slog("weights", weights);

        //weights.unsqueeze_(2); // 5x1500x1 matrix

        //sTensor num = tmp * weights; // 5x1500x2 matrix
        //slog("num", num);

        //sTensor new_batch = num.sum(1); // this needs to sum the MIDDLE dimension! -> 5x2 matrix

        //slog("new_batch", new_batch);
    }

    //{
    //    sTensor tmp = _data.samples.clone();
    //    tmp.unsqueeze_(0);

    //    sTensor sum = tmp - batch;
    //    sTensor mbatch = _data.samples.slice_rows(0, 5);
    //    sTensor mult = weights.MatMult(mbatch);
    //    slog("matrix mult", mult);
    //}



    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    slog("meanshift_step duration: %u ms", duration);
}
