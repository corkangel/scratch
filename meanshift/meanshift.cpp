#include <chrono>

#include "scratch/stensor.h"

const uint numCentroids = 6;
const uint numSample = 250;
const float spread = 5.f;

struct meandata
{
    pTensor centroids = sTensor::Dims(numCentroids, uint(2));
    pTensor samples = sTensor::Dims(0,2);
};

meandata _data;

pTensor meanshift_centroids()
{
    return _data.centroids;
}

pTensor meanshift_samples()
{
    return _data.samples;
}

void meanshift_init()
{
    sTensor::enableAutoLog = true;
    _data.centroids = sTensor::Randoms(numCentroids, uint(2))->multiply_(70)->add_(-35.f)->set_label("centroids");
    _data.samples = sTensor::Dims(0, 2)->set_label("samples");

    slog("centroids", _data.centroids);
    slog("");

    // generate samples
    for (sTensorRowIterator riter = _data.centroids->begin_rows(); riter != _data.centroids->end_rows(); ++riter)
    {
        pTensor centroid = *riter; // 2 matrix
        centroid->unsqueeze_(0);    // 1x2 matrix

        pTensor batch = sTensor::NormalDistribution(0.f, spread, numSample, uint(2)); // 250x2 matrix
        batch = batch + centroid; // 250x2 matrix broadcasted with 1x2 matrix
        _data.samples->cat0_(batch);
    }

    slog("samples", _data.samples);
}

void meanshift_iterate_rows()
{

    // process a batch of samples once
    for (sTensorRowIterator riter = _data.samples->begin_rows(); riter != _data.samples->end_rows(); ++riter)
    {
        pTensor sample = *riter; // 2 matrix
        sample->unsqueeze_(0);    // 1x2 matrix
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
        pTensor random_samples = _data.samples->random_sample_rows(0.2f);

        // all the above steps in two lines
        pTensor weights2 = (random_samples - sample)->pow_(2)->sum(1)->sqrt_()->gaussian_(2.5f)->unsqueeze_(1);
        pTensor all = (random_samples * weights2)->sum(0) / weights2->sum();
        //if (riter.row() == 0) slog("all", all);
        //assert(all(0, 0) == sample_new(0, 0));

        _data.samples->set_row_(riter.row(), all->squeeze_());
    }
    //slog("new_samples", _data.samples);
}


pTensor dist(pTensor& a, pTensor& b)
{
    return (a->unsqueeze(0) - b->unsqueeze(1))->set_label("weights")->pow_(2)->sum(2)->sqrt_();
}

pTensor tri(pTensor& t, const  float i)
{
    return (t->multiply_(-1.f) + i)->clamp_min_(0.0f) / i;
}

void meanshift_step()
{
    sTensor::enableAutoLog = true;
    auto start = std::chrono::high_resolution_clock::now();

    pTensor random_samples = _data.samples->random_sample_rows(0.5f);
    pTensor weights = dist(random_samples, _data.samples)->gaussian_(2.5f);
    //pTensor weights = tri(dist(random_samples, _data.samples), 6.0f);
    pTensor div = weights->sum(1)->unsqueeze(1)->set_label("div");
    pTensor new_batch = (weights->MatMult(random_samples)) / div;
    _data.samples = new_batch;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    slog("meanshift_step duration: %u ms", duration);
    sTensor::enableAutoLog = false;
}
