

#include "stensor.h"

const uint numCentroids = 5;
const uint numSample = 200;
const float spread = 5.f;

struct meandata
{
    sTensor centroids = sTensor::Dims(numCentroids, uint(2));
    sTensor samples = sTensor::Dims(0,2);
};

meandata _data;

void meanshift_init()
{
    _data.centroids = sTensor::Randoms(numCentroids, uint(2)).multiply_(70.f).add_(-35.f);
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
}
void meanshift_step()
{
    for (uint i = 0; i < 4; i++)
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

            // all the above steps in two lines
            sTensor weights2 = (_data.samples - sample).pow_(2).sum_columns().sqrt_().gaussian_(2.5f);
            sTensor all = (_data.samples * weights2).sum_rows() / weights2.sum();
            //if (riter.row() == 0) slog("all", all);
            //assert(all(0, 0) == sample_new(0, 0));

            _data.samples.set_row_(riter.row(), all.squeeze_());
        }
        slog("new_samples", _data.samples);
    }
}
