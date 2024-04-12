#pragma once

#include <vector>
#include <cassert>
#include <chrono>

constexpr float PI = 3.14159265359f;


using uint = unsigned int;
constexpr uint max_tensor_dimensions = 4;

using timepoint = std::chrono::steady_clock::time_point;

inline float gaussian(float x, float bw)
{
    return float(exp(-0.5f* float(pow(x/bw, 2)) / (bw*sqrt(2 * PI))));
}

