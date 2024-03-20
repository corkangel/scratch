#pragma once

#include <vector>
#include <cassert>

constexpr float PI = 3.14159265359f;


using uint = unsigned int;
constexpr uint max_tensor_dimensions = 4;

inline float gaussian(float x, float bw)
{
    return float(exp(-0.5f* float(pow(x/bw, 2)) / (bw*sqrt(2 * PI))));
}