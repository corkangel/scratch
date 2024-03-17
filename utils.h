#pragma once

#include <vector>
#include <cassert>

constexpr float PI = 3.14159265359f;

using uint = unsigned int;

inline float gaussian(float x, float bw)
{
    return float(exp(-0.5f* float(pow(x/bw, 2)) / (bw*sqrt(2 * PI))));
}