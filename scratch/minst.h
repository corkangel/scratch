#pragma once

#include "stensor.h"

sTensor minstLoadImages(const char* filename, const uint numImages, const uint imageArraySize);
sTensor minstLoadLabels(const char* filename, const uint numImages);
