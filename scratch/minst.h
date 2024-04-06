#pragma once

#include "stensor.h"

pTensor minstLoadImages(const char* filename, const uint numImages, const uint imageArraySize);
pTensor minstLoadLabels(const char* filename, const uint numImages);
