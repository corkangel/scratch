#include "minst.h"

#include <fstream>

// constant to covert from 255 to float in 0-to-1 range
const float convert255 = float(1) / float(255);

sTensor minstLoadImages(const char* filename, const uint numImages, const uint imageArraySize)
{
    std::ifstream input(filename, std::ios::binary);
    std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(input), {});

    const uint headerSize = 16;
    //assert(buffer.size() == numImages * g_imageArraySize + headerSize);

    sTensor images = sTensor::Dims(numImages, imageArraySize);

    // read image bytes into the images tensor
    for (uint i = 0; i < numImages; i++)
    {
        unsigned char* imagePtr = &buffer[headerSize + i * imageArraySize];
        for (uint j = 0; j < imageArraySize; j++)
        {
            images.set2d(i, j, float(imagePtr[j + 1]) * convert255);
        }
    }
    return images;
}

sTensor minstLoadLabels(const char* filename, const uint numImages)
{
    std::ifstream input(filename, std::ios::binary);
    std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(input), {});

    const uint headerSize = 8;
    //assert(buffer.size() == numImages + headerSize);

    sTensor categories = sTensor::Zeros(numImages, uint(1));

    // read label bytes into the categories tensor
    for (uint i = 0; i < numImages; i++)
    {
        unsigned char* labelPtr = &buffer[headerSize + i];

        // hot encoded, so use value as index
        //categories.set2d(i, labelPtr[0], 1.0f);

        // raw
        categories.set2d(i, 0, labelPtr[0]);
    }
    return categories;
}
