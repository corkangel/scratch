#pragma once

#include <iostream>
#include <sstream>
#include <cstdarg>
#include <vector>

#include "utils.h"

constexpr uint sInfoDataSize = 8;


void slog(const std::string& msg);
void slog(const std::stringstream& ss);
void slog(const char* format, ...);
const std::vector<std::string>& get_logs();



struct sTensorInfo
{
    uint id;
    uint rank;
    uint dimensions[max_tensor_dimensions];
    const char* label;
    const char* operation;
    float data_front[sInfoDataSize];
    float data_back[sInfoDataSize];
    long long time;
};

std::ostream& operator<<(std::ostream& os, const sTensorInfo& m);

void log_tensor_info(const sTensorInfo& info);
const std::vector<sTensorInfo>& get_tensor_infos();
void clear_tensor_infos();


void slog(const sTensorInfo& info);