#pragma once

#include <iostream>
#include <sstream>
#include <cstdarg>

#include "utils.h"

void slog(const std::string& msg);
void slog(const std::stringstream& ss);
void slog(const char* format, ...);
const std::vector<std::string>& get_logs();

