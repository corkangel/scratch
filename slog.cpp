#include "slog.h"


#include <sstream>
#include <vector>
#include <string>

std::vector<std::string> _logs;

void slog(const std::string& msg)
{
    _logs.push_back(msg);
}

void slog(const std::stringstream& ss)
{
    _logs.push_back(ss.str());
}

void slog(const char* format, ...)
{
    va_list args;
    va_start(args, format);

    // We use vsnprintf to count the number of characters for our format string
    int size = std::vsnprintf(nullptr, 0, format, args);
    std::string str(size, '\0');

    // We have to call va_start again before vsnprintf
    va_start(args, format);
    std::vsnprintf(&str[0], size + 1, format, args);

    va_end(args);

    _logs.push_back(str);
}

const std::vector<std::string>& get_logs()
{
    return _logs;
}