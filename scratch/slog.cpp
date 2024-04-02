#include "slog.h"


#include <sstream>
#include <vector>
#include <string>
#include <iomanip>

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

void slog(const sTensorInfo& info)
{
    std::stringstream ss;
    ss << info;
    slog(ss);
}

// ------------------------------- tensor info logging -------------------------------

std::vector<sTensorInfo> _tensor_infos;

void log_tensor_info(const sTensorInfo& info)
{
    _tensor_infos.push_back(info);
}

const std::vector<sTensorInfo>& get_tensor_infos()
{
    return _tensor_infos;
}

void clear_tensor_infos()
{
    _tensor_infos.clear();
}

std::ostream& operator<<(std::ostream& os, const sTensorInfo& m)
{
    os << "sTensorInfo { id: " << m.id << ", dims: [";
    for (uint i = 0; i < m.rank; ++i)
    {
        os << m.dimensions[i];
        if (i != m.rank - 1) os << ", ";
    }
    os << "], label: " << (m.label ? m.label : "_") << ", op: " << (m.operation ? m.operation : "_") << ",front: [";
    for (uint i = 0; i < std::min(sInfoDataSize, m.dimensions[0]); i++)
    {
        os << std::fixed << std::setprecision(2) << m.data_front[i];
        if (i != m.dimensions[0] - 1) os << ", ";
    }
    os << "], back: [";
    for (uint i = 0; i < std::min(sInfoDataSize, m.dimensions[0]); i++)
    {
        os << std::fixed << std::setprecision(2) << m.data_back[sInfoDataSize - i - 1];
        if (i != m.dimensions[0] - 1) os << ", ";
    }
    os << "]\n";
    return os;
}