#pragma once

#include "spdlog/sinks/stdout_sinks.h" // or "../stdout_sinks.h" if no colors needed
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/spdlog.h"

class ManageLoggers {
public:
    void initLoggers() const;
};
