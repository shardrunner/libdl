#include "ManageLoggers.h"

#include <iostream>
#include <memory>

//from https://github.com/gabime/spdlog/wiki/1.-QuickStart
void ManageLoggers::initLoggers() const {
    try {
        std::vector<spdlog::sink_ptr> sinks;
        auto console_sink = std::make_shared<spdlog::sinks::stdout_sink_st>();
        console_sink->set_level(spdlog::level::warn);
        console_sink->set_pattern("[%H:%M:%S.%e] [%n] [%l] %v");
        //console_sink->set_pattern("[%^%l%$] %v");

        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_st>("MlibLogs.txt", true);
        file_sink->set_level(spdlog::level::trace);
        file_sink->set_pattern("[%H:%M:%S.%e] [%n] [%l] %v");

        sinks.push_back(std::move(console_sink));
        sinks.push_back(std::move(file_sink));

        auto nn_logger=std::make_shared<spdlog::logger>("nn", begin(sinks), end(sinks));
        auto convlayer_logger=std::make_shared<spdlog::logger>("convlayer", begin(sinks), end(sinks));
        //auto net_logger = std::make_shared<spdlog::logger>("net", file_sink);
        //spdlog::logger logger("multi_sink", {console_sink, file_sink});
        convlayer_logger->set_level(spdlog::level::debug);
        nn_logger->set_level(spdlog::level::debug);

        if (!spdlog::get("convlayer")) {
            spdlog::register_logger(convlayer_logger);
        }
        if (!spdlog::get("nn")) {
            spdlog::register_logger(nn_logger);
        }

        // or you can even set multi_sink logger as default logger
        //spdlog::set_default_logger(
        //        std::make_shared<spdlog::logger>("multi_sink", spdlog::sinks_init_list({console_sink, file_sink})));
    }
    catch (const spdlog::spdlog_ex &ex) {
        std::cout << "Log initialization failed: " << ex.what() << std::endl;
    }
}