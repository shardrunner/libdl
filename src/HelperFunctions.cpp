#include "HelperFunctions.h"

#include "spdlog/async.h"
#include "spdlog/sinks/stdout_sinks.h" // or "../stdout_sinks.h" if no colors needed
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/spdlog.h"

#include <iostream>
#include <memory>

std::string HelperFunctions::toString(const Eigen::MatrixXf &mat) {
    std::stringstream ss;
    ss << mat;
    return ss.str();
}

std::string
HelperFunctions::get_representation(const Eigen::MatrixXf &input, int img_height, int img_width, int num_channels) {
    std::stringstream ss;
    ss << "Matrix of size: " << input.size() << "; Number of samples: " << input.cols() << "; Images of height: "
       << img_height << ", width: " << img_width << ", channels: " << num_channels << std::endl;
    for (long i = 0; i < input.cols(); i++) {
        ss << "\nSample: " << i << std::endl;
        for (int j = 0; j < num_channels; j++) {
            ss << "Channel: " << j << std::endl;
            for (int k = 0; k < img_height; k++) {
                for (int l = 0; l < img_width; l++) {
                    ss << input(k + l * img_height + img_height * img_width * j, i) << " ";
                }
                ss << std::endl;
            }
        }
    }
    return ss.str();
}

std::string HelperFunctions::get_comma_representation(const Eigen::MatrixXf &input) {
    std::stringstream ss;
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", " << ", ";");
    ss << input.format(CommaInitFmt) << std::endl;
    return ss.str();
}

std::string HelperFunctions::get_representation_first(const Eigen::MatrixXf &input, int img_height, int img_width,
                                                      int num_channels) {
    std::stringstream ss;
    ss << "Matrix of size: " << input.size() << "; Number of samples: " << input.cols() << "; Images of height: "
       << img_height << ", width: " << img_width << ", channels: " << num_channels << std::endl;

    for (int j = 0; j < num_channels; j++) {
        ss << "Channel: " << j << std::endl;
        for (int k = 0; k < img_height; k++) {
            for (int l = 0; l < img_width; l++) {
                ss << input(k + l * img_height + img_height * img_width * j, 0) << " ";
            }
            ss << std::endl;
        }
    }
    return ss.str();
}

//from https://github.com/gabime/spdlog/wiki/1.-QuickStart
void HelperFunctions::initLoggers() {
    if (!spdlog::get("convlayer")) {
        try {
            spdlog::init_thread_pool(8192, 1);
            std::vector<spdlog::sink_ptr> sinks;
            auto console_sink = std::make_shared<spdlog::sinks::stdout_sink_mt>();
            console_sink->set_level(spdlog::level::warn);
            console_sink->set_pattern("[%H:%M:%S.%e] [%n] [%l] %v");
            //console_sink->set_pattern("[%^%l%$] %v");

            auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("MlibLogs.txt", true);
            file_sink->set_level(spdlog::level::trace);
            file_sink->set_pattern("[%H:%M:%S.%e] [%n] [%l] %v");

            sinks.push_back(std::move(console_sink));
            sinks.push_back(std::move(file_sink));

            auto nn_logger = std::make_shared<spdlog::async_logger>("nn", begin(sinks), end(sinks),
                                                                    spdlog::thread_pool(),
                                                                    spdlog::async_overflow_policy::block);
            auto convlayer_logger = std::make_shared<spdlog::async_logger>("convlayer", begin(sinks), end(sinks),
                                                                           spdlog::thread_pool(),
                                                                           spdlog::async_overflow_policy::block);

            convlayer_logger->set_level(spdlog::level::debug);
            nn_logger->set_level(spdlog::level::debug);


            spdlog::register_logger(convlayer_logger);
            spdlog::register_logger(nn_logger);
        }
        catch (const spdlog::spdlog_ex &ex) {
            std::cout << "Log initialization failed: " << ex.what() << std::endl;
        }
    }
}