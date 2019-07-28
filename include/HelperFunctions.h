#pragma once

#include <Eigen/Core>
#include <string>

namespace HelperFunctions {
    std::string toString(const Eigen::MatrixXf& mat);

    /**
     * Print the custom tensor format as split matrices.
     * @param input The tensor to print.
     */
    std::string print_tensor(const Eigen::MatrixXf &input, int img_height, int img_width, int num_channels);

    /**
     * Print the custom tensor format so that it can used for the eigen comma initialization.
     * @param input
     * @param img_height
     * @param img_width
     * @param num_channels
     * @return
     */
    std::string print_tensor_comma(const Eigen::MatrixXf &input);

    /**
     * Print the first matrix/sample of the custom tensor format.
     * @param input The tensor to print.
     */
    std::string print_tensor_first(const Eigen::MatrixXf &input, int img_height, int img_width, int num_channels);
}

