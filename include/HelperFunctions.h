#pragma once

#include <Eigen/Core>
#include <string>

/**
 * A helper class to provide various useful helper methods.
 *
 * Currently implements methods to print matrices and to manage loggers.
 */
namespace HelperFunctions {
/**
 * Allows spdlog to print Eigen matrices.
 * @param mat The matrix to print.
 * @return A string representation of the matrix.
 */
std::string to_string(const Eigen::MatrixXf &mat);

/**
 * Print the custom tensor format as split matrices.
 * @param input The matrix to print.
 * @param img_height Embedded image height.
 * @param img_width Embedded image width.
 * @param num_channels Number of Channels.
 * @return The string representation.
 */
std::string get_representation(const Eigen::MatrixXf &input, int img_height,
                               int img_width, int num_channels);

/**
 * Print the custom tensor format so that it can used for the Eigen comma
 * initialization.
 * @param input The input matrix to print as comma representation.
 * @return The string representation of the matrix
 */
std::string get_comma_representation(const Eigen::MatrixXf &input);

/**
 * Print the first matrix/sample of the custom tensor format.
 * @param input The tensor to print.
 * @param img_height Embedded image height.
 * @param img_width Embedded image width.
 * @param num_channels Number of Channels.
 * @return The string representation.
 */
std::string get_representation_first(const Eigen::MatrixXf &input,
                                     int img_height, int img_width,
                                     int num_channels);

/**
 * Initializes and sets up the logger.
 */
void init_loggers();
} // namespace HelperFunctions
