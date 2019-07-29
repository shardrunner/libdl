#include <catch2/catch.hpp>

#include "ActivationFunction/IdentityFunction.h"
#include "Layer/ConvolutionalLayer.h"
#include "RandomInitialization/DeterministicInitialization.h"
#include "HelperFunctions.h"

#include <iostream>

SCENARIO("Test Convolutional Layer") {
    GIVEN("Custom Matrices") {
        WHEN("The im2col function is applied") {
            THEN("It works for a one channel one sample input matrix") {
                ConvolutionalLayer conv_layer(
                        3, 3, 1, 2, 2, 1, 1, 0, std::make_unique<IdentityFunction>(),
                        std::make_unique<DeterministicInitialization>());
                Eigen::MatrixXf input_matrix(9, 1);
                input_matrix << 1, 4, 7, 2, 5, 8, 3, 6, 9;
                auto im2col_matrix = conv_layer.im2col(input_matrix, 3, 3, 1, 2, 2, 1, 0);
                Eigen::MatrixXf output_matrix(4, 4);
                output_matrix << 1, 4, 2, 5,
                        4, 7, 5, 8,
                        2, 5, 3, 6,
                        5, 8, 6, 9;
                REQUIRE(im2col_matrix->isApprox(output_matrix));
            }THEN("It works for a multi channel one sample input matrix") {
                ConvolutionalLayer conv_layer(
                        3, 3, 2, 2, 2, 1, 1, 0, std::make_unique<IdentityFunction>(),
                        std::make_unique<DeterministicInitialization>());
                Eigen::MatrixXf input_matrix(18, 1);
                input_matrix << 1, 4, 7, 2, 5, 8, 3, 6, 9, 10, 13, 16, 11, 14, 17, 12, 15, 18;
                auto im2col_matrix = conv_layer.im2col(input_matrix, 3, 3, 2, 2, 2, 1, 0);
                Eigen::MatrixXf output_matrix(8, 4);
                output_matrix << 1, 4, 2, 5,
                        4, 7, 5, 8,
                        2, 5, 3, 6,
                        5, 8, 6, 9,
                        10, 13, 11, 14,
                        13, 16, 14, 17,
                        11, 14, 12, 15,
                        14, 17, 15, 18;
                REQUIRE(im2col_matrix->isApprox(output_matrix));
            }THEN("It works for a one channel two sample input matrix") {
                ConvolutionalLayer conv_layer(
                        3, 3, 1, 2, 2, 1, 1, 0, std::make_unique<IdentityFunction>(),
                        std::make_unique<DeterministicInitialization>());
                Eigen::MatrixXf input_matrix(2, 9);
                input_matrix << 1, 4, 7, 2, 5, 8, 3, 6, 9, 10, 13, 16, 11, 14, 17, 12, 15,
                        18;
                input_matrix.transposeInPlace();
                auto im2col_matrix = conv_layer.im2col(input_matrix, 3, 3, 1, 2, 2, 1, 0);
                Eigen::MatrixXf output_matrix(4, 8);
                output_matrix << 1, 4, 2, 5, 10, 13, 11, 14, 4, 7, 5, 8, 13, 16, 14, 17,
                        2, 5, 3, 6, 11, 14, 12, 15, 5, 8, 6, 9, 14, 17, 15, 18;
                REQUIRE(im2col_matrix->isApprox(output_matrix));
            }THEN("It works for a one channel one sample not matching stride two input "
                  "matrix") {
                ConvolutionalLayer conv_layer(
                        3, 3, 1, 2, 2, 1, 2, 0, std::make_unique<IdentityFunction>(),
                        std::make_unique<DeterministicInitialization>());
                Eigen::MatrixXf input_matrix(9, 1);
                input_matrix << 1, 4, 7, 2, 5, 8, 3, 6, 9;
                auto im2col_matrix = conv_layer.im2col(input_matrix, 3, 3, 1, 2, 2, 2, 0);
                Eigen::MatrixXf output_matrix(4, 1);
                output_matrix << 1, 4, 2, 5;
                REQUIRE(im2col_matrix->isApprox(output_matrix));
            }THEN("It works for a one channel one sample matching stride two input "
                  "matrix") {
                ConvolutionalLayer conv_layer(
                        4, 3, 1, 2, 2, 1, 2, 0, std::make_unique<IdentityFunction>(),
                        std::make_unique<DeterministicInitialization>());
                Eigen::MatrixXf input_matrix(12, 1);
                input_matrix << 1, 4, 7, 2, 5, 8, 3, 6, 9, 10, 11, 12;
                auto im2col_matrix = conv_layer.im2col(input_matrix, 4, 3, 1, 2, 2, 2, 0);
                Eigen::MatrixXf output_matrix(4, 2);
                output_matrix << 1, 7, 4, 2, 5, 3, 8, 6;
                REQUIRE(im2col_matrix->isApprox(output_matrix));
            }
        }WHEN("Im2col is used on a filter") {
            THEN("The resulting matrix is correct") {
                ConvolutionalLayer conv_layer(
                        3, 3, 1, 2, 2, 1, 1, 0, std::make_unique<IdentityFunction>(),
                        std::make_unique<DeterministicInitialization>());
                Eigen::MatrixXf input_matrix(2, 12);
                input_matrix << 0, 1, -1, 0, 5, 3, 4, 2, 16, 68, 24, -2,
                        60, 32, 22, 18, 35, 7, 46, 23, 78, 20, 81, 42;
                input_matrix.transposeInPlace();
                auto im2col_matrix = conv_layer.im2col(input_matrix, 2, 2, 3, 2, 2, 1, 0);
                REQUIRE(im2col_matrix->isApprox(input_matrix));
            }
        }WHEN("The im2col matrix is reshaped") {
            THEN("The reshaping is correct for a simple matrix") {
                ConvolutionalLayer conv_layer(
                        3, 3, 1, 2, 2, 1, 1, 0, std::make_unique<IdentityFunction>(),
                        std::make_unique<DeterministicInitialization>());
                Eigen::MatrixXf input_matrix(9, 1);
                input_matrix << -2, -0.5, 1, -1.5, 0, 1.5, -1.0, 0.5, 2;
                Eigen::MatrixXf filter(1, 4);
                filter << -0.5, 0.5, 0, 1;
                auto im2col_matrix = conv_layer.im2col(input_matrix, 3, 3, 1, 2, 2, 1, 0);
                auto im2col_reshaped = conv_layer.reshape_im2col_result((filter * (*im2col_matrix)).transpose(), 3, 3,
                                                                        2, 2, 1, 1, 0,
                                                                        input_matrix.cols());
                Eigen::MatrixXf output_matrix(4, 1);
                output_matrix << 0.75, 2.25, 1.25, 2.75;
                REQUIRE(im2col_reshaped->isApprox(output_matrix));
            }THEN("The reshaping is correct for a more complex matrix") {
                ConvolutionalLayer conv_layer(
                        3, 3, 1, 2, 2, 2, 2, 0, std::make_unique<IdentityFunction>(),
                        std::make_unique<DeterministicInitialization>());
                Eigen::MatrixXf input_matrix(9, 2);
                input_matrix << 1, 10, 2, 11, 3, 12, 4, 13, 5, 14, 6, 15, 7, 16, 8, 17, 9, 18;
                Eigen::MatrixXf filter(2, 4);
                filter << -1, -0.5, 0, 0.5,
                        1, 1.5, 2, 0;
                auto im2col_matrix = conv_layer.im2col(input_matrix, 3, 3, 1, 2, 2, 2, 0);
                auto im2col_reshaped = conv_layer.reshape_im2col_result((filter * (*im2col_matrix)).transpose(), 3, 3,
                                                                        2, 2, 2, 2, 0,
                                                                        input_matrix.cols());
                Eigen::MatrixXf output_matrix(2, 2);
                output_matrix << 0.5, -8.5,
                        12, 52.5;
                REQUIRE(im2col_reshaped->isApprox(output_matrix));
            }
        }WHEN("The im2col matrix is forward propagated") {
            THEN("The forward propagation is correct for a simple matrix") {
                ConvolutionalLayer conv_layer(
                        3, 3, 1, 2, 2, 1, 1, 0, std::make_unique<IdentityFunction>(),
                        std::make_unique<DeterministicInitialization>());
                Eigen::MatrixXf input_matrix(9, 1);
                input_matrix << -2, -0.5, 1, -1.5, 0, 1.5, -1.0, 0.5, 2;
                Eigen::MatrixXf filter(1, 4);
                filter << -0.5, 0.5, 0, 1;
                conv_layer.set_weights(filter);
                conv_layer.feed_forward(input_matrix);
                Eigen::MatrixXf output_matrix(4, 1);
                output_matrix << 0.75, 2.25, 1.25, 2.75;
                REQUIRE(conv_layer.get_forward_output().isApprox(output_matrix));
            }THEN("The forward propagation is correct for a more complex matrix") {
                ConvolutionalLayer conv_layer(
                        3, 3, 1, 2, 2, 2, 2, 0, std::make_unique<IdentityFunction>(),
                        std::make_unique<DeterministicInitialization>());
                Eigen::MatrixXf input_matrix(9, 2);
                input_matrix << 1, 10, 2, 11, 3, 12, 4, 13, 5, 14, 6, 15, 7, 16, 8, 17, 9, 18;
                Eigen::MatrixXf filter(2, 4);
                filter << -1, -0.5, 0, 0.5,
                        1, 1.5, 2, 0;
                conv_layer.set_weights(filter);
                conv_layer.feed_forward(input_matrix);
                Eigen::MatrixXf output_matrix(2, 2);
                output_matrix << 0.5, -8.5,
                        12, 52.5;
                REQUIRE(conv_layer.get_forward_output().isApprox(output_matrix));
            }THEN("The forward propagation is correct another more complex matrix") {
                ConvolutionalLayer conv_layer(
                        3, 3, 3, 2, 2, 2, 1, 0, std::make_unique<IdentityFunction>(),
                        std::make_unique<DeterministicInitialization>());
                Eigen::MatrixXf input_matrix(27, 1);
                input_matrix
                        << 16, 47, 68, 24, 18, 12, 32, 26, 9, 26, 24, 2, 57, 21, 11, 43, 12, 19, 18, 4, 81, 47, 6, 22, 21, 12, 13;
                HelperFunctions::print_tensor(input_matrix, 3, 3, 3);
                Eigen::MatrixXf filter(2, 12);
                filter << 0, 1, -1, 0, 5, 3, 4, 2, 16, 68, 24, -2,
                        60, 32, 22, 18, 35, 7, 46, 23, 78, 20, 81, 42;
                conv_layer.set_weights(filter);
                conv_layer.feed_forward(input_matrix);
                Eigen::MatrixXf output_matrix(8, 1);
                output_matrix << 2171, 5954, 2170, 2064, 13042, 11023, 13575, 6425;
                REQUIRE(conv_layer.get_forward_output().isApprox(output_matrix));
            }THEN("The forward propagation is correct for another matrix") {
                ConvolutionalLayer conv = ConvolutionalLayer(3, 3, 1, 2, 2, 1, 1, 0,
                                                             std::make_unique<IdentityFunction>(),
                                                             std::make_unique<DeterministicInitialization>());
                Eigen::MatrixXf input(9, 1);
                input << -2, -1.5, -1.0, -0.5, 0, 0.5, 1, 1.5, 2;
                Eigen::MatrixXf filter(1, 4);
                filter << -0.5, 0, 0.5, 1;
                conv.set_weights(filter);
                Eigen::MatrixXf result(4, 1);
                result << 0.75, 1.25, 2.25, 2.75;
                conv.feed_forward(input);
                THEN("The forward feed ist correct") {
                    REQUIRE(conv.get_forward_output().isApprox(result));
                }
            }
        }
    }GIVEN("Padding function for the convolution layer") {
        ConvolutionalLayer conv_layer(
                2, 3, 2, 1, 1, 2, 1, 0, std::make_unique<IdentityFunction>(),
                std::make_unique<DeterministicInitialization>());
        Eigen::MatrixXf input_matrix(12, 2);
        input_matrix.transpose()
                << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24;
        WHEN("Pad matrix function is applied with padding 2") {
            auto input_padded = conv_layer.pad_matrix(input_matrix, 2, 2, 3, 2);
            THEN("The matrix is correctly padded with 2") {
                Eigen::MatrixXf output_matrix(84, 2);
                output_matrix
                        << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 13, 2, 14, 0, 0, 0, 0, 0, 0, 0, 0, 3, 15, 4, 16, 0, 0, 0, 0, 0, 0, 0, 0, 5, 17, 6, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 19, 8, 20, 0, 0, 0, 0, 0, 0, 0, 0, 9, 21, 10, 22, 0, 0, 0, 0, 0, 0, 0, 0, 11, 23, 12, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

                REQUIRE(input_padded->isApprox(output_matrix));
            }
        }WHEN("Pad matrix function is applied with padding 0") {
            auto input_padded = conv_layer.pad_matrix(input_matrix, 0, 2, 3, 2);
            THEN("The matrix is correctly padded with 0") {
                REQUIRE(input_padded->isApprox(input_matrix));
            }
        }
    }GIVEN("A flip kernel function") {
        WHEN("A 2x2 filter is flipped") {
            ConvolutionalLayer conv_layer(
                    2, 3, 2, 2, 2, 2, 1, 0, std::make_unique<IdentityFunction>(),
                    std::make_unique<DeterministicInitialization>());
            Eigen::MatrixXf filter_matrix(2, 8);
            filter_matrix << 0, 1, -1, 0, 5, 3, 4, 2,
                    1, 2, 3, 4, 5, 6, 7, 8;
            conv_layer.set_weights(filter_matrix);
            auto filter_flipped = conv_layer.flip_filter();
            THEN("The flipping should be correct") {
                Eigen::MatrixXf output_matrix(2, 8);
                output_matrix << 0, -1, 1, 0, 2, 4, 3, 5, 4, 3, 2, 1, 8, 7, 6, 5;
                REQUIRE(filter_flipped->isApprox(output_matrix));

            }
        }WHEN("A 3x3 filter is flipped") {
            ConvolutionalLayer conv_layer(
                    3, 3, 2, 3, 3, 1, 1, 0, std::make_unique<IdentityFunction>(),
                    std::make_unique<DeterministicInitialization>());
            Eigen::MatrixXf filter_matrix(1, 18);
            filter_matrix << 0, 1, -1, 0, 5, 3, 4, 2, 5, 1, 2, 3, 4, 5, 6, 7, 8, 9;
            conv_layer.set_weights(filter_matrix);
            auto filter_flipped = conv_layer.flip_filter();
            THEN("The flipping should be correct") {
                Eigen::MatrixXf output_matrix(1, 18);
                output_matrix << 5, 2, 4, 3, 5, 0, -1, 1, 0, 9, 8, 7, 6, 5, 4, 3, 2, 1;
                REQUIRE(filter_flipped->isApprox(output_matrix));
            }
        }
    }GIVEN("A backpropagation function") {
        WHEN("The bias is backpropagated") {

            THEN("It should be correct for a simple input") {
                ConvolutionalLayer conv_layer(
                        4, 4, 1, 2, 2, 2, 1, 0, std::make_unique<IdentityFunction>(),
                        std::make_unique<DeterministicInitialization>());
                Eigen::MatrixXd input_matrixd(18, 1);
                input_matrixd.transpose()
                        << 0.5, 0.2, 0.7, 0.3, 0.65, 0.23, 0.13, 0.18, 0.42, 0.75, 0.08, 0.21, 0.12, 0.04, 0.24, 0.68, 0.15, 0.34;
                Eigen::MatrixXf input_matrix = input_matrixd.cast<float>();
                conv_layer.backpropagate_bias(input_matrix);
                Eigen::VectorXf output_matrix(2);
                output_matrix << (float) 3.31, (float) 2.61;
                REQUIRE(conv_layer.get_bias_derivative().isApprox(output_matrix));
            }THEN("It should be correct for a multi-sample input") {
                ConvolutionalLayer conv_layer(
                        4, 4, 1, 3, 3, 1, 1, 0, std::make_unique<IdentityFunction>(),
                        std::make_unique<DeterministicInitialization>());
                Eigen::MatrixXf input_matrix(4, 2);
                input_matrix.transpose() << 1, 2, 3, 4, 5, 6, 7, 8;
                conv_layer.backpropagate_bias(input_matrix);
                REQUIRE(conv_layer.get_bias_derivative()(0) == 18);
            }
        }WHEN("The weights are backpropagated") {
            THEN("It should be correct for a simple input") {
                ConvolutionalLayer conv_layer(
                        3, 3, 1, 2, 2, 2, 1, 0, std::make_unique<IdentityFunction>(),
                        std::make_unique<DeterministicInitialization>());
                Eigen::MatrixXf input_matrix(9, 1);
                input_matrix.transpose() << 16, 47, 68, 24, 18, 12, 32, 26, 9;
                Eigen::MatrixXd der_matrix(8, 1);
                der_matrix.transpose() << 0, -0.0000294504954, 0, 0, 0, 0.00000639539432, 0, 0;
                Eigen::MatrixXf de_matrix = der_matrix.cast<float>();

                conv_layer.backpropagate_weights(input_matrix, de_matrix);

                Eigen::MatrixXd output_matrixd(2, 4);
                output_matrixd << -0.00138417, -0.00200263, -0.000530109, -0.000353406,
                        0.000300584, 0.000434887, 0.000115117, 7.67447e-05;
                Eigen::MatrixXf output_matrix = output_matrixd.cast<float>();
                //std::cout << "dw\n" << conv_layer.get_weights_derivative() << "\npredicte\n" << output_matrix << std::endl;

                REQUIRE(conv_layer.get_weights_derivative().isApprox(output_matrix));
            }THEN("It should be correct for a more complex input") {
                //std::cout << "\n\n-----------------------------------------------------------------------------------------------------------\n\n"<< std::endl;
                ConvolutionalLayer conv_layer(
                        3, 3, 3, 2, 2, 2, 1, 0, std::make_unique<IdentityFunction>(),
                        std::make_unique<DeterministicInitialization>());
                Eigen::MatrixXf input_matrix(27, 1);
                input_matrix
                        << 16, 47, 68, 24, 18, 12, 32, 26, 9, 26, 24, 2, 57, 21, 11, 43, 12, 19, 18, 4, 81, 47, 6, 22, 21, 12, 13;
                //std::cout << "Filter\n"<<HelperFunctions::print_tensor(filter_matrix.transpose(),2,2,3);
                Eigen::MatrixXd der_matrix(8, 1);
                der_matrix.transpose() << 0.1678, 0.002, 0.098, 0.246, 0.5, 0.21, 0.67, 0.487;
                Eigen::MatrixXf de_matrix = der_matrix.cast<float>();
                //std::cout << "Derivative\n"<<HelperFunctions::print_tensor(de_matrix,2,2,2);

                conv_layer.backpropagate_weights(input_matrix, de_matrix);

                Eigen::MatrixXd output_matrixd(2, 12);
                output_matrixd
                        << 9.5588, 12.7386, 13.5952, 7.8064, 15.1628, 8.7952, 16.7726, 9.3958, 9.1104, 6.8332, 12.9086, 5.4248, 42.716, 55.684, 49.882, 33.323, 66.457, 31.847, 67.564, 30.103, 44.252, 33.744, 44.674, 21.991;

                Eigen::MatrixXf output_matrix = output_matrixd.cast<float>();
                REQUIRE(conv_layer.get_weights_derivative().isApprox(output_matrix));
            }
        }WHEN("The inputs are backpropagated") {
            THEN("It should be correct for a simple input") {
                ConvolutionalLayer conv_layer(
                        3, 3, 1, 2, 2, 2, 1, 0, std::make_unique<IdentityFunction>(),
                        std::make_unique<DeterministicInitialization>());
                Eigen::MatrixXf filter_matrix(2, 4);
                filter_matrix << 0, 1, -1, 0, 5, 3, 4, 2;
                Eigen::MatrixXd der_matrix(8, 1);
                der_matrix.transpose() << 0, -0.0000294504954, 0, 0, 0, 0.00000639539432, 0, 0;
                Eigen::MatrixXf de_matrix = der_matrix.cast<float>();

                conv_layer.set_weights(filter_matrix);

                conv_layer.backpropagate_input(de_matrix);

                Eigen::MatrixXd output_matrixd(9, 1);
                output_matrixd.transpose()
                        << 0, 3.1977e-05, -1.02643e-05, 0, 5.50321e-05, 1.27908e-05, 0, 0, 0;
                Eigen::MatrixXf output_matrix = output_matrixd.cast<float>();
                REQUIRE(conv_layer.get_input_derivative().isApprox(output_matrix));
            }THEN("It should be correct for a more complex input") {
                //std::cout << "\n\n-----------------------------------------------------------------------------------------------------------\n\n"<< std::endl;
                ConvolutionalLayer conv_layer(
                        3, 3, 3, 2, 2, 2, 1, 0, std::make_unique<IdentityFunction>(),
                        std::make_unique<DeterministicInitialization>());
                Eigen::MatrixXf filter_matrix(2, 12);
                filter_matrix << 0, -1, 1, 0, 2, 4, 3, 5, -2, 24, 68, 16, 18, 22, 32, 60, 23, 46, 7, 35, 42, 81, 20, 78;
                conv_layer.set_weights(filter_matrix);
                filter_matrix = *(conv_layer.flip_filter());
                //std::cout << "Filter\n"<<HelperFunctions::print_tensor(filter_matrix.transpose(),2,2,3);
                Eigen::MatrixXd der_matrix(8, 1);
                der_matrix.transpose() << 0.1678, 0.002, 0.098, 0.246, 0.5, 0.21, 0.67, 0.487;
                Eigen::MatrixXf de_matrix = der_matrix.cast<float>();
                //std::cout << "Derivative\n"<<HelperFunctions::print_tensor(de_matrix,2,2,2);

                conv_layer.set_weights(filter_matrix);

                conv_layer.backpropagate_input(de_matrix);

                Eigen::MatrixXd output_matrixd(27, 1);
                output_matrixd.transpose()
                        << 30, 28.7678, 6.722, 51.0322, 64.376, 19.61, 14.642, 22.528, 8.766, 18.339, 11.3634, 1.476, 47.6112, 44.7626, 8.981, 31.212, 38.992, 11.693, 41.6848, 37.8224, 4.336, 98.3552, 99.7084, 35.284, 56.622, 73.295, 19.962;
                Eigen::MatrixXf output_matrix = output_matrixd.cast<float>();
                REQUIRE(conv_layer.get_input_derivative().isApprox(output_matrix));
            }
        }WHEN("A complete backpropagation is used") {
            ConvolutionalLayer conv = ConvolutionalLayer(3, 3, 1, 2, 2, 1, 1, 0, std::make_unique<IdentityFunction>(),
                                                         std::make_unique<DeterministicInitialization>());
            Eigen::MatrixXf input(9, 1);
            input << -2, -1.5, -1.0, -0.5, 0, 0.5, 1, 1.5, 2;
            Eigen::MatrixXf filter(1, 4);
            filter << -0.5, 0, 0.5, 1;
            conv.set_weights(filter);
            conv.feed_forward(input);
            Eigen::MatrixXf dC_da(4, 1);
            dC_da << 0, -1, 1.5, 2;
            conv.backpropagation(input, dC_da);
            THEN("The components should be correctly backpropagated") {
                Eigen::MatrixXf res_dC_dw(1, 4);
                res_dC_dw << 0.75, 2, 4.5, 5.75;
                REQUIRE(conv.get_weights_derivative().isApprox(res_dC_dw));
                Eigen::MatrixXf res_dC_da_prev(9, 1);
                res_dC_da_prev << 0, 0.5, 0, -0.75, -1.5, -1, 0.75, 2.5, 2;
                REQUIRE(conv.get_input_derivative().isApprox(res_dC_da_prev));
                Eigen::VectorXf res_dC_db(1);
                res_dC_db << 2.5;
                REQUIRE(conv.get_bias_derivative().isApprox(res_dC_db));
            }
        }

    }
    GIVEN("A stride greater than 1") {
        ConvolutionalLayer conv = ConvolutionalLayer(4, 4, 1, 2, 2, 1, 2, 0, std::make_unique<IdentityFunction>(),
                                                     std::make_unique<DeterministicInitialization>());
        Eigen::MatrixXf input_matrix(16, 1);
        input_matrix << 1, 4, 7, 2, 5, 8, 3, 6, 9,0,-3,2,0,0,1,1;
        WHEN("Im2col is applied") {
            auto im2col_matrix = conv.im2col(input_matrix, 4, 4, 1, 2, 2, 2, 0);
            Eigen::MatrixXf output_matrix(4, 4);
            output_matrix << 1, 7, 9, -3, 4, 2, 0, 2, 5, 3, 0, 1, 8, 6, 0, 1;
            THEN("It should work") {
                REQUIRE(im2col_matrix->isApprox(output_matrix));
            }
        }
    }
}