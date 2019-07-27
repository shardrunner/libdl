#include <catch2/catch.hpp>

#include "ActivationFunction/IdentityFunction.h"
#include "Layer/ConvolutionalLayer.h"
#include "ManageLoggers.h"
#include "RandomInitialization/DeterministicInitialization.h"
#include "HelperFunctions.h"
#include <thread>
#include "omp.h"

#include <iostream>

SCENARIO("Test Convolutional Layer") {
/*    //omp_set_num_threads(2);
    std::cout << "C++ Test: " << std::thread::hardware_concurrency() << std::endl;
    std::cout << "OpenMP setting:" << omp_get_num_procs() << std::endl;

    //int num_threads =4;
    //omp_set_num_threads ( num_threads );
# pragma omp parallel for
    for (int i = 0; i < 8 ; i++)
    {
# pragma omp critical
        std :: cout << "My id is: "
                    << omp_get_thread_num () << std :: endl;
    }*/
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
            }
            THEN("It works for a multi channel one sample input matrix") {
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
            }
            THEN("It works for a one channel two sample input matrix") {
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
            }
            THEN("It works for a one channel one sample not matching stride two input "
                 "matrix") {
                ConvolutionalLayer conv_layer(
                        3, 3, 1, 2, 2, 1, 2, 0, std::make_unique<IdentityFunction>(),
                        std::make_unique<DeterministicInitialization>());
                Eigen::MatrixXf input_matrix(9, 1);
                input_matrix << 1, 4, 7, 2, 5, 8, 3, 6, 9;
                auto im2col_matrix = conv_layer.im2col(input_matrix, 3, 3, 1, 2, 2, 2, 0);
                Eigen::MatrixXf output_matrix(4, 1);
                // std::cout << "output m:\n" << *im2col_input;
                output_matrix << 1, 4, 2, 5;
                REQUIRE(im2col_matrix->isApprox(output_matrix));
            }
            THEN("It works for a one channel one sample matching stride two input "
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
        }
        WHEN("Im2col is used on a filter") {
            THEN("The resulting matrix is correct") {
                ConvolutionalLayer conv_layer(
                        3, 3, 1, 2, 2, 1, 1, 0, std::make_unique<IdentityFunction>(),
                        std::make_unique<DeterministicInitialization>());
                Eigen::MatrixXf input_matrix(2, 12);
                input_matrix << 0,1,-1,0,5,3,4,2,16,68,24,-2,
                                60,32,22,18,35,7,46,23,78,20,81,42;
                input_matrix.transposeInPlace();
                //std::cout << "Res1. \n" << input_matrix << std::endl;
                auto im2col_matrix = conv_layer.im2col(input_matrix, 2, 2, 3, 2, 2, 1, 0);
                //std::cout << "Res. \n" << *im2col_matrix << std::endl;
                REQUIRE(im2col_matrix->isApprox(input_matrix));
            }
        }
        WHEN("The im2col matrix is reshaped") {
            THEN("The reshaping is correct for a simple matrix") {
                ConvolutionalLayer conv_layer(
                        3, 3, 1, 2, 2, 1, 1, 0, std::make_unique<IdentityFunction>(),
                        std::make_unique<DeterministicInitialization>());
                Eigen::MatrixXf input_matrix(9, 1);
                input_matrix << -2, -0.5, 1, -1.5, 0, 1.5, -1.0, 0.5, 2;
                Eigen::MatrixXf filter(1, 4);
                filter << -0.5, 0.5, 0, 1;
                auto im2col_matrix = conv_layer.im2col(input_matrix, 3, 3, 1, 2, 2, 1, 0);
                auto im2col_reshaped = conv_layer.reshape_im2col_result(filter * (*im2col_matrix), 3, 3, 2, 2, 1, 1, 0,
                                                                        input_matrix.cols());
                //std::cout << HelperFunctions::print_tensor(conv_layer.get_m_z(), 4,1,1);
                Eigen::MatrixXf output_matrix(4, 1);
                output_matrix << 0.75, 2.25, 1.25, 2.75;
                REQUIRE(im2col_reshaped->isApprox(output_matrix));
            }
            THEN("The reshaping is correct for a more complex matrix") {
                ConvolutionalLayer conv_layer(
                        3, 3, 1, 2, 2, 2, 2, 0, std::make_unique<IdentityFunction>(),
                        std::make_unique<DeterministicInitialization>());
                Eigen::MatrixXf input_matrix(9, 2);
                input_matrix << 1, 10, 2, 11, 3, 12, 4, 13, 5, 14, 6, 15, 7, 16, 8, 17, 9, 18;
                //std::cout << "Input: \n" << HelperFunctions::print_tensor(input_matrix,3,3,1) << std::endl;
                //input_matrix.transposeInPlace();
                Eigen::MatrixXf filter(2, 4);
                filter << -1, -0.5, 0, 0.5,
                        1, 1.5, 2, 0;
                auto im2col_matrix = conv_layer.im2col(input_matrix, 3, 3, 1, 2, 2, 2, 0);
                auto im2col_reshaped = conv_layer.reshape_im2col_result(filter * (*im2col_matrix), 3, 3, 2, 2, 2, 2, 0,
                                                                        input_matrix.cols());
                //std::cout << HelperFunctions::print_tensor(conv_layer.get_m_z(), 1,1,2);
                Eigen::MatrixXf output_matrix(2, 2);
                output_matrix << 0.5, -8.5,
                        12, 52.5;
                REQUIRE(im2col_reshaped->isApprox(output_matrix));
            }
        }
        WHEN("The im2col matrix is forward propagated") {
            THEN("The forward propagation is correct for a simple matrix") {
                ConvolutionalLayer conv_layer(
                        3, 3, 1, 2, 2, 1, 1, 0, std::make_unique<IdentityFunction>(),
                        std::make_unique<DeterministicInitialization>());
                Eigen::MatrixXf input_matrix(9, 1);
                input_matrix << -2, -0.5, 1, -1.5, 0, 1.5, -1.0, 0.5, 2;
                Eigen::MatrixXf filter(1, 4);
                filter << -0.5, 0.5, 0, 1;
                conv_layer.set_filter(filter);
                conv_layer.feed_forward(input_matrix);
                //std::cout << HelperFunctions::print_tensor(conv_layer.get_m_z(), 4,1,1);
                Eigen::MatrixXf output_matrix(4, 1);
                output_matrix << 0.75, 2.25, 1.25, 2.75;
                REQUIRE(conv_layer.get_forward_output().isApprox(output_matrix));
            }
            THEN("The forward propagation is correct for a more complex matrix") {
                ConvolutionalLayer conv_layer(
                        3, 3, 1, 2, 2, 2, 2, 0, std::make_unique<IdentityFunction>(),
                        std::make_unique<DeterministicInitialization>());
                Eigen::MatrixXf input_matrix(9, 2);
                input_matrix << 1, 10, 2, 11, 3, 12, 4, 13, 5, 14, 6, 15, 7, 16, 8, 17, 9, 18;
                //std::cout << "Input: \n" << HelperFunctions::print_tensor(input_matrix,3,3,1) << std::endl;
                //input_matrix.transposeInPlace();
                Eigen::MatrixXf filter(2, 4);
                filter << -1, -0.5, 0, 0.5,
                        1, 1.5, 2, 0;
                conv_layer.set_filter(filter);
                conv_layer.feed_forward(input_matrix);
                //std::cout << HelperFunctions::print_tensor(conv_layer.get_m_z(), 1,1,2);
                Eigen::MatrixXf output_matrix(2, 2);
                output_matrix << 0.5, -8.5,
                        12, 52.5;
                REQUIRE(conv_layer.get_forward_output().isApprox(output_matrix));
            }
            THEN("The forward propagation is correct another more complex matrix") {
                ConvolutionalLayer conv_layer(
                        3, 3, 3, 2, 2, 2, 1, 0, std::make_unique<IdentityFunction>(),
                        std::make_unique<DeterministicInitialization>());
                Eigen::MatrixXf input_matrix(27, 1);
                input_matrix << 16,47,68,24,18,12,32,26,9,26,24,2,57,21,11,43,12,19,18,4,81,47,6,22,21,12,13;
                HelperFunctions::print_tensor(input_matrix, 3,3,3);
                //std::cout << "Input: \n" << HelperFunctions::print_tensor(input_matrix,3,3,1) << std::endl;
                //input_matrix.transposeInPlace();
                Eigen::MatrixXf filter(2, 12);
                filter << 0,1,-1,0,5,3,4,2,16,68,24,-2,
                        60,32,22,18,35,7,46,23,78,20,81,42;
                //filter.transposeInPlace();
                conv_layer.set_filter(filter);
                conv_layer.feed_forward(input_matrix);
                //std::cout << HelperFunctions::print_tensor(conv_layer.get_forward_output(), 2,2,2) << std::endl;
                Eigen::MatrixXf output_matrix(8, 1);
                output_matrix << 2171,5954,2170,2064,13042,11023,13575,6425;
                //std::cout << "Other\n" << HelperFunctions::print_tensor(output_matrix, 2,2,2) << std::endl;
                REQUIRE(conv_layer.get_forward_output().isApprox(output_matrix));
            }
        }
    }
}