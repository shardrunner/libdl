#include <catch2/catch.hpp>

#include "Layer/ConvolutionalLayer.h"
#include "ActivationFunction/IdentityFunction.h"
#include "RandomInitialization/DeterministicInitialization.h"
#include "ManageLoggers.h"

#include <iostream>



SCENARIO("Test Convolutional Layer") {
    ManageLoggers loggers;
    loggers.initLoggers();
    WHEN("The im2col function is applied") {
        THEN("It works for a one channel one sample input matrix") {
            ConvolutionalLayer conv_layer(3, 3, 1, 1, 2, 2, 1, std::make_unique<IdentityFunction>(),
                                          std::make_unique<DeterministicInitialization>());
            Eigen::MatrixXf input_matrix(9,1);
            input_matrix << 1,4,7,2,5,8,3,6,9;
            conv_layer.im2col(input_matrix);
            Eigen::MatrixXf output_matrix(4,4);
            output_matrix << 1, 4, 2, 5,
            4, 7, 5, 8,
            2, 5, 3, 6,
            5, 8, 6, 9;
            REQUIRE(conv_layer.getMIm2ColMatrix().isApprox(output_matrix));
        }
        THEN("It works for a multi channel one sample input matrix") {
            ConvolutionalLayer conv_layer(3, 3, 2, 1, 2, 2, 1, std::make_unique<IdentityFunction>(),
                                          std::make_unique<DeterministicInitialization>());
            Eigen::MatrixXf input_matrix(18,1);
            input_matrix << 1,4,7,2,5,8,3,6,9,10,13,16,11,14,17,12,15,18;
            conv_layer.im2col(input_matrix);
            Eigen::MatrixXf output_matrix(8,4);
            output_matrix << 1, 4, 2, 5,
            4, 7, 5, 8,
            2, 5, 3, 6,
            5, 8, 6, 9,
            10,13,11,14,
            13,16,14,17,
            11,14,12,15,
            14,17,15,18;
            REQUIRE(conv_layer.getMIm2ColMatrix().isApprox(output_matrix));
        }
            THEN("It works for a one channel two sample input matrix") {
            ConvolutionalLayer conv_layer(3, 3, 1, 1, 2, 2, 1, std::make_unique<IdentityFunction>(),
                                          std::make_unique<DeterministicInitialization>());
            Eigen::MatrixXf input_matrix(2,9);
            input_matrix << 1,4,7,2,5,8,3,6,9,
                            10,13,16,11,14,17,12,15,18;
            input_matrix.transposeInPlace();
            conv_layer.im2col(input_matrix);
            Eigen::MatrixXf output_matrix(4,8);
            output_matrix << 1, 4, 2, 5,10,13,11,14,
            4, 7, 5, 8,13,16,14,17,
            2, 5, 3, 6,11,14,12,15,
            5, 8, 6, 9,14,17,15,18;
            REQUIRE(conv_layer.getMIm2ColMatrix().isApprox(output_matrix));
            }
            THEN("It works for a one channel one sample not matching stride two input matrix") {
            ConvolutionalLayer conv_layer(3, 3, 1, 1, 2, 2, 2, std::make_unique<IdentityFunction>(),
                                          std::make_unique<DeterministicInitialization>());
            Eigen::MatrixXf input_matrix(9,1);
            input_matrix << 1,4,7,2,5,8,3,6,9;
            conv_layer.im2col(input_matrix);
            Eigen::MatrixXf output_matrix(4,1);
            //std::cout << "output m:\n" << *im2col_input;
            output_matrix << 1, 4, 2, 5;
            REQUIRE(conv_layer.getMIm2ColMatrix().isApprox(output_matrix));
            }
            THEN("It works for a one channel one sample matching stride two input matrix") {
            ConvolutionalLayer conv_layer(4, 3, 1, 1, 2, 2, 2, std::make_unique<IdentityFunction>(),
                                          std::make_unique<DeterministicInitialization>());
            Eigen::MatrixXf input_matrix(12,1);
            input_matrix << 1,4,7,2,5,8,3,6,9,10,11,12;
            conv_layer.im2col(input_matrix);
            Eigen::MatrixXf output_matrix(4,2);
            output_matrix << 1, 7,
                            4, 2,
                            5, 3,
                            8, 6;
            REQUIRE(conv_layer.getMIm2ColMatrix().isApprox(output_matrix));
            }
    }
}