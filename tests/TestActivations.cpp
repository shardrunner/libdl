#include <catch2/catch.hpp>

#include "ActivationFunction/ReluFunction.h"
#include "ActivationFunction/SoftmaxFunction.h"
#include "ActivationFunction/SigmoidFunction.h"
#include "ActivationFunction/TanhFunction.h"
#include "ActivationFunction/IdentityFunction.h"
#include <Eigen/Core>

#include <iostream>

SCENARIO("Test activation functions") {
    Eigen::MatrixXf input(2, 3);
    input << 5.0, 9.0, 0.1,
            49.0, -3.0, 0.0;
    Eigen::MatrixXf result(2, 3);
    Eigen::MatrixXf dummy_dC_da = Eigen::MatrixXf::Constant(input.rows(), input.cols(), 1);

    GIVEN("A sigmoid function") {
        SigmoidFunction sigmoidFun = SigmoidFunction();
        result << 0.99330715, 0.99987661, 0.52497919,
                1, 0.04742587, 0.5;
        sigmoidFun.apply_function(input);
        WHEN("The sigmoid activation function is applied") {
            THEN("The result is correctly calculated") {
                REQUIRE(input.isApprox(result));
            }
        }WHEN("The sigmoid activation derivative is applied") {
            THEN("The result should be correct") {
                result << 6.64805576e-03, 1.23374775e-04, 2.49376040e-01,
                        0.00000000e+00, 4.51766569e-02, 2.50000000e-01;
                auto res_sigmoid = sigmoidFun.apply_derivative(input, dummy_dC_da);
                REQUIRE(res_sigmoid.isApprox(result));
            }
        }
    }

    GIVEN("A relu function") {
        ReluFunction relu_func = ReluFunction();
        relu_func.apply_function(input);
        WHEN("The relu activation function is applied") {
            THEN("The relu is correctly calculated") {
                result << 5.0, 9.0, 0.1,
                        49.0, 0.0, 0.0;
                REQUIRE(input.isApprox(result));
            }
        }WHEN("The relu activation derivative is applied") {
            auto res = relu_func.apply_derivative(input, dummy_dC_da);
            THEN("The result should be correct") {
                result << 1.0, 1.0, 1.0,
                        1.0, 0.0, 0.0;
                REQUIRE(res.isApprox(result));
            }
        }
    }

    GIVEN("A softmax function") {
        SoftmaxFunction softmax = SoftmaxFunction();
        Eigen::MatrixXf soft_in(2, 2);
        soft_in << 1.0, -2.0, 2.0, 3.25;
        softmax.apply_function(soft_in);
        WHEN("The softmax function is applied") {
            THEN("The result is correct") {
                Eigen::MatrixXf soft_res(2, 2);
                soft_res << 0.26894142, 1.0 - 0.9947798743,
                        0.73105858, 0.9947798743;
                REQUIRE(soft_in.isApprox(soft_res));
                //REQUIRE(soft_in.sum() == 2);
                REQUIRE(std::abs(soft_in.sum() - float(2)) < double(0.00001));
            }
        }WHEN("The derivative of the softmax function is applied") {
            Eigen::MatrixXf soft_in(2, 1);
            Eigen::MatrixXf dummy_dC_da = Eigen::MatrixXf::Constant(2, 1, 1);
            soft_in << 1.0, 2.0;
            softmax.apply_function(soft_in);
            auto res = softmax.apply_derivative(soft_in, dummy_dC_da);
            THEN("The result should be correct") {
                Eigen::MatrixXf soft_res(2, 1);
                soft_res << 0,
                        0;
                REQUIRE(res.isApprox(soft_res));
            }
        }
        soft_in.resize(3, 2);
        soft_in << 5, 0,
                -1, 2,
                3, 1;
        softmax.apply_function(soft_in);
        WHEN("The softmax function is applied") {
            THEN("The result is correct") {
                Eigen::MatrixXf soft_res(3, 2);
                soft_res.col(0) << 0.87887824, 0.00217852, 0.11894324;
                soft_res.col(1) << 0.09003057, 0.66524096, 0.24472847;
                REQUIRE(soft_in.isApprox(soft_res));
                //REQUIRE(soft_in.sum() == 2);
                REQUIRE(std::abs(soft_in.sum() - float(2)) < float(0.00001));
            }
        }WHEN("The derivative of the softmax function is applied") {
            Eigen::MatrixXf dummy_dC_da(3, 2);
            dummy_dC_da << 0.5, 0.8,
                    -0.2, 1,
                    0, 0.4;
            soft_in << 0.6, 0.2,
                    0.1, 0.4,
                    0.3, 0.4;
            auto res = softmax.apply_derivative(soft_in, dummy_dC_da);
            THEN("The result should be correct") {
                Eigen::MatrixXf soft_res(3, 2);
                soft_res << 0.132, 0.016,
                        -0.048, 0.112,
                        -0.084, -0.128;
                REQUIRE(res.isApprox(soft_res));
            }
        }
    }

    GIVEN("A identity function") {
        IdentityFunction identity = IdentityFunction();
        identity.apply_function(input);
        WHEN("The identity function is applied") {
            Eigen::MatrixXf result = input;
            THEN("The result is correct") {
                REQUIRE(input.isApprox(result));
            }
        }WHEN("The derivative of the identity function is applied") {
            auto result = identity.apply_derivative(input, dummy_dC_da);
            THEN("The result should be correct") {
                REQUIRE(result.isApprox(dummy_dC_da));
            }
        }
    }
}
