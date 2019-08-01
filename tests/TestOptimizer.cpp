#include "catch2/catch.hpp"

#include "OptimizationFunction/Adam.h"

#include <Eigen/Core>

SCENARIO("Test optimizer") {
    GIVEN("The adam optimizer") {
        WHEN("Bias optimizer and weights optimizer are called with the same values") {
            Adam adam(4, 1, 4);
            Eigen::MatrixXf input_w(4, 1);
            Eigen::VectorXf input_b(4);
            input_w << 1, 5, 3, 0;
            input_b << 1, 5, 3, 0;
            Eigen::MatrixXf der_w(4, 1);
            Eigen::VectorXf der_b(4);
            der_w << -1, 2, 0, 3;
            der_b << -1, 2, 0, 3;
            THEN("They should return the same values") {
                adam.optimize_bias(input_b, der_b);
                adam.optimize_bias(input_b, der_b);
                adam.optimize_weights(input_w, der_w);
                adam.optimize_weights(input_w, der_w);
                REQUIRE(input_w.isApprox(input_b));
            }
        }WHEN("Optimize weights is executed") {
            Adam adam(0.1, 0.2, 0.5, 0.9, 3, 2, 4);
            Eigen::MatrixXf input_w(3, 2);
            input_w << 1, 0, -3, 2, 0, -1;
            Eigen::MatrixXf der_w(3, 2);
            der_w << 0.5, -0.1, 0, 0, 1, 0.2;
            THEN("It should update correctly") {
                adam.optimize_weights(input_w, der_w);
                Eigen::MatrixXf result(3, 2);
                result << 0.928571, 0.0333333, -3, 2, -0.0833333, -1.05;
                REQUIRE(result.isApprox(input_w));
                adam.optimize_weights(input_w, der_w);
                result << 0.844226, 0.077733, -3, 2, -0.178366, -1.11307;
                REQUIRE(result.isApprox(input_w));
            }
        }
    }
}