#include <catch2/catch.hpp>

#include "HelperFunctions.h"
#include "LossFunction/BinaryCrossEntropyLoss.h"
#include "LossFunction/MultiCrossEntropyLoss.h"
#include <iostream>

SCENARIO("Test loss function") {
  GIVEN("Binary Cross Entropy Loss") {
    BinaryCrossEntropyLoss binary_loss = BinaryCrossEntropyLoss();
    Eigen::MatrixXf input(1, 4);
    input << 0.7, 0.3, 0.1, 0.2;
    Eigen::VectorXi result(4);
    result << 0, 1, 0, 0;
    WHEN("The loss of the forward output is calculated") {
      THEN("The loss is correct") {
        float cost = 0.684112418905977;
        REQUIRE(std::abs(binary_loss.calculate_loss(input, result) - cost) <
                0.0000001);
      }
    }
    WHEN("The derivative of the loss function is calculated") {
      binary_loss.backpropagate(input, result);
      THEN("The result is correct") {
        Eigen::MatrixXf derivative_loss = binary_loss.get_backpropagate();
        Eigen::MatrixXf derivative_result(1, 4);
        derivative_result << 3.33333, -3.33333, 1.11111, 1.25;
        REQUIRE(derivative_loss.isApprox(derivative_result));
      }
    }
  }

  GIVEN("MultiClassLoss") {
    MultiCrossEntropyLoss multiclass_loss = MultiCrossEntropyLoss();
    Eigen::MatrixXf input(2, 4);
    input << 0.5, 0.3, 0.07, 0.13, 0.5, 0.7, 0.93, 0.87;
    Eigen::VectorXi result(4);
    result << 0, 1, 0, 0;
    WHEN("The loss of the forward output is calculated") {
      THEN("The loss is correct") {
        float cost = 5.74930299 / 4.0;
        REQUIRE(std::abs(multiclass_loss.calculate_loss(input, result) - cost) <
                0.0000001);
      }
    }
    WHEN("The derivative of the loss is calculated") {
      multiclass_loss.backpropagate(input, result);
      THEN("The result is correct") {
        Eigen::MatrixXf res(2, 4);
        res << -1 / 0.5, 0, -1 / 0.07, -1 / 0.13, 0, -1 / 0.7, 0, 0;
        REQUIRE(res.isApprox(multiclass_loss.get_backpropagate()));
      }
    }
  }
}