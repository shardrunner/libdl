#include <catch2/catch.hpp>

#include "ActivationFunction/IdentityFunction.h"
#include "ActivationFunction/ReluFunction.h"
#include "Layer/FullyConnectedLayer.h"
#include "OptimizationFunction/SimpleOptimizer.h"
#include "RandomInitialization/DeterministicInitialization.h"

#include <vector>

SCENARIO("Test Compute Layer") {
  GIVEN("A simple fully connected layer") {
    auto fc_layer =
        FullyConnectedLayer(4, 3, std::make_unique<IdentityFunction>(),
                            std::make_unique<DeterministicInitialization>(),
                            std::make_unique<SimpleOptimizer>(0.1));
    Eigen::MatrixXf input(4, 2);
    Eigen::VectorXf bias(3);
    input.col(0) << 1, 3, 2, 0;
    input.col(1) << 0, -2, 0, -1;

    Eigen::MatrixXf weights(4, 3);
    weights.col(0) << 0, 2, 0, -1;
    weights.col(1) << 3, 1, 1, 0;
    weights.col(2) << -1, 0, 0, 4;
    fc_layer.set_weights(weights);
    Eigen::MatrixXf result(3, 2);

    WHEN("The bias is 0") {
      bias.setZero();
      THEN("The feed forward operation is correct") {
        fc_layer.feed_forward(input);
        result << 6, -3, 8, -2, -1, -4;
        REQUIRE(fc_layer.get_forward_output().isApprox(result));
      }
    }
    WHEN("The bias is not 0") {
      bias << 1, -1, 0;
      fc_layer.set_bias(bias);
      THEN("the computation is still correct") {
        fc_layer.feed_forward(input);
        result << 7, -2, 7, -3, -1, -4;
        REQUIRE(fc_layer.get_forward_output().isApprox(result));
      }
    }
  }
}
