#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file

#include <catch2/catch.hpp>

#include <Eigen/Core>
#include "NN.h"
#include "ReluFunction.h"
#include "SigmoidFunction.h"



SCENARIO("Test Compute Layer") {
    GIVEN("A input matrix, a set of weights") {
        NN NeN=NN(2,3,1);
        Eigen::Matrix<float, 2, 4> input;
        Eigen::Matrix<float, 3,1> bias;
        input << 1.8, 2.7, 3.6, 4.5, 5.4, 6.3, 7.2, 8.1;

        Eigen::Matrix<float, 3, 2> weights;
        weights << 0.9, 0.4, 0.3, -0.6, 0.2, 0.0;
        Eigen::Matrix<float, 3, 4> result;

            WHEN("The bias is 0") {
                bias.setZero();
                THEN("the computation is correct") {
                    result << 3.78, 4.95, 6.12, 7.29, -2.7, -2.97, -3.24, -3.51, 0.36, 0.54, 0.72, 0.9;
                    REQUIRE(NeN.compute_layer(input, weights,bias).isApprox(result));
                }
            }
            WHEN("The bias is not 0") {
                bias << 0.6,-0.1,0.9;
                THEN("the computation is still correct") {
                    result << 4.38,  5.55,  6.72,  7.89,-2.8 , -3.07, -3.34, -3.61, 1.26,  1.44,  1.62,  1.8 ;
                    REQUIRE(NeN.compute_layer(input, weights,bias).isApprox(result));
                }
            }
    }
}

SCENARIO("Test cost function") {
    NN NeN=NN(2,3,1);
    GIVEN("A predicted result and an output") {
        Eigen::Matrix<float, 4, 1> pred;
        pred << 0.7,0.3,0.1,0.2;
        Eigen::Matrix<float, 4,1> truth;
        truth << 0.0,1,0.0,0.0;

        THEN("The cost is correct") {
            float cost=0.684112418905977;
            REQUIRE(std::abs(NeN.cost_func(pred,truth)-cost) < 0.0000001);
        }
    }
}

SCENARIO("Test old sigmoid function") {
    NN NeN=NN(2,3,1);
    GIVEN("A Matrix") {
        Eigen::Matrix<float, Eigen::Dynamic,Eigen::Dynamic> m;
        m.resize(2,3);
        m << 5.0,9.0,0.1,49.0,-3.0,0.0;
        THEN("The sigmoid is correctly calculated") {
            Eigen::Matrix<float,2,3> result;
            result << 0.99330715, 0.99987661, 0.52497919,1, 0.04742587, 0.5;
            NeN.sigmoid(m);
            REQUIRE(m.isApprox(result));
        }
    }
}

SCENARIO("Test sigmoid implementation")  {
  SigmoidFunction sigmoidFun=SigmoidFunction();
  GIVEN("A Matrix the sigmoid function should be applied") {
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> m;
    m.resize(2, 3);
    m << 5.0, 9.0, 0.1, 49.0, -3.0, 0.0;
    THEN("The sigmoid is correctly calculated") {
      Eigen::Matrix<float, 2, 3> result;
      result << 0.99330715, 0.99987661, 0.52497919, 1, 0.04742587, 0.5;
      sigmoidFun.apply_function(m);
      REQUIRE(m.isApprox(result));
    }
  }
  GIVEN("A Matrix the derivate of the sigmoid function should be applied") {
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> m;
    m.resize(2, 3);
    m << 5.0, 9.0, 0.1, 49.0, -3.0, 0.0;
    THEN("The derivate of the sigmoid function should be correctly applied") {

    }
  }
}
