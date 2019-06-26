#include <catch2/catch.hpp>

#include "ActivationFunction/ReluFunction.h"
#include "ActivationFunction/SigmoidFunction.h"
//#include "NN.h"
#include "Layer/ConvolutionalLayer.h"
#include "LossFunction/BinaryCrossEntropyLoss.h"
//#include <Eigen/Core>
#include "ActivationFunction/IdentityFunction.h"
#include "ActivationFunction/SoftmaxFunction.h"
#include "LossFunction/MultiClassLoss.h"
#include "RandomInitialization/SimpleRandomInitialization.h"
#include <iostream>
#include <vector>

#include "RandomInitialization/XavierInitialization.h"

/*SCENARIO("Test Compute Layer") {
    GIVEN("A input matrix, a set of m_w") {
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
                    result << 3.78, 4.95, 6.12, 7.29, -2.7, -2.97, -3.24, -3.51,
0.36, 0.54, 0.72, 0.9; REQUIRE(NeN.compute_layer(input,
weights,bias).isApprox(result));
                }
            }
            WHEN("The bias is not 0") {
                bias << 0.6,-0.1,0.9;
                THEN("the computation is still correct") {
                    result << 4.38,  5.55,  6.72,  7.89,-2.8 , -3.07, -3.34,
-3.61, 1.26,  1.44,  1.62,  1.8 ; REQUIRE(NeN.compute_layer(input,
weights,bias).isApprox(result));
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
}*/

SCENARIO("Test activation functions") {
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> input;
  input.resize(2, 3);
  input << 5.0, 9.0, 0.1, 49.0, -3.0, 0.0;
  Eigen::Matrix<float, 2, 3> result;
  GIVEN("A sigmoid function") {
    SigmoidFunction sigmoidFun = SigmoidFunction();
    THEN("The sigmoid is correctly calculated") {
      result << 0.99330715, 0.99987661, 0.52497919, 1, 0.04742587, 0.5;
      auto res_sigmoid = sigmoidFun.apply_function(input);
      REQUIRE(res_sigmoid.isApprox(result));
    }
    THEN("The derivate of the sigmoid function should be correctly applied") {
      result << 6.64805576e-03, 1.23374775e-04, 2.49376040e-01, 0.00000000e+00,
          4.51766569e-02, 2.50000000e-01;
      auto res_sigmoid = sigmoidFun.apply_derivate(input);
      REQUIRE(res_sigmoid.isApprox(result));
    }
  }
  GIVEN("A relu function") {
    ReluFunction relu_func = ReluFunction();
    THEN("The relu is correctly calculated") {
      result << 5.0, 9.0, 0.1, 49.0, 0.0, 0.0;
      // std::cout << "Result: " << result << std::endl;
      auto res = relu_func.apply_function(input);
      // std::cout << "Red: " << res << std::endl;
      // std::cout << "Diff " << res-result << std::endl;
      REQUIRE(res.isApprox(result));
    }
    THEN("The derivative of the relu function should be correctly applied") {
      result << 1.0, 1.0, 1.0, 1.0, 0.0, 0.0;
      auto res = relu_func.apply_derivate(input);
      //      std::cout << "Result: " << result << std::endl;
      //      std::cout << "Red: " << res << std::endl;
      //      std::cout << "Diff " << res-result << std::endl;
      REQUIRE(res.isApprox(result));
    }
  }

  GIVEN("A softmax function") {
    SoftmaxFunction softmax = SoftmaxFunction();
    THEN("The softmax is correctly calculated") {
      Eigen::Matrix<float, 2, 2> soft_in;
      soft_in << 1.0, -2.0, 2.0, 3.25;
      Eigen::Matrix<float, 2, 2> soft_res;
      soft_res << 0.26894142, 1.0 - 0.9947798743, 0.73105858, 0.9947798743;
      auto res = softmax.apply_function(soft_in);
      REQUIRE(res.isApprox(soft_res));
    }
    THEN("The derivative of the softmax function should be correctly applied") {
      Eigen::Matrix<float, 2, 1> soft_in;
      soft_in << 1.0, 2.0;
      Eigen::Matrix<float, 2, 1> soft_res;
      soft_res << -0.196612, 0.196612;
      auto res = softmax.apply_derivate(soft_in);
      REQUIRE(res.isApprox(soft_res));
    }
  }
}

SCENARIO("Test loss function") {

  GIVEN("Binary Cross Entropy Loss") {
    BinaryCrossEntropyLoss binary_loss = BinaryCrossEntropyLoss();
    Eigen::Matrix<float, 1, 4> input;
    input << 0.7, 0.3, 0.1, 0.2;
    Eigen::Vector<int, 4> result;
    result << 0, 1, 0, 0;
    THEN("The loss is correct") {
      float cost = 0.684112418905977;
      // std::cout << "Binary loss: " << binary_loss.calculate_loss(input,
      // result);
      REQUIRE(std::abs(binary_loss.calculate_loss(input, result) - cost) <
              0.0000001);
    }
  }

  GIVEN("MultiClassLoss") {
    MultiClassLoss binary_loss = MultiClassLoss();
    Eigen::Matrix<float, 2, 4> input;
    input << 0.5, 0.3, 0.07, 0.13, 0.5, 0.7, 0.93, 0.87;
    Eigen::Vector<int, 4> result;
    result << 0, 1, 0, 0;
    THEN("The loss is correct") {
      float cost = 5.74930299 / 4.0;
      // std::cout << "loss: " << binary_loss.calculate_loss(input, result);
      REQUIRE(std::abs(binary_loss.calculate_loss(input, result) - cost) <
              0.0000001);
    }
    THEN("The derivate of the loss is correct") {
      Eigen::Matrix<float, 2, 4> res;
      res << -1 / 0.5, 0, -1 / 0.07, -1 / 0.13, 0, -1 / 0.7, 0, 0;
      binary_loss.backpropagate(input, result);
      // std::cout << "res\n" << res << "\nshould be\n" <<
      // binary_loss.get_backpropagate() << std::endl;
      REQUIRE(res.isApprox(binary_loss.get_backpropagate()));
    }
  }
}

SCENARIO("Test layers") {
  GIVEN("Test convoluational layer") {
    ConvolutionalLayer conv =
        ConvolutionalLayer(1, 1, 3, 3, std::make_unique<IdentityFunction>(),
                           std::make_unique<SimpleRandomInitialization>());
    Eigen::Matrix<float, 3, 3> input;
    input << -2, -1.5, -1.0, -0.5, 0, 0.5, 1, 1.5, 2;
    Eigen::Matrix<float, 2, 2> filter;
    filter << -0.5, 0, 0.5, 1;
    conv.m_w = filter;
    Eigen::Matrix<float, 2, 2> result;
    result << 0.75, 1.25, 2.25, 2.75;
    std::vector<Eigen::MatrixXf> in;
    in.push_back(input);
    conv.feed_forward(in);
    THEN("The forward feed ist correct") {
      // std::cout << "Input: "<< input << "\nfilter: " << filter
      // <<"\nRealResult: " <<result << "\nGotten Res: " <<conv.m_a <<
      // std::endl;
      REQUIRE(conv.m_a[0].isApprox(result));
    }
    Eigen::Matrix<float, 2, 2> dev_prev;
    dev_prev << 0, -1, 1.5, 2;
    Eigen::Matrix<float, 2, 2> res_dC_dw;
    res_dC_dw << 0.75, 2, 4.5, 5.75;
    Eigen::Matrix<float, 3, 3> res_dC_da_prev;
    res_dC_da_prev << 2, 2.5, 0.75, -1, -1.5, -0.75, 0, 0.5, 0;
    std::vector<Eigen::MatrixXf> prev;
    prev.push_back(dev_prev);
    THEN("Backward is correct") {
      conv.backpropagation(in, prev);
      REQUIRE(conv.m_dC_dw[0].isApprox(res_dC_dw));
      // std::cout << "Res_prev\n" << res_dC_da_prev << "\nIs\n"
      // <<conv.m_dC_da_prev;
      REQUIRE(conv.m_dC_da_prev[0].isApprox(res_dC_da_prev));
    }
  }
}

SCENARIO("Test random initialization") {
  GIVEN("Xavier initialization") {
    XavierInitialization xav_init=XavierInitialization();
    Eigen::Matrix<float,20,20> input;
    xav_init.initialize(input);
    THEN("The mean of the output should be correct") {
      REQUIRE(std::abs(input.mean()) <0.01);
    }
    THEN("The variance of the output should be correct") {
      //REQUIRE(input.v)
      //std::cout << "rand\n" << input << std::endl;
    }
  }
}