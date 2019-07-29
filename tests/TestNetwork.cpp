#include <catch2/catch.hpp>

#include "NeuralNetwork.h"
#include "Layer/ConvolutionalLayer.h"
#include "Layer/FullyConnectedLayer.h"
#include "RandomInitialization/DeterministicInitialization.h"
#include "ActivationFunction/ReluFunction.h"
#include "ActivationFunction/SigmoidFunction.h"
#include "ActivationFunction/TanhFunction.h"
#include "ActivationFunction/IdentityFunction.h"
#include "ActivationFunction/SoftmaxFunction.h"
#include "LossFunction/BinaryCrossEntropyLoss.h"
#include "LossFunction/MultiCrossEntropyLoss.h"

SCENARIO("Test complete network") {
    GIVEN("A deterministic network architecture") {
        Eigen::MatrixXf input(27, 2);
        input.col(0) << 1, 0, -3, 4, 0, 3, 4, 1, 0, 0, 0, 2, 4, 5, 2, 3, -4, 0, 0, 0, -4, -5, 3, 1, 0, -3, 0;
        input.col(1) << 0, 0, -4, -5, 3, 1, 0, -3, 0, 1, 2, 1, 0, 0, 3, -2, 0, 4, 0, 3, 4, 1, 0, 0, 0, 2, 4;
        Eigen::VectorXi labels(2);
        labels << 1, 0;

        // initialize loss function
        auto loss_func = std::make_unique<MultiCrossEntropyLoss>();

        // initialize network with loss function
        auto mnet = NeuralNetwork(std::move(loss_func));

        auto hidden_layer0 = std::make_unique<ConvolutionalLayer>(
                3, 3, 3, 2, 2, 2, 1, 0, std::make_unique<ReluFunction>(),
                std::make_unique<DeterministicInitialization>());
        auto hidden_layer1 = std::make_unique<ConvolutionalLayer>(
                2, 2, 2, 2, 2, 4, 1, 0, std::make_unique<TanhFunction>(),
                std::make_unique<DeterministicInitialization>());
        auto hidden_layer2 = std::make_unique<FullyConnectedLayer>(
                4, 8, std::make_unique<SigmoidFunction>(),
                std::make_unique<DeterministicInitialization>());
        auto hidden_layer3 = std::make_unique<FullyConnectedLayer>(
                8, 3, std::make_unique<IdentityFunction>(),
                std::make_unique<DeterministicInitialization>());
        auto output_layer = std::make_unique<FullyConnectedLayer>(
                3, 2, std::make_unique<SoftmaxFunction>(),
                std::make_unique<DeterministicInitialization>());
        mnet.add_layer(std::move(hidden_layer0));
        mnet.add_layer(std::move(hidden_layer1));
        mnet.add_layer(std::move(hidden_layer2));
        mnet.add_layer(std::move(hidden_layer3));
        mnet.add_layer(std::move(output_layer));
        WHEN("The network is trained") {
            mnet.train_network(input, labels, -1, 50, -1);
            THEN("The loss should be predictable") {
                REQUIRE(std::abs(mnet.get_loss(mnet.test_network(input), labels) - 0.403376) < 0.0001);
            }

        }
    }GIVEN("A deterministic network architecture") {
        Eigen::MatrixXf input(27, 2);
        input.col(0) << 1, 0, -3, 4, 0, 3, 4, 1, 0, 0, 0, 2, 4, 5, 2, 3, -4, 0, 0, 0, -4, -5, 3, 1, 0, -3, 0;
        input.col(1) << 0, 0, -4, -5, 3, 1, 0, -3, 0, 1, 2, 1, 0, 0, 3, -2, 0, 4, 0, 3, 4, 1, 0, 0, 0, 2, 4;
        Eigen::VectorXi labels(2);
        labels << 1, 0;

        // initialize loss function
        auto loss_func = std::make_unique<BinaryCrossEntropyLoss>();

        // initialize network with loss function
        auto mnet = NeuralNetwork(std::move(loss_func));

        auto hidden_layer0 = std::make_unique<ConvolutionalLayer>(
                3, 3, 3, 2, 2, 2, 1, 0, std::make_unique<ReluFunction>(),
                std::make_unique<DeterministicInitialization>());
        auto hidden_layer1 = std::make_unique<ConvolutionalLayer>(
                2, 2, 2, 2, 2, 4, 1, 0, std::make_unique<TanhFunction>(),
                std::make_unique<DeterministicInitialization>());
        auto hidden_layer2 = std::make_unique<FullyConnectedLayer>(
                4, 8, std::make_unique<SoftmaxFunction>(),
                std::make_unique<DeterministicInitialization>());
        auto hidden_layer3 = std::make_unique<FullyConnectedLayer>(
                8, 3, std::make_unique<IdentityFunction>(),
                std::make_unique<DeterministicInitialization>());
        auto output_layer = std::make_unique<FullyConnectedLayer>(
                3, 1, std::make_unique<SigmoidFunction>(),
                std::make_unique<DeterministicInitialization>());
        mnet.add_layer(std::move(hidden_layer0));
        mnet.add_layer(std::move(hidden_layer1));
        mnet.add_layer(std::move(hidden_layer2));
        mnet.add_layer(std::move(hidden_layer3));
        mnet.add_layer(std::move(output_layer));
        WHEN("The network is trained") {
            mnet.train_network(input, labels, -1, 50, -11);
            THEN("The loss should be predictable") {
                REQUIRE(std::abs(mnet.get_loss(mnet.test_network(input), labels) - 0.496379) < 0.0001);
            }

        }
    }

}