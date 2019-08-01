#include "ActivationFunction/ReluFunction.h"
#include "ActivationFunction/SoftmaxFunction.h"
#include "ActivationFunction/TanhFunction.h"
#include "Layer/ConvolutionLayer.h"
#include "Layer/FullyConnectedLayer.h"
#include "LossFunction/MultiCrossEntropyLoss.h"
#include "NeuralNetwork.h"
#include "RandomInitialization/UniformHeInitialization.h"
#include "RandomInitialization/UniformXavierInitialization.h"
#include "OptimizationFunction/Adam.h"

#include "extern/mnist/include/mnist/mnist_reader.hpp"
#include "extern/mnist/include/mnist/mnist_utils.hpp"

#include <Eigen/Core>

#include <memory>
#include <vector>
#include <iostream>



/**
 * MNist example.
 * Is also easily adaptable for other problems.
 * @return Status
 */
int main() {
    // MNIST_DATA_LOCATION set by MNIST cmake config
    // std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

    // Load MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
            mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(
                    MNIST_DATA_LOCATION);

    // convert mnist to binary
    mnist::binarize_dataset(dataset);


    // convert mnist training dataset to eigen matrix
    // take 2000 training samples
    unsigned int samples = 2000;

    Eigen::VectorXi labels;
    labels.resize(samples);
    Eigen::MatrixXf image;
    image.resize(784, samples);
    image.setZero();

    for (unsigned int i = 0; i < samples; i++) {
        for (unsigned int j = 0; j < 28; j++) {
            for (unsigned int k = 0; k < 28; k++) {
                image(j * 28 + k, i) =
                        (float) unsigned(dataset.training_images[i][j * 28 + k]);
            }
        }
        labels(i) = (int) unsigned(dataset.training_labels[i]);
    }

    // take 200 test samples
    unsigned int test_samples = 200;

    Eigen::VectorXi test_labels;
    test_labels.resize(test_samples);
    Eigen::MatrixXf test_image;
    test_image.resize(784, test_samples);
    test_image.setZero();

    for (unsigned int i = 0; i < test_samples; i++) {
        for (unsigned int j = 0; j < 28; j++) {
            for (unsigned int k = 0; k < 28; k++) {
                test_image(j * 28 + k, i) =
                        (float) unsigned(dataset.test_images[i][j * 28 + k]);
            }
        }
        test_labels(i) = (int) unsigned(dataset.test_labels[i]);
    }

    // initialize loss function
    auto loss_func = std::make_unique<MultiCrossEntropyLoss>();

    // initialize network with loss function
    auto mnet = NeuralNetwork(std::move(loss_func));

    // Network architecture
    auto hidden_layer_0 = std::make_unique<ConvolutionLayer>(
            28, 28, 1, 5, 5, 6, 1, 0, std::make_unique<ReluFunction>(),
            std::make_unique<UniformHeInitialization>(),std::make_unique<Adam>(6,25*1,6));
    auto hidden_layer_1 = std::make_unique<ConvolutionLayer>(
            24, 24, 6, 2, 2, 6, 2, 0, std::make_unique<ReluFunction>(),
            std::make_unique<UniformHeInitialization>(),std::make_unique<Adam>(6,4*6,6));
    auto hidden_layer_2 = std::make_unique<ConvolutionLayer>(
            12, 12, 6, 5, 5, 16, 1, 0, std::make_unique<ReluFunction>(),
            std::make_unique<UniformHeInitialization>(),std::make_unique<Adam>(16,25*6,16));
    auto hidden_layer_3 = std::make_unique<ConvolutionLayer>(
            8, 8, 16, 2, 2, 16, 2, 0, std::make_unique<ReluFunction>(),
            std::make_unique<UniformHeInitialization>(),std::make_unique<Adam>(16,4*16,16));
    auto hidden_layer_4 = std::make_unique<FullyConnectedLayer>(
            4 * 4 * 16, 84, std::make_unique<TanhFunction>(),
            std::make_unique<UniformXavierInitialization>(),std::make_unique<Adam>(4*4*16,84,84));
    auto out_layer = std::make_unique<FullyConnectedLayer>(
            84, 10, std::make_unique<SoftmaxFunction>(),
            std::make_unique<UniformXavierInitialization>(),std::make_unique<Adam>(84,10,10));
    mnet.add_layer(std::move(hidden_layer_0));
    mnet.add_layer(std::move(hidden_layer_1));
    mnet.add_layer(std::move(hidden_layer_2));
    mnet.add_layer(std::move(hidden_layer_3));
    mnet.add_layer(std::move(hidden_layer_4));
    mnet.add_layer(std::move(out_layer));

    // train the network
    mnet.train_network(image, labels, 128, 40, 1);

    // test part of the test set
    auto predictions = mnet.calculate_accuracy(test_image, test_labels);

    // print some images from the test set with the resulting label
    for (int z = 0; z < 100; z += 10) {
        std::cout << "Image from testset: \n";
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                std::cout << test_image(j + i * 28, z) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "Label from testset: " << test_labels(z) << std::endl;
        std::cout << "Predicted: " << predictions(z) << std::endl;
    }
}
