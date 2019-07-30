#include <iostream>

#include "ActivationFunction/IdentityFunction.h"
#include "ActivationFunction/ReluFunction.h"
#include "ActivationFunction/SigmoidFunction.h"
#include "ActivationFunction/SoftmaxFunction.h"
#include "ActivationFunction/TanhFunction.h"
#include "Layer/ConvolutionalLayer.h"
#include "Layer/FullyConnectedLayer.h"
#include "LossFunction/BinaryCrossEntropyLoss.h"
#include "LossFunction/MultiCrossEntropyLoss.h"
#include "NeuralNetwork.h"
#include "RandomInitialization/DeterministicInitialization.h"
#include "RandomInitialization/HetalInitialization.h"
#include "RandomInitialization/UniformHeInitialization.h"
#include "RandomInitialization/SimpleRandomInitialization.h"
#include "RandomInitialization/XavierInitialization.h"
#include "RandomInitialization/UniformXavierInitialization.h"
#include "OptimizationFunction/SimpleOptimizer.h"
#include "OptimizationFunction/Adam.h"
#include <Eigen/Core>
#include <memory>
#include <vector>

#include "extern/mnist/include/mnist/mnist_reader.hpp"
#include "extern/mnist/include/mnist/mnist_utils.hpp"
//#include "omp.h"

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
    int samples = 2000;

    Eigen::VectorXi labels;
    labels.resize(samples);
    Eigen::MatrixXf image;
    image.resize(784, samples);
    image.setZero();

    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < 28; j++) {
            for (int k = 0; k < 28; k++) {
                image(j * 28 + k, i) =
                        (float) unsigned(dataset.training_images[i][j * 28 + k]);
            }
        }
        labels(i) = (int) unsigned(dataset.training_labels[i]);
    }

    // convert mnist test dataset to eigen matrix

    // take 50 test samples
    int test_samples = 100;

    Eigen::VectorXi test_labels;
    test_labels.resize(test_samples);
    Eigen::MatrixXf test_image;
    test_image.resize(784, test_samples);
    test_image.setZero();

    for (int i = 0; i < test_samples; i++) {
        for (unsigned long j = 0; j < 28; j++) {
            for (unsigned long k = 0; k < 28; k++) {
                // std::cout << (float) unsigned(dataset.training_images[i][j*28+k]) <<
                // " ";
                test_image(long(j * 28 + k), long(i)) =
                        (float) unsigned(dataset.test_images[i][j * 28 + k]);
            }
        }
        test_labels(long(i)) = (int) unsigned(dataset.test_labels[i]);
    }

    Eigen::MatrixXf img;
    img.resize(27, 2);
    //img << 1,2,3,4,5,6,7,8,9;//,10,11,12,13,14,15,16,17,18;
    img.col(0) << 1, 0, -3, 4, 0, 3, 4, 1, 0, 0, 0, 2, 4, 5, 2, 3, -4, 0, 0, 0, -4, -5, 3, 1, 0, -3, 0;
    img.col(1) << 0, 0, -4, -5, 3, 1, 0, -3, 0, 1, 2, 1, 0, 0, 3, -2, 0, 4, 0, 3, 4, 1, 0, 0, 0, 2, 4;
    //img.col(0) <<0,0,0,0,0,0,0,0,0;
    //images.push_back(img);
    Eigen::VectorXi labels3;
    labels3.resize(2);
    labels3 << 1, 3;

    // initialize loss function
    auto loss_func = std::make_unique<MultiCrossEntropyLoss>();

    // initialize network with loss function
    auto mnet = NeuralNetwork(std::move(loss_func));

    // TODO Add check input fully conected layer size
    // Network architecture
/*    auto hid_layer = std::make_unique<ConvolutionalLayer>(
            3, 3, 3, 2, 2, 2,1,0, std::make_unique<ReluFunction>(),
            std::make_unique<DeterministicInitialization>());
    auto hid2_layer = std::make_unique<FullyConnectedLayer>(
            8, 6, std::make_unique<SigmoidFunction>(),
            std::make_unique<DeterministicInitialization>());
    auto out_layer = std::make_unique<FullyConnectedLayer>(
            6, 4, std::make_unique<SoftmaxFunction>(),
            std::make_unique<DeterministicInitialization>());
    mnet.add_layer(std::move(hid_layer));
    mnet.add_layer(std::move(hid2_layer));
    mnet.add_layer(std::move(out_layer));*/

    auto hidden_layer_0 = std::make_unique<ConvolutionalLayer>(
            28, 28, 1, 5, 5, 6, 1, 0, std::make_unique<ReluFunction>(),
            std::make_unique<UniformHeInitialization>(),std::make_unique<Adam>(6,25*1,6));
    auto hidden_layer_1 = std::make_unique<ConvolutionalLayer>(
            24, 24, 6, 2, 2, 6, 2, 0, std::make_unique<ReluFunction>(),
            std::make_unique<UniformHeInitialization>(),std::make_unique<Adam>(6,4*6,6));
    auto hidden_layer_2 = std::make_unique<ConvolutionalLayer>(
            12, 12, 6, 5, 5, 16, 1, 0, std::make_unique<ReluFunction>(),
            std::make_unique<UniformHeInitialization>(),std::make_unique<Adam>(16,25*6,16));
    auto hidden_layer_3 = std::make_unique<ConvolutionalLayer>(
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


    //test_image=image.block(0,200,image.rows(),100);
    //test_labels=labels.segment(200,100);

    // train the network
    mnet.train_network(image, labels, -1, 5, 1);

    // test part of the test set
    auto predictions = mnet.calculate_accuracy(test_image, test_labels);

    std::cout << "Some tests \n" << std::endl;

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
