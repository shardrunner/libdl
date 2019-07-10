#include <iostream>

#include "ActivationFunction/IdentityFunction.h"
#include "ActivationFunction/ReluFunction.h"
#include "ActivationFunction/SigmoidFunction.h"
#include "ActivationFunction/SoftmaxFunction.h"
#include "ActivationFunction/TanFunction.h"
#include "Layer/ConvolutionalLayer.h"
#include "Layer/FullyConnectedLayer.h"
#include "LossFunction/BinaryCrossEntropyLoss.h"
#include "LossFunction/MultiClassLoss.h"
#include "NeuralNetwork.h"
#include "RandomInitialization/DeterministicInitialization.h"
#include "RandomInitialization/HetalInitialization.h"
#include "RandomInitialization/SimpleRandomInitialization.h"
#include "RandomInitialization/XavierInitialization.h"
#include <Eigen/Core>
#include <memory>
#include <vector>

#include "extern/mnist/include/mnist/mnist_reader.hpp"
#include "extern/mnist/include/mnist/mnist_utils.hpp"
#include "omp.h"

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
            (float)unsigned(dataset.training_images[i][j * 28 + k]);
      }
    }
    labels(i) = (int)unsigned(dataset.training_labels[i]);
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
    for (int j = 0; j < 28; j++) {
      for (int k = 0; k < 28; k++) {
        // std::cout << (float) unsigned(dataset.training_images[i][j*28+k]) <<
        // " ";
        test_image(j * 28 + k, i) =
            (float)unsigned(dataset.test_images[i][j * 28 + k]);
      }
    }
    test_labels(i) = (int)unsigned(dataset.test_labels[i]);
  }

  // initialize loss function
  auto loss_func = std::make_unique<MultiClassLoss>();

  // initialize network with loss function
  auto mnet = NeuralNetwork(std::move(loss_func));

  // TODO Add check input fully conected layer size
  // Network architecture
  auto hid_layer = std::make_unique<ConvolutionalLayer>(
      28, 28, 1, 1, 3, 3, std::make_unique<ReluFunction>(),
      std::make_unique<XavierInitialization>());
  auto hid2_layer = std::make_unique<FullyConnectedLayer>(
      26 * 26, 16, std::make_unique<ReluFunction>(),
      std::make_unique<HetalInitialization>());
  auto out_layer = std::make_unique<FullyConnectedLayer>(
      16, 10, std::make_unique<SoftmaxFunction>(),
      std::make_unique<XavierInitialization>());
  mnet.add_layer(std::move(hid_layer));
  mnet.add_layer(std::move(hid2_layer));
  mnet.add_layer(std::move(out_layer));

  // train the network
  mnet.train_network(image, labels, -1, 2, 1);

  // test part of the test set
  auto predictions = mnet.calc_accuracy(test_image, test_labels);

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
