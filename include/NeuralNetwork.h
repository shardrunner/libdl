#pragma once

#include <Eigen/Core>
#include <iostream>
#include <memory>
#include <vector>

#include "Layer/BaseLayer.h"
#include "LossFunction/LossFunction.h"


#include "spdlog/spdlog.h"

/**
 * Represents the neural network and performs the operations on it.
 * Can be populated depending on the needs.
 */
class NeuralNetwork {
public:
  /**
   * Construct a empty neural network with a loss function.
   * @param loss_function The desired loss function.
   */
  explicit NeuralNetwork(std::unique_ptr<LossFunction> loss_function);

  /**
   * Adds a layer to the back of the network.
   * @param layer The desired layer.
   */
  void add_layer(std::unique_ptr<BaseLayer> layer);

  /**
   * Train the network according to the input parameters.
   * @param input The training samples to train on.
   * @param label The corresponding ground truth labels for the training
   * samples.
   * @param batch_size The desired batch size.
   * @param iterations The desired batch size/epochs.
   * @param divisor Prints the current loss every i-th iteration.
   */
  void train_network(const Eigen::MatrixXf &input, const Eigen::MatrixXi &label,
                     int batch_size, int iterations, int divisor);

  /**
   * Tests the network by feeding the input in.
   * Can be used for validation and testing.
   * return The predicted labels.
   */
  const Eigen::MatrixXf &test_network(const Eigen::MatrixXf &input);

  /**
   * Calculates the accuracy of the prediction.
   * @param input The prediction done by the network.
   * @param label The true labels.
   * @return Conversion of the prediction to binary values.
   */
  Eigen::MatrixXi calc_accuracy(const Eigen::MatrixXf &input,
                                const Eigen::MatrixXi &label);

  void check_network(long input_size);
private:
  /**
   * Vector of pointers to the layers contained in the neural network.
   */
  std::vector<std::unique_ptr<BaseLayer>> m_layer_list;

  /**
   * Pointer to the loss function.
   */
  std::unique_ptr<LossFunction> m_loss_function;

  /**
   * Does the feed forward step for all layers contained in the network using
   * the provided input.
   * @param input The input to the first layer of the network.
   */
  void feed_forward(const Eigen::MatrixXf &input);

  /**
   * Does the backpropagation for all layers contained in the network using the
   * data calculated in the feed forward step.
   * @param input The input to the first layer of the network.
   * @param label The ground truth labels for the input.
   */
  void backpropagation(const Eigen::MatrixXf &input,
                       const Eigen::MatrixXi &label);
  /**
   * Updates the parameters of the layers depending of the calculated
   * minimization in the backpropagation step.
   */
  void update();
private:
    std::shared_ptr<spdlog::logger> m_nn_logger;
};