#pragma once

#include <Eigen/Core>
#include <iostream>
#include <memory>
#include <vector>

#include "Layer/Layer.h"
#include "LossFunction/LossFunction.h"

#include "spdlog/spdlog.h"
#include <tuple>

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
   * Empty constructor.
   *
   * Initializes logger and sets cores.
   */
  NeuralNetwork();

  /**
   * Initialise multiclass loss.
   */
  void use_multiclass_loss();

  /**
   * Pypind unique_ptr replacement.
   */
  void add_conv_layer(int input_height, int input_width, int input_channels,
                      int filter_height, int filter_width, int output_channels,
                      int stride, int padding);

  /**
   * Pypind unique_ptr replacement.
   */
  void add_conv_layer_simple(int input_height, int input_width,
                             int input_channels, int filter_height,
                             int filter_width, int output_channels, int stride,
                             int padding);

  /**
   * Pypind unique_ptr replacement.
   */
  void add_fc_layer(int input_size, int output_size);

  /**
   * Pypind unique_ptr replacement.
   */
  void add_fc_layer_relu(int input_size, int output_size);

  /**
   * Pypind unique_ptr replacement.
   */
  void add_output_layer(int input_size, int output_size);

  /**
   * Pypind unique_ptr replacement.
   */
  void add_output_layer_simple(int input_size, int output_size);

  /**
   * Trains a batch.
   * @param input_batch The input batch.
   * @param label_batch The input label.
   */
  void train_batch(Eigen::Ref<const Eigen::MatrixXf> &input_batch,
                   Eigen::Ref<const Eigen::VectorXi> &label_batch);

  /**
   * Feed forward for pybind.
   * @param input_batch The input to feed forward.
   */
  void feed_forward_py(Eigen::Ref<const Eigen::MatrixXf> &input_batch);

  /**
   * Set weight for layer at indices position.
   * @param param The new weights.
   * @param position The layer position.
   */
  void set_layer_weights(Eigen::Ref<const Eigen::MatrixXf> &param,
                         unsigned long position);

  /**
   * Get the layer weights at the given position.
   * @param position The layer position.
   * @return The weights from that layer.
   */
  [[nodiscard]] const Eigen::MatrixXf &
  get_layer_weights(unsigned long position) const;

  /**
   * Set bias for layer at indices position.
   * @param param The new bias.
   * @param position The layer position.
   */
  void set_layer_bias(Eigen::Ref<const Eigen::VectorXf> &param,
                      unsigned long position);

  /**
   * Set bias for layer at indices position.
   * @param param The new bias.
   * @param position The layer position.
   */
  [[nodiscard]] const Eigen::VectorXf &
  get_layer_bias(unsigned long position) const;

  /**
   * Calculate the accuracy for the last processed feed forward batch.
   * @param labels The labels for the last batch.
   * @return The accuracy.
   */
  [[nodiscard]] float get_current_accuracy(const Eigen::VectorXi &labels) const;

  /**
   * Calculate the error for the last processed feed forward batch.
   * @param labels The labels for the last batch.
   * @return The error.
   */
  [[nodiscard]] float get_current_error(const Eigen::VectorXi &labels) const;

  /**
   * Calculate the predicted classes a given feed forward prediction.
   * @param The feed forward prediction.
   * @return A vector with the predictions as elements.
   */
  [[nodiscard]] Eigen::VectorXi
  get_predicted_classes(const Eigen::MatrixXf &prediction) const;

  /**
   * Get the feed forward prediction from the last processed feed forward batch.
   * @return The prediction for the batch.
   */
  [[nodiscard]] const Eigen::MatrixXf &get_current_prediction() const;

  /**
   * Get the size of the network.
   * @return The size of the network.
   */
  [[nodiscard]] unsigned long layer_size() const;

  /**
   * Does the feed forward step for all layers contained in the network using
   * the provided input.
   * @param input The input to the first layer of the network.
   */
  void feed_forward(const Eigen::MatrixXf &input);

  /**
   * Adds a layer to the back of the network.
   * @param layer The desired layer.
   */
  void add_layer(std::unique_ptr<Layer> layer);

  /**
   * Train the network according to the input parameters.
   * @param input The training samples to train on.
   * @param label The corresponding ground truth labels for the training
   * samples.
   * @param batch_size The desired batch size.
   * @param epochs The desired batch size/epochs.
   * @param divisor Prints the current loss every i-th iteration.
   */
  void train_network(const Eigen::MatrixXf &input, const Eigen::VectorXi &label,
                     int batch_size, int epochs, int divisor);

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
  Eigen::VectorXi calculate_accuracy(const Eigen::MatrixXf &input,
                                     const Eigen::VectorXi &label);

  /**
   * Checks if the given network dimensions match.
   * @param input_size The input for the first layer.
   */
  void check_network(long input_size);

  /**
   * Gets the layer parameter as vector of tuple.
   * @return The layer parameter.
   */
  [[nodiscard]] std::vector<std::tuple<Eigen::MatrixXf, Eigen::VectorXf>>
  get_layer_parameter() const;

  /**
   * Sets the layer parameter.
   * @param parameter_list The new layer parameter.
   */
  void set_layer_parameter(
      const std::vector<std::tuple<Eigen::MatrixXf, Eigen::VectorXf>>
          &parameter_list);

  /**
   * Calculates the loss with the given feed forward input and the label vector.
   * @param feed_forward_input The feed forward input.
   * @param label The label vector.
   * @return The loss.
   */
  [[nodiscard]] double get_loss(const Eigen::MatrixXf &feed_forward_input,
                                const Eigen::VectorXi &label) const;

private:
  /**
   * Vector of pointers to the layers contained in the neural network.
   */
  std::vector<std::unique_ptr<Layer>> m_layer_list;

  /**
   * Pointer to the loss function.
   */
  std::unique_ptr<LossFunction> m_loss_function;

  /**
   * Does the backpropagation for all layers contained in the network using the
   * data calculated in the feed forward step.
   * @param input The input to the first layer of the network.
   * @param label The ground truth labels for the input.
   */
  void backpropagation(const Eigen::MatrixXf &input,
                       const Eigen::VectorXi &label);

  /**
   * Updates the parameters of the layers depending of the calculated
   * minimization in the backpropagation step.
   */
  void update();

private:
  std::shared_ptr<spdlog::logger> m_nn_logger;
};