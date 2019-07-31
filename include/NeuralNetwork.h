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

    NeuralNetwork();

    void use_multiclass_loss();

    void add_conv_layer(int input_height,int input_width, int input_channels, int filter_height, int filter_width, int output_channels, int stride, int padding);

    void add_conv_layer_simple(int input_height,int input_width, int input_channels, int filter_height, int filter_width, int output_channels, int stride, int padding);

    void add_fc_layer(int input_size, int output_size);

    void add_fc_layer_relu(int input_size, int output_size);

    void add_output_layer(int input_size, int output_size);

    void add_output_layer_simple(int input_size, int output_size);

    void train_batch(Eigen::Ref<const Eigen::MatrixXf> &input_batch, Eigen::Ref<const Eigen::VectorXi> &label_batch);

    void feed_forward_py(Eigen::Ref<const Eigen::MatrixXf> &input_batch);

    void set_layer_weights(Eigen::Ref<const Eigen::MatrixXf> &param, unsigned long position);

    const Eigen::MatrixXf &get_layer_weights(unsigned long position) const;

    void set_layer_bias(Eigen::Ref<const Eigen::VectorXf> &param, unsigned long position);

    const Eigen::VectorXf &get_layer_bias(unsigned long position) const;

    float get_current_accuracy(const Eigen::VectorXi &labels) const;

    float get_current_error(const Eigen::VectorXi &labels) const;

    Eigen::VectorXi get_predicted_classes(const Eigen::MatrixXf &prediction) const;

    const Eigen::MatrixXf &get_current_prediction() const;

    int layer_size() const;



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
    Eigen::MatrixXi calculate_accuracy(const Eigen::MatrixXf &input,
                                       const Eigen::MatrixXi &label);

    void check_network(long input_size);


    [[nodiscard]] std::vector<std::tuple<Eigen::MatrixXf, Eigen::VectorXf>> get_layer_parameter() const;

    void set_layer_parameter(const std::vector<std::tuple<Eigen::MatrixXf, Eigen::VectorXf>> &parameter_list);

    [[nodiscard]] double get_loss(const Eigen::MatrixXf &feed_forward_input, const Eigen::VectorXi &label) const;

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
                         const Eigen::MatrixXi &label);

    /**
     * Updates the parameters of the layers depending of the calculated
     * minimization in the backpropagation step.
     */
    void update();

private:
    std::shared_ptr<spdlog::logger> m_nn_logger;
};