#pragma once

#include <Eigen/Core>
#include <memory>

/**
 * Simple Deep Learning Library from Brunner Michael.
 *
 * You can currently set the number of iterations, the input dimension, the
 * hidden layer size and the learning rate. It currently uses binary
 * Cross-Entropy and simple gradient descent. Currently most things are
 * hardcoded. The calculations are done using Eigen float Matrices. Used
 * inspiration from
 * https://towardsdatascience.com/how-to-build-a-simple-neural-network-from-scratch-with-python-9f011896d2f3
 * to get the backpropagation working.
 */
class NN {
public:
  /**
   * The constructors of the library sets the dimensions of the network.
   * @param dim_x The number of input features. For the XOR problem this is two.
   * @param dim_h The number of hidden layer neurons.
   * @param dim_y The number of outputs. Currently only one should work.
   */
  NN(int dim_x, int dim_h, int dim_y);

  /**
   * A binary Cross-Entropy cost function to compute the loss.
   *
   * This is a measure of the accuracy of the network.
   * @param NN_result The predicted result of the Neural Network.
   * @param y_output The real known output.
   * @return The Cross-Entropy loss as a measure of the accuracy.
   */
  float cost_func(
      const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &NN_result,
      const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &y_output);

  /**
   * The backpropagation of the Neural Network.
   *
   * The general structure of the backpropagation was taken from
   * https://towardsdatascience.com/how-to-build-a-simple-neural-network-from-scratch-with-python-9f011896d2f3
   * .
   *
   * @param x_input The inputs to the network.
   * @param y_output The given labels for the inputs.
   * @param o_h_layer The output of the hidden layer with the given inputs.
   * @param o_o_layer The output of the output layer with the given inputs and
   * the output of the hidden layer.
   * @param h_layer The weights of the hidden layer.
   * @param o_layer The weights of the output layer.
   * @param h_bias The weights of the hidden layer bias.
   * @param o_bias The weights of the output layer bias.
   * @param learning_rate The learning rate for the weights updating.
   */
  void backprop(
      const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &x_input,
      const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &y_output,
      const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &o_h_layer,
      const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &o_o_layer,
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &h_layer,
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &o_layer,
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &h_bias,
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &o_bias,
      float learning_rate);

  /**
   * The compute layer computes the output of one of the NN layers with all its
   * neurons.
   *
   * It takes the input to the layer, the weights, the bias and returns the
   * output of the layer without the activation.
   * @param input The input to the layer.
   * @param weights The weights of the layer.
   * @param bias The bias of the layer
   * @return The output of the layer without the activation.
   */
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> compute_layer(
      const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &input,
      const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &weights,
      const Eigen::Matrix<float, Eigen::Dynamic, 1> &bias);

  /**
   * Takes a matrix and performs the sigmoid function on this matrix.
   * @param input A matrix, which the sigmoid will be applied to.
   */
  void sigmoid(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &input);

  /**
   * Performs the training and testing of the network.
   *
   * There are currently no sophisticated training algorithms. Just a number of
   * iterations. This functions connects the other functions and also performs
   * the activation. Also does a very weak proof the function.
   * @param iterations The number of iterations to process.
   * @param x_input The inputs for the NN.
   * @param y_output The true labels for the input.
   * @param learning_rate The learning rate for the weights.
   */
  void train_net(
      int iterations,
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &x_input,
      const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &y_output,
      float learning_rate);

  /**
   * The dimensions of the network.
   */
private:
  int dim_x;
  int dim_y;
  int dim_h;
};
