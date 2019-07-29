#pragma once

#include <Eigen/Core>

/**
 * Represents a layer in the neural network. An abstract base class.
 */
class BaseLayer {
public:
  /**
   * Default virtual destructor.
   */
  virtual ~BaseLayer() = default;

  /**
   * Feed forwards the input in this layer and stores the output internally for
   * the backpropagation.
   * @param input The input matrix.
   */
  virtual void feed_forward(const Eigen::MatrixXf &input) = 0;

  /**
   * Does the backpropagation for this layer. Takes the internally stored value
   * of the feed forward, the input of the previous layer and the derivative of
   * the next layer. Stores the result internally.
   * @param a_prev Feed forward result of the previous layer.
   * @param dC_da Derivative of the next layer.
   */
  virtual void backpropagation(const Eigen::MatrixXf &a_prev,
                               const Eigen::MatrixXf &dC_da) = 0;

  /**
   * Returns a reference to the stored feed forward output.
   * @return The reference to the stored feed forward output.
   */
  virtual const Eigen::MatrixXf &get_forward_output() const = 0;

  /**
   * Returns a reference to the stored backpropagation output.
   * @return The reference to the stored backpropagation output.
   */
  virtual const Eigen::MatrixXf &get_backward_output() const = 0;

  /**
   * Initializes the internal parameters bias and weight according to the
   * provided random function.
   */
  virtual void initialize_parameter() = 0;

  /**
   * Updates the parameters weight and bias according to the chosen optimization
   * method.
   */
  virtual void update_parameter() = 0;

  /**
   * Return the expected number of inputs values per sample.
   * Used to check if the constructed network is correct.
   * @return The expected number of input values per sample.
   */
  [[nodiscard]] virtual int get_number_inputs() const = 0;

  /**
   * Returns the expected number of output values per sample.
   * Used to check if the constructed network is correct.
   * @return The expected number of output values per sample.
   */
  [[nodiscard]] virtual int get_number_outputs() const =0;

  virtual void set_weights(const Eigen::MatrixXf &new_weights) = 0;

  virtual void set_bias(const Eigen::VectorXf &new_bias)=0;

  [[nodiscard]] virtual const Eigen::MatrixXf &get_weights() const =0;

  [[nodiscard]] virtual const Eigen::VectorXf &get_bias() const =0;
protected:
  /**
   * The size of the input.
   */
  int m_input_size;

  /**
   * The size of the output.
   */
  int m_output_size;
};