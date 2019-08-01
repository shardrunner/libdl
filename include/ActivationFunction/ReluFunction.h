#pragma once

#include "ActivationFunction/ActivationFunction.h"

/**
 * Represents a Relu function as activation function.
 *
 * This can also be a leaky Relu.
 * The Relu activation function is mainly used for convolution layers.
 * They allow for deeper networks.
 * For the default of 0, values smaller than 0 are set to 0. This can lead to
 * undesirable behaviour in the form of dying gradients. To circumvent this
 * issue LeakyRelu has a value a bit smaller than 0 for values smaller than 0.
 *
 */
class ReluFunction : public ActivationFunction {
public:
  /**
   * The constructor for the (Leaky)Relu function.
   * This constructor allows the definition of a leak factor for the Relu,
   * making it a LeakyRelu.
   * @param leak_factor The leak factor controls the Relu behaviour for values
   * smaller than 0.
   */
  explicit ReluFunction(float leak_factor = 0.0);

  /**
   * [See abstract base class](@ref ActivationFunction)
   *
   * if (x>0) {return x;}
   *
   * else {return leak_factor;}
   *
   *Apply an unary expression corresponding to a Relu lambda to every matrix
   *element if elem <0 -> 0 else -> elem
   */
  void forward_propagation(Eigen::MatrixXf &input) const override;

  /**
   * [See abstract base class](@ref ActivationFunction)
   *
   *Apply an unary expression corresponding to a derivative relu lambda to every
   *matrix element and multiply with derivative next layer if elem >0 ->1 else
   *-> -elem*leak_factor (leak_factor 0 by default)
   */
  [[nodiscard]] Eigen::MatrixXf
  apply_derivative(const Eigen::MatrixXf &m_a,
                   const Eigen::MatrixXf &dC_da) const override;

private:
  /**
   * The leak value for the leaky Relu. Defaults to 0.
   */
  float leak_factor;
};