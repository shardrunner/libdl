#pragma once

#include <Eigen/Core>

// TODO Xavier + He fix matrix operation population

/**
 * Abstract base class for the random initialization for the layer parameters.
 * Used for weights and bias initialization, when the setup is initialized.
 * A good initialization leeds to faster and better convergence due to better
 * weight behaviour. The most useful initialization function depends mainly on
 * the activation function of the layer.
 */
class RandomInitialization {
public:
  /**
   * Initializes the matrix with random values depending on the child
   * implementations.
   * @param input The matrix to populate with random values.
   */
  virtual void initialize(Eigen::Ref<Eigen::MatrixXf> input) const = 0;
};