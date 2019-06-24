#pragma once

#include <Eigen/Core>

/**
 * Base class for the random initialization for the layer parameter
 */
class RandomInitialization {
public:
  virtual void initialize(Eigen::Ref<Eigen::MatrixXf> input) const =0;
  virtual void print(int i)=0;
};