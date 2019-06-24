#pragma once

#include <Eigen/Core>

/**
 * Base class for the loss functions.
 */
class LossFunction {
public:
  virtual ~LossFunction() = default;
  //TODO Change Row vectors for multi classification
  virtual float calculate_loss() const =0; //const Eigen::MatrixXf &a_prev, const Eigen::RowVectorXf &label) const=0;
  virtual void backpropagate(const Eigen::MatrixXf &a_prev, const Eigen::MatrixXf &label)=0;
  virtual const Eigen::MatrixXf &get_backpropagate() const=0;

protected:
  Eigen::MatrixXf temp_loss;
};