#pragma once

#include "LossFunction/LossFunction.h"

class MultiClassLoss : public LossFunction {
public:
  float calculate_loss(const Eigen::MatrixXf &a_prev,
                       const Eigen::VectorXi &label) const override;
  void backpropagate(const Eigen::MatrixXf &a_prev,
                     const Eigen::VectorXi &label) override;
  const Eigen::MatrixXf &get_backpropagate() const override;
};