#pragma once

#include "LossFunction/LossFunction.h"

class CrossEntropyLoss : public LossFunction {
public:
  float calculate_loss(const Eigen::MatrixXf &a_prev,
                       const Eigen::RowVectorXf &label) const override;
  void backpropagate(const Eigen::MatrixXf &a_prev,
                     const Eigen::RowVectorXf &label) override;
  const Eigen::MatrixXf &get_backpropagate() const override;
};