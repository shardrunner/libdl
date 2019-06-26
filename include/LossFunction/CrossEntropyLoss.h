#pragma once

#include "LossFunction/LossFunction.h"

class CrossEntropyLoss : public LossFunction {
public:
  float calculate_loss(const Eigen::MatrixXf &a_prev, const Eigen::VectorXi &label) const override;//const Eigen::MatrixXf &a_prev,
                     //  const Eigen::RowVectorXf &label) const override;
  void backpropagate(const Eigen::MatrixXf &a_prev,
                     const Eigen::VectorXi &label) override;
  const Eigen::MatrixXf &get_backpropagate() const override;
};