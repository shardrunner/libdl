#pragma once

#include "LossFunction/LossFunction.h"

/**
 * Calculates the cross entropy loss for multi class problems.
 */
class MultiCrossEntropyLoss : public LossFunction {
public:
  [[nodiscard]] double
  calculate_loss(const Eigen::MatrixXf &a_prev,
                 const Eigen::VectorXi &label) const override;
  void backpropagate(const Eigen::MatrixXf &a_prev,
                     const Eigen::VectorXi &label) override;
  [[nodiscard]] const Eigen::MatrixXf &get_backpropagate() const override;
};