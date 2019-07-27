#pragma once

#include "LossFunction/LossFunction.h"

/**
 * Calculates the binary cross entropy loss for binary problems.
 */
class BinaryCrossEntropyLoss : public LossFunction {
public:
  [[nodiscard]] double
  calculate_loss(const Eigen::MatrixXf &a_prev,
                 const Eigen::VectorXi &label) const override;

  void backpropagate(const Eigen::MatrixXf &a_prev,
                     const Eigen::VectorXi &label) override;
  [[nodiscard]] const Eigen::MatrixXf &get_backpropagate() const override;
};
