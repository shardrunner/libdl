#pragma once

#import "LossFunction/LossFunction.h"

class MultiCrossEntropyLoss : public LossFunction {
public:
  float calculate_loss(const Eigen::MatrixXf &a_prev,
                       const Eigen::VectorXi &label) const override;
  void backpropagate(const Eigen::MatrixXf &a_prev,
                     const Eigen::VectorXi &label) override;
  const Eigen::MatrixXf &get_backpropagate() const override;
};
