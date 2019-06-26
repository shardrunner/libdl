#include "ActivationFunction/ActivationFunction.h"

class SoftmaxFunction :public ActivationFunction {
public:
  Eigen::MatrixXf apply_function(const Eigen::MatrixXf &input) const override;
  Eigen::MatrixXf apply_derivate(const Eigen::MatrixXf &m_a,
                                 const Eigen::MatrixXf &dC_da) const override;
};