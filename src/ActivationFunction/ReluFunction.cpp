#include <ActivationFunction/ReluFunction.h>

ReluFunction::ReluFunction(float leak_factor) {
  this->leak_factor = leak_factor;
}

void ReluFunction::forward_propagation(Eigen::MatrixXf &input) const {
  input = (input.unaryExpr([this](float x) {
    if (x >= 0.0) {
      return x;
    } else {
      return (-1) * leak_factor * x;
    }
  }));
}

Eigen::MatrixXf
ReluFunction::apply_derivative(const Eigen::MatrixXf &m_a,
                               const Eigen::MatrixXf &dC_da) const {
  return (m_a.unaryExpr([this](float x) -> float {
               if (x > 0.0) {
                 return 1.0;
               } else {
                 return (-1) * leak_factor * x;
               }
             })
              .array() *
          dC_da.array())
      .matrix();
}