#include "ActivationFunction/TanhFunction.h"

void
TanhFunction::forward_propagation(Eigen::MatrixXf &input) const {
    input.array() = input.array().tanh();
}

Eigen::MatrixXf
TanhFunction::apply_derivative(const Eigen::MatrixXf &m_a,
                               const Eigen::MatrixXf &dC_da) const {
    return ((1 - m_a.array().pow(2)) * dC_da.array()).matrix();
}
