#include <ActivationFunction/ReluFunction.h>

#include <iostream>

ReluFunction::ReluFunction(float leak_factor) {
    this->leak_factor = leak_factor;
}

Eigen::MatrixXf
ReluFunction::apply_function(const Eigen::MatrixXf &input) const {
    // Apply an unary expression corresponding to a relu lambda to every matrix
    // element if elem <0 -> 0 else -> elem
    auto output = (input.unaryExpr([this](float x) {
        if (x >= 0.0) {
            return x;
        } else {
            return (-1) * leak_factor * x;
        }
    })).eval();

    return output;
}

Eigen::MatrixXf
ReluFunction::apply_derivative(const Eigen::MatrixXf &m_a,
                               const Eigen::MatrixXf &dC_da) const {
    // Apply an unary expression corresponding to a derivative relu lambda to
    // every matrix element and multiply with derivative next layer if elem >0 ->
    // 1 else -> -elem*leak_factor (leak_factor 0 by defautl)
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