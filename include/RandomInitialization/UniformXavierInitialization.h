#pragma once

#include "RandomInitialization/RandomInitialization.h"

/**
 * Uniform Xavier initialization used for tanh, softmax and sigmoid activation
 * functions.
 */
class UniformXavierInitialization : public RandomInitialization {
public:
    /**
* [See abstract base class](@ref RandomInitialization)
*/
    void initialize(Eigen::Ref<Eigen::MatrixXf> input) const override;
};
