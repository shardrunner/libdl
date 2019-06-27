#pragma once

#include "RandomInitialization.h"
#include <Eigen/Core>

class SimpleRandomInitialization : public RandomInitialization {
public:
  void initialize(Eigen::Ref<Eigen::MatrixXf> input) const override;
};