#pragma once

#include "RandomInitialization.h"
#include <Eigen/Core>
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"

class SimpleRandomInitialization : public RandomInitialization {
public:
  void initialize(Eigen::Ref<Eigen::MatrixXf> input) const override;
  void print(int i);
};