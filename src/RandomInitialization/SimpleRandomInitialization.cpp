#include "RandomInitialization/SimpleRandomInitialization.h"

void SimpleRandomInitialization::initialize(
    Eigen::Ref<Eigen::MatrixXf> input) const {
  srand((unsigned int) time(0));
  srand(0);
  input.setRandom();
}

void SimpleRandomInitialization::print(int i) {
  //spdlog::error("Call {} successful",i);
}