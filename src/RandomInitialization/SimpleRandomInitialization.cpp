#include "RandomInitialization/SimpleRandomInitialization.h"

void SimpleRandomInitialization::initialize(
    Eigen::Ref<Eigen::MatrixXf> input) const {
  srand((unsigned int) time(0));
  input.setRandom();
}