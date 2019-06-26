#include "RandomInitialization/DeterministicInitialization.h"

void DeterministicInitialization::initialize(
    Eigen::Ref<Eigen::MatrixXf> input) const {
  srand(0);
  input.setRandom();
  //input=Eigen::MatrixXf::Ones(input.rows(),input.cols());
  //input(1,0)=0;
}