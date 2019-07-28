#include "RandomInitialization/DeterministicInitialization.h"

#include <iostream>

void DeterministicInitialization::initialize(
    Eigen::Ref<Eigen::MatrixXf> input) const {
  srand(0);
  input.setRandom();
  //input.array()=input.array().round();
   // std::cout << "init_Randos\n" << input << std::endl;
  //input.setOnes();
  // input=Eigen::MatrixXf::Ones(input.rows(),input.cols());
  // input(1,0)=0;
}