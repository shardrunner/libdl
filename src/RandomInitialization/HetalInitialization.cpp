#include "RandomInitialization/HetalInitialization.h"

#include <random>

#include <iostream>
//from https://stackoverflow.com/questions/38244877/how-to-use-stdnormal-distribution
//http://eigen.tuxfamily.org/dox-devel/classEigen_1_1DenseBase.html#title91
void HetalInitialization::initialize(Eigen::Ref<Eigen::MatrixXf> input) const {
  // random device class instance, source of 'true' randomness for initializing random seed
  std::random_device rd;

  // Mersenne twister PRNG, initialized with seed from previous random device instance
  std::mt19937 gen(rd());


  std::default_random_engine generator(rd());

  //stddev = var^-2, Hetal -> 2/num_inputs
  float sigma=std::sqrt(float(2.0)/float(input.rows()));

  // instance of class std::normal_distribution with specific mean and stddev
  std::normal_distribution<float> distribution(0, sigma);

  for (int i =0; i<input.rows(); i++) {
    for (int j=0; j<input.cols(); j++) {
      input(i, j) = distribution(generator);
    }
  }

  //auto temp=input.unaryExpr([&] (float x) {return distribution(generator);});




}
