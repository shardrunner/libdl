#include "RandomInitialization/UniformXavierInitialization.h"

#include <random>

void UniformXavierInitialization::initialize(
    Eigen::Ref<Eigen::MatrixXf> input) const {
  // random device class instance, source of 'true' randomness for initializing
  // random seed
  std::random_device rd;

  // Mersenne twister PRNG, initialized with seed from previous random device
  // instance
  std::mt19937 gen(rd());

  std::default_random_engine generator(rd());

  // stddev = var^-2,
  // Xavier init uniform -> limit sqrt(6/(fan_in+fan_out))
  float limit = std::sqrt(float(6.0) / float(input.rows() + input.cols()));

  // instance of class std::normal_distribution with specific mean and stddev
  std::uniform_real_distribution<float> distribution((-1) * limit, limit);

  for (int i = 0; i < input.rows(); i++) {
    for (int j = 0; j < input.cols(); j++) {
      input(i, j) = distribution(generator);
    }
  }

  // auto temp=input.unaryExpr([&] (float x) {return distribution(generator);});
}
