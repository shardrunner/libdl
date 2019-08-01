#include "RandomInitialization/UniformHeInitialization.h"

#include <random>

// https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution
// https://towardsdatascience.com/hyper-parameters-in-action-part-ii-weight-initializers-35aee1a28404
void UniformHeInitialization::initialize(
    Eigen::Ref<Eigen::MatrixXf> input) const {
  // random device class instance, source of 'true' randomness for initializing
  // random seed
  std::random_device rd;

  // Mersenne twister PRNG, initialized with seed from previous random device
  // instance
  std::mt19937 gen(rd());

  std::default_random_engine generator(rd());

  // stddev = var^-2,
  // He init uniform -> limit sqrt(6/fan_in)
  float limit = std::sqrt(float(6.0) / float(input.rows()));

  // instance of class std::normal_distribution with specific mean and stddev
  std::uniform_real_distribution<float> distribution((-1) * limit, limit);

  for (long i = 0; i < input.rows(); i++) {
    for (long j = 0; j < input.cols(); j++) {
      input(i, j) = distribution(generator);
    }
  }
}
