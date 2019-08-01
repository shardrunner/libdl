#include "RandomInitialization/DeterministicInitialization.h"

void DeterministicInitialization::initialize(
        Eigen::Ref<Eigen::MatrixXf> input) const {
    srand(0);
    input.setRandom();
}