#include "RandomInitialization/RandomInitialization.h"

class UniformXavierInitialization : public RandomInitialization {
public:
  void initialize(Eigen::Ref<Eigen::MatrixXf> input) const override;
};
