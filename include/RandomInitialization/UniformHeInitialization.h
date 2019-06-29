#include "RandomInitialization/RandomInitialization.h"

class UniformHeInitialization : public RandomInitialization {
public:
  void initialize(Eigen::Ref<Eigen::MatrixXf> input) const override;
};
