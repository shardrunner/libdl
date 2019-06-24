#pragma once

#include <Eigen/Core>
#include <list>

class MTensor : public Eigen::MatrixXf {
public:
  int z_dim();

private:
  int z_dim;

  Eigen::MatrixXf tensor;
};