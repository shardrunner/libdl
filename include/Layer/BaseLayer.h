#pragma once

#include <Eigen/Core>

class BaseLayer {
public:
  virtual ~BaseLayer() = default;
  virtual void feed_forward(const Eigen::MatrixXf &input)=0;
  virtual void backpropagation(const Eigen::MatrixXf &a_prev, const Eigen::MatrixXf &dC_da)=0;
  virtual const Eigen::MatrixXf &get_forward_output()=0;
  virtual const Eigen::MatrixXf &get_backward_output()=0;
  virtual void initialize_parameter()=0;
  virtual void update_parameter()=0;

protected:
  int m_input_size;
  int m_output_size;
};