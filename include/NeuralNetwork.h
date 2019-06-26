#pragma once

#include <Eigen/Core>
#include <vector>
#include <memory>

#include "Layer/BaseLayer.h"
#include "LossFunction/LossFunction.h"


class NeuralNetwork {
public:
  NeuralNetwork(std::unique_ptr<LossFunction> loss_function, int iterations, int divisor);
  void add_layer(std::unique_ptr<BaseLayer> layer);
  void train_network(const Eigen::MatrixXf &input, const Eigen::MatrixXi &label);
  const Eigen::MatrixXf &test_network(const Eigen::MatrixXf &input);
  Eigen::MatrixXf get_result();

private:
  std::vector<std::unique_ptr<BaseLayer>> m_layer_list;
  std::unique_ptr<LossFunction> m_loss_function;


  void feed_forward(const Eigen::MatrixXf &input);
  void backpropagation(const Eigen::MatrixXf &input, const Eigen::MatrixXi &label);
  void update();
  int m_iterations;
  int m_divisor;
};