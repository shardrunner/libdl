#pragma once

#include "NeuralNetwork.h"
#include <iostream>
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"

void NeuralNetwork::add_layer(std::unique_ptr<BaseLayer> layer) {
  m_layer_list.push_back(std::move(layer));
}
void NeuralNetwork::feed_forward(const Eigen::MatrixXf &input) {
  m_layer_list[0]->feed_forward(input);



  for (int i=1; i<m_layer_list.size(); i++) {
    m_layer_list[i]->feed_forward(m_layer_list[i-1]->get_forward_output());
  }
  //std::cout << "\nOutput: \n"<<m_layer_list[m_layer_list.size()-1]->get_forward_output() << std::endl;
}


void NeuralNetwork::backpropagation(const Eigen::MatrixXf &input, const Eigen::MatrixXf &label) {
  int number_layers=m_layer_list.size();

  spdlog::debug("rows output: {}, columns output: {}",m_layer_list[number_layers-1]->get_forward_output().rows(),m_layer_list[number_layers-1]->get_forward_output().cols());
  spdlog::debug("rows label: {}, columns label: {}",label.rows(),label.cols());
  spdlog::debug("before segfault");
  auto temp=(m_layer_list[number_layers-1]->get_forward_output()).eval();
  //std::cout << label.eval();
  //std::cout << temp.eval();
  spdlog::debug("temp");
  m_loss_function->backpropagate(temp, label);
  spdlog::debug("after segfault");

  if (number_layers==1) {
    m_layer_list[0]->backpropagation(input, m_loss_function->get_backpropagate());
    return;
  }

  m_layer_list[number_layers-1]->backpropagation(m_layer_list[number_layers-2]->get_forward_output(), m_loss_function->get_backpropagate());

  for (int i =number_layers-2; i>0; i--) {
    m_layer_list[i]->backpropagation(m_layer_list[i-1]->get_forward_output(), m_layer_list[i+1]->get_backward_output());
  }

  m_layer_list[0]->backpropagation(input, m_layer_list[1]->get_backward_output());

}
void NeuralNetwork::update() {
  for (int i = 0; i < m_layer_list.size(); i++) {
    m_layer_list[i]->update_parameter();
  }
}
NeuralNetwork::NeuralNetwork(std::unique_ptr<LossFunction> loss_function, int iterations, int divisor) : m_loss_function(std::move(loss_function)), m_iterations(iterations), m_divisor(divisor) {

}
void NeuralNetwork::train_network(const Eigen::MatrixXf &input,
                                  const Eigen::MatrixXf &label) {
  int number_layer=m_layer_list.size();

  for (int i=0; i < m_iterations; i++) {
    feed_forward(input);
    backpropagation(input, label);
    update();
    if (i % m_divisor == 0) {
      auto temp1=m_layer_list[number_layer-1]->get_forward_output();
      //std::cout << temp1 << "\n" << label;
      spdlog::debug("tesp");
      auto temp2=m_loss_function->calculate_loss(); //(temp1, label);
      std::cout << "Loss at iteration number:" << i << " " << temp2  << std::endl;
    }

  }
}
const Eigen::MatrixXf &
NeuralNetwork::test_network(const Eigen::MatrixXf &input) {
    feed_forward(input);
    return m_layer_list[m_layer_list.size() - 1]->get_forward_output();
}