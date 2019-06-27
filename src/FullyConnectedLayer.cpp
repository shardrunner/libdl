#include "Layer/FullyConnectedLayer.h"
#include <ctime>
#include <iostream>

FullyConnectedLayer::FullyConnectedLayer(const int input_size, const int output_size, std::unique_ptr<ActivationFunction> activation_function, std::unique_ptr<RandomInitialization> random_initialization) : m_activation_function(std::move(activation_function)), m_random_initialization(std::move(random_initialization)) {
  this->m_input_size=input_size;
  this->m_output_size=output_size;
  m_w.resize(input_size, output_size);
  m_b.resize(output_size);
  initialize_parameter();
  //this->m_activation_function=std::move(activation_function);
  //this->m_random_initialization=std::move(random_initialization);
  //std::cout<< "Weights: " <<m_w << "\nBias: " <<m_b<<"\n\n";
}


void FullyConnectedLayer::feed_forward(const Eigen::MatrixXf &input) {
  //TODO bias correct per row/column added?
/*  int a = m_w.rows();
  int b = m_w.cols();
  int c = input.rows();
  int d = input.cols();
  int e = m_b.size();*/

  m_z = (m_w.transpose() * input).colwise() + m_b;

  //std::cout << "temp:\n" << m_w.transpose() * input << "\nbias\n" << m_b << std::endl;

  m_a=m_activation_function->apply_function(m_z);

  //std::cout << "prev_layer\n" << input << std::endl;

  //std::cout << "Z_forward\n" << m_z << std::endl;

  //std::cout << "Activated_forward\n" << m_a << std::endl;
}
void FullyConnectedLayer::backpropagation(const Eigen::MatrixXf &a_prev, const Eigen::MatrixXf &dC_da) {
  /*
   * Notation from 3blue1brown NeuralNetwork series:
   *
   * Input -> prev layer -> current layer -> next layer -> Loss
   *
   *
   * C: Loss, w: weights, z: result before activation [z=w*a(prev)+b], a: activated z [sigma(z)], d: derivative, a_prev: a of previous layer
   * dC/dw=(dz/dw)*(da/dz)*(dC/da)=(dz/dw)*(dC/dz)
   * dz/dw=a_prev
   * da/dz=derivative activation function with z
   * dC/da=derivative from next layer (see intermediate results dC/da(prev))
   *
   * Intermediate results:
   * derivative for previous layer computation (for next step/prev layer in backprop):
   * dC/da_prev=sum[(dz/da_prev)*(da/dz)*(dC/da)]=sum[(dz/da_prev)*(dC/dz)]
   * dz/da_prev=w
   *
   * bias backprop:
   * dC/db=(dz/db)*(da/dz)*(dC/da)=(dz/db)*(dC/dz)
   * dz/db=1
   *
   * a_prev: m_input_size x number_training_samples
   * dC/da=next layer derivative: m_output_size x number_training_samples
   *
   */

  const int number_training_samples=a_prev.cols();
  //calcualate intermediate value dC/dz
  Eigen::MatrixXf dC_dz=m_activation_function->apply_derivate(m_a, dC_da);

  //normalize sum over changes/derivatives from all samples, by dividing by number of samples
  m_dC_dw=a_prev*dC_dz.transpose()/number_training_samples;

  //also normalize sum of bias changes calculating mean and shrinking to vector
  m_dC_db=dC_dz.rowwise().mean();

  //calcualate derivative for prev layer (next step in backprop)
  m_dC_da_prev=m_w*dC_dz;
  //std::cout<<"dC_dw: " << m_dC_dw << "\ndC_db: " << m_dC_db << "\n\n+++++++\n" <<std::endl;//dC_dw2: "<<diff_h_layer << "\ndC_db2: " <<diff_h_bias << "\n-----------------------------------------"<<std::endl;

  //std::cout << "Backward_prop\n" << m_dC_da_prev << std::endl;
}

void FullyConnectedLayer::initialize_parameter() {
//  spdlog::debug("Before call");
  //spdlog::debug("after call");
  m_random_initialization->initialize(m_w);
  m_b.setZero();
  //m_w.setRandom();
  //m_b.setRandom();
  //std::cout << "init_Randos\n" << m_w << std::endl;
}

//TODO: Use optimizer class
void FullyConnectedLayer::update_parameter() {
  m_w = m_w - 0.3 * m_dC_dw;
  m_b = m_b - 0.3 * m_dC_db;

  //std::cout << "After update: w\n" << m_w << "\nb\n" << m_b << std::endl;
}

const Eigen::MatrixXf &FullyConnectedLayer::get_forward_output() {
//  spdlog::debug("before a");
  return m_a;
}
const Eigen::MatrixXf &FullyConnectedLayer::get_backward_output() {
  return m_dC_da_prev;
}
