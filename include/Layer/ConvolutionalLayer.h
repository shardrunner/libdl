//#include "BaseLayer.h"

#include "ActivationFunction/ActivationFunction.h"
#include "RandomInitialization/RandomInitialization.h"
#include <vector>

#include <memory>

class ConvolutionalLayer {
public:
  ConvolutionalLayer(int number_input_channel, int number_output_channel, int filter_width, int filter_heigth, std::unique_ptr<ActivationFunction> activation_function, std::unique_ptr<RandomInitialization> random_initialization);
  void feed_forward(const std::vector<Eigen::MatrixXf> &input);
  void backpropagation(const std::vector<Eigen::MatrixXf> &a_prev,
                       const std::vector<Eigen::MatrixXf> &dC_da);
  const std::vector<Eigen::MatrixXf> &get_forward_output();
  const std::vector<Eigen::MatrixXf> &get_backward_output();
  void initialize_parameter();
  void update_parameter();

public:
  Eigen::MatrixXf m_w;
  float m_b;
  std::vector<Eigen::MatrixXf> m_a;
  std::vector<Eigen::MatrixXf> m_z;
  std::vector<Eigen::MatrixXf> m_dC_dw;
  std::vector<Eigen::MatrixXf> m_dC_db;
  std::vector<Eigen::MatrixXf> m_dC_da_prev;

  int m_number_input_channel;
  int m_number_output_channel;

  std::unique_ptr<ActivationFunction> m_activation_function;
  std::unique_ptr<RandomInitialization> m_random_initialization;
};