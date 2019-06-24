#include "Layer/BaseLayer.h"

#include "ActivationFunction/ActivationFunction.h"
#include "RandomInitialization/RandomInitialization.h"
#include <memory>

class MaxPoolLayer : public BaseLayer {
public:
  MaxPoolLayer(int number_input_channel, int number_output_channel, int filter_width, int filter_heigth, std::unique_ptr<ActivationFunction> activation_function, std::unique_ptr<RandomInitialization> random_initialization);
  void feed_forward(const Eigen::MatrixXf &input) override;
  void backpropagation(const Eigen::MatrixXf &a_prev,
                       const Eigen::MatrixXf &dC_da) override;
  const Eigen::MatrixXf &get_forward_output() override;
  const Eigen::MatrixXf &get_backward_output() override;
  void initialize_parameter() override;
  void update_parameter() override;

private:
  Eigen::MatrixXf m_w;
  Eigen::VectorXf m_b;
  Eigen::MatrixXf m_a;
  Eigen::MatrixXf m_z;
  Eigen::MatrixXf m_dC_dw;
  Eigen::MatrixXf m_dC_db;
  Eigen::MatrixXf m_dC_da_prev;

  int m_number_input_channel;
  int m_number_output_channel;

  std::unique_ptr<ActivationFunction> m_activation_function;
  std::unique_ptr<RandomInitialization> m_random_initialization;
};
