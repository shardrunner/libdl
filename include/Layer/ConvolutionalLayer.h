#include "BaseLayer.h"

#include "ActivationFunction/ActivationFunction.h"
#include "RandomInitialization/RandomInitialization.h"
#include <vector>

#include <memory>
#include "spdlog/spdlog.h"

class ConvolutionalLayer : public BaseLayer {
public:
  ConvolutionalLayer(int input_height, int input_width, int number_input_channel,
      int number_output_channel, int filter_heigth, int filter_width, int stride,
      std::unique_ptr<ActivationFunction> activation_function,
      std::unique_ptr<RandomInitialization> random_initialization);
  void feed_forward(const Eigen::MatrixXf &input) override;
  void backpropagation(const Eigen::MatrixXf &a_prev,
                       const Eigen::MatrixXf &dC_da) override;
  const Eigen::MatrixXf &get_forward_output() override;
  const Eigen::MatrixXf &get_backward_output() override;
  void initialize_parameter() override;
  void update_parameter() override;

public:
  Eigen::MatrixXf m_w;
  Eigen::VectorXf m_b;
  Eigen::MatrixXf m_a;
  Eigen::MatrixXf m_z;
  Eigen::MatrixXf m_dC_dw;
  Eigen::VectorXf m_dC_db;
  Eigen::MatrixXf m_dC_da_prev;

  int output_values;

  int m_number_input_channel;
  int m_number_output_channel;
  int m_input_width;
  int m_input_height;
  int m_filter_width;
  int m_filter_height;
  int output_img_width;
  int output_img_height;
  int m_stride;

  std::unique_ptr<ActivationFunction> m_activation_function;
  std::unique_ptr<RandomInitialization> m_random_initialization;
private:
    std::shared_ptr<spdlog::logger> m_convlayer_logger;
};