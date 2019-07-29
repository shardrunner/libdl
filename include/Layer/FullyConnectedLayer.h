#include <memory>

#include "../RandomInitialization/RandomInitialization.h"
#include "ActivationFunction/ActivationFunction.h"
#include "BaseLayer.h"

class FullyConnectedLayer : public BaseLayer {
public:
  FullyConnectedLayer(
      int input_size, int output_size,
      std::unique_ptr<ActivationFunction> activation_function,
      std::unique_ptr<RandomInitialization> random_initialization);
  void feed_forward(const Eigen::MatrixXf &input) override;
  void backpropagation(const Eigen::MatrixXf &a_prev,
                       const Eigen::MatrixXf &dC_da) override;
  [[nodiscard]] const Eigen::MatrixXf &get_forward_output() const override;
  [[nodiscard]] const Eigen::MatrixXf &get_backward_output() const override;
  void initialize_parameter() override;
  void update_parameter() override;

  void set_weights(const Eigen::MatrixXf &new_weights) override;

    [[nodiscard]] int get_number_inputs() const override;

    [[nodiscard]] int get_number_outputs() const override;

    void set_bias(const Eigen::VectorXf &new_bias) override;

    [[nodiscard]] const Eigen::MatrixXf &get_weights() const override;

    [[nodiscard]] const Eigen::VectorXf &get_bias() const override;

private:
  Eigen::MatrixXf m_w;
  Eigen::VectorXf m_b;
  Eigen::MatrixXf m_a;

private:
//  Eigen::MatrixXf m_z;
  Eigen::MatrixXf m_dC_dw;
  Eigen::MatrixXf m_dC_db;
  Eigen::MatrixXf m_dC_da_prev;

  std::unique_ptr<ActivationFunction> m_activation_function;
  std::unique_ptr<RandomInitialization> m_random_initialization;
};
