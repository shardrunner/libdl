#include <memory>

#include "ActivationFunction/ActivationFunction.h"
#include "Layer.h"
#include "OptimizationFunction/OptimizationFunction.h"
#include "RandomInitialization/RandomInitialization.h"

/**
 * FullyConnected layer.
 * Implements the matrix multiplication of a filter with the input values.
 */
class FullyConnectedLayer : public Layer {
public:
  /**
   * The standard constructor for adding this layer to the network.
   *
   * @param input_size Total size of the input.
   * @param output_size Total size of the output.
   * @param activation_function The activation function for the layer.
   * @param random_initialization The random initialization function for the
   * layer.
   * @param optimization_function The optimization function for the layer.
   */
  FullyConnectedLayer(
      int input_size, int output_size,
      std::unique_ptr<ActivationFunction> activation_function,
      std::unique_ptr<RandomInitialization> random_initialization,
      std::unique_ptr<OptimizationFunction> optimization_function);

  /**
   * [See abstract base class](@ref Layer)
   */
  void feed_forward(const Eigen::MatrixXf &input) override;

  /**
   * Notation from 3blue1brown NeuralNetwork series:
   *
   * Input -> prev layer -> current layer -> next layer -> Loss
   * *
   * C: Loss, w: weights, z: result before activation [z=w*a(prev)+b],
   * a: activated z [sigma(z)], d: derivative, a_prev: a of previous layer
   * dC/dw=(dz/dw)*(da/dz)*(dC/da)=(dz/dw)*(dC/dz) (dC/dw is normalized with the
   * number of samples) dz/dw=a_prev da/dz=derivative activation function with z
   * dC/da=derivative from next layer (computation see intermediate results
   * dC/da(prev))
   *
   * Intermediate results:
   * derivative for previous layer computation (for next step/prev layer in
   * backprop):
   * dC/da_prev=(dz/da_prev)*(da/dz)*(dC/da)]=(dz/da_prev)*(dC/dz)
   * dz/da_prev=w
   *
   * bias backprop:
   * dC/db=(dz/db)*(da/dz)*(dC/da)=(dz/db)*(dC/dz)
   * dz/db=1
   *
   * a_prev: m_input_size x number_training_samples
   * dC/da=next layer derivative: m_output_size x number_training_samples
   */
  void backpropagation(const Eigen::MatrixXf &a_prev,
                       const Eigen::MatrixXf &dC_da) override;

  /**
   * [See abstract base class](@ref Layer)
   */
  [[nodiscard]] const Eigen::MatrixXf &get_forward_output() const override;

  /**
   * [See abstract base class](@ref Layer)
   */
  [[nodiscard]] const Eigen::MatrixXf &get_backward_output() const override;

  /**
   * [See abstract base class](@ref Layer)
   */
  void initialize_parameter() override;

  /**
   * [See abstract base class](@ref Layer)
   */
  void update_parameter() override;

  /**
   * [See abstract base class](@ref Layer)
   */
  void set_weights(const Eigen::MatrixXf &new_weights) override;

  /**
   * [See abstract base class](@ref Layer)
   */
  [[nodiscard]] int get_number_inputs() const override;

  /**
   * [See abstract base class](@ref Layer)
   */
  [[nodiscard]] int get_number_outputs() const override;

  /**
   * [See abstract base class](@ref Layer)
   */
  void set_bias(const Eigen::VectorXf &new_bias) override;

  /**
   * [See abstract base class](@ref Layer)
   */
  [[nodiscard]] const Eigen::MatrixXf &get_weights() const override;

  /**
   * [See abstract base class](@ref Layer)
   */
  [[nodiscard]] const Eigen::VectorXf &get_bias() const override;

private:
  Eigen::MatrixXf m_w;
  Eigen::VectorXf m_b;
  Eigen::MatrixXf m_a;

  Eigen::MatrixXf m_dC_dw;
  Eigen::MatrixXf m_dC_db;
  Eigen::MatrixXf m_dC_da_prev;

  std::unique_ptr<ActivationFunction> m_activation_function;
  std::unique_ptr<RandomInitialization> m_random_initialization;
  std::unique_ptr<OptimizationFunction> m_optimization_function;
};
