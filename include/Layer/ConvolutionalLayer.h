#include "BaseLayer.h"

#include "ActivationFunction/ActivationFunction.h"
#include "RandomInitialization/RandomInitialization.h"
#include <vector>

#include <memory>
#include "spdlog/spdlog.h"

class ConvolutionalLayer : public BaseLayer {
public:
    ConvolutionalLayer(int input_height, int input_width, int number_input_channel, int number_output_channel,
                       int filter_heigth, int filter_width, int stride,
                       std::unique_ptr<ActivationFunction> activation_function,
                       std::unique_ptr<RandomInitialization> random_initialization);

    void feed_forward(const Eigen::MatrixXf &input) override;

    void backpropagation(const Eigen::MatrixXf &a_prev,
                         const Eigen::MatrixXf &dC_da) override;

    const Eigen::MatrixXf &get_forward_output() override;

    const Eigen::MatrixXf &get_backward_output() override;

    void initialize_parameter() override;

    void update_parameter() override;


    /**
     * Transforms input matrix to a im2col matrix for matrix multiplication with the kernel and stores it internally
     * @param input_matrix The matrix to transform
     */
    void im2col(const Eigen::MatrixXf &input_matrix);

    /**
     * Returns the number of filter positions.
     * @return The number of filter positions
     */
    [[nodiscard]] int row_filter_positions() const;

    [[nodiscard]] int col_filter_positions() const;

    /**
     * Returns the im2col matrix of the input with the filter size
     * @return The im2col matrix
     */
    [[nodiscard]] const Eigen::MatrixXf &get_im2col_matrix() const;

    /**
     * Reshapes the computed im2col matrix to the correct format.
     * The im2col matrix has a row per output channel followed by the rows of the next sample.
     * Therefore the maxtrix has to be transposed and the output channel columns have to be put in their corresponding sample column.
     * @param num_samples The number of samples in the input.
     */
    void reshape_forward_propagation(const Eigen::MatrixXf &input, int num_samples);

public:
    Eigen::MatrixXf m_w;
    Eigen::VectorXf m_b;
    Eigen::MatrixXf m_a;
    Eigen::MatrixXf m_z;
    Eigen::MatrixXf m_dC_dw;
    Eigen::VectorXf m_dC_db;
    Eigen::MatrixXf m_dC_da_prev;
    Eigen::MatrixXf m_im2col_matrix;
    Eigen::MatrixXf m_im2col_reshaped;

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