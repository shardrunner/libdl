#include "BaseLayer.h"

#include "ActivationFunction/ActivationFunction.h"
#include "RandomInitialization/RandomInitialization.h"
#include <vector>

#include <memory>
#include "spdlog/spdlog.h"

class ConvolutionalLayer : public BaseLayer {
public:
    ConvolutionalLayer(int input_height, int input_width, int number_input_channel,
                       int filter_height, int filter_width, int number_output_channel, int stride,
                       int padding, std::unique_ptr<ActivationFunction> activation_function,
                       std::unique_ptr<RandomInitialization> random_initialization);

    void feed_forward(const Eigen::MatrixXf &input) override;

    void feed_forward_old(const Eigen::MatrixXf &input);

    void backpropagation_old(const Eigen::MatrixXf &a_prev,
                         const Eigen::MatrixXf &dC_da);

    /**
     * Backpropagation operation of the convolution layer.
     * Implemented using im2col for faster computation.
     * dC_dw=conv(a_prev, (dC_da coeffmult* da_dz)
     * a_prev: feed forward result previous layer
     * dC_da: derivative next layer
     * da_dz: derivative of activation function of layer
     * @param a_prev Feed forward result of the previous layer
     * @param dC_da Derivative of the next layer
     */
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
    [[nodiscard]] std::unique_ptr<Eigen::MatrixXf>
    im2col(const Eigen::MatrixXf &input_matrix, int img_height, int img_width,
           int number_img_channels, int filter_height, int filter_width, int stride, int padding) const;

    [[nodiscard]] std::unique_ptr<Eigen::MatrixXf>
    im2col2(const Eigen::MatrixXf &input_matrix, int img_height, int img_width,
           int number_img_channels, int filter_height, int filter_width, int stride, int padding) const;

    /**
     * Returns the number of filter positions.
     * @return The number of filter positions
     */
    [[nodiscard]] int row_filter_positions(int img_width, int filter_width, int stride, int padding) const;

    [[nodiscard]] int col_filter_positions(int img_height, int filter_height, int stride, int padding) const;

    /**
     * Reshapes the computed im2col matrix to the correct format.
     * The im2col matrix has a row per output channel followed by the rows of the next sample.
     * Therefore the maxtrix has to be transposed and the output channel columns have to be put in their corresponding sample column.
     * @param num_samples The number of samples in the input.
     */
    [[nodiscard]] std::unique_ptr<Eigen::MatrixXf>
    reshape_im2col_result(const Eigen::MatrixXf &input, int input_height, int input_width,
                          int filter_height, int filter_width, int number_output_channels, int stride,
                          int padding, long num_samples) const;

    [[nodiscard]] const Eigen::MatrixXf &get_m_z() const;

    [[nodiscard]] const Eigen::MatrixXf &get_dC_dw() const;

    void set_filter(const Eigen::MatrixXf &new_filter);

    void set_bias(const Eigen::VectorXf &new_bias);

    [[nodiscard]] const Eigen::VectorXf& get_bias() const;

    [[nodiscard]] const Eigen::VectorXf& get_bias_derivative() const;

    [[nodiscard]] const Eigen::MatrixXf& get_filter() const;

    const Eigen::MatrixXf& get_weights_derivative() const;

    const Eigen::MatrixXf& get_input_derivative() const;

    void backpropagate_bias(const Eigen::MatrixXf &dC_dz);

    void backpropagate_weights(const Eigen::MatrixXf& a_prev, const Eigen::MatrixXf &dC_dz);

    //void backpropagate_activation(const Eigen::MatrixXf& dC_da);

    void backpropagate_input(const Eigen::MatrixXf& dC_dz);

    /**
     * Add zero padding to the input matrix.
     * Generates a matrix with +2*padding cols and rows and copies the old matrix in the middle.
     * Because a tensor is embedded in the matrix, the padding is a bit more complex.
     * @param input The input matrix that should be padded.
     * @param padding The size of the padding. A ring of 0 would be a padding of size 1.
     * @return A pointer to the padded new matrix.
     */
    [[nodiscard]] std::unique_ptr<Eigen::MatrixXf>
    pad_matrix(const Eigen::MatrixXf &input, int padding, int img_height, int img_width,
               int number_channels) const;

    [[nodiscard]] std::unique_ptr<Eigen::MatrixXf> flip_filter() const;

public:
    Eigen::MatrixXf m_w;
    Eigen::VectorXf m_b;
    Eigen::MatrixXf m_a;
    Eigen::MatrixXf m_z;
    Eigen::MatrixXf m_dC_dw;
    Eigen::VectorXf m_dC_db;
    Eigen::MatrixXf m_dC_da_prev;
    //Eigen::MatrixXf m_im2col_matrix;
    //Eigen::MatrixXf m_im2col_reshaped;

    int m_number_output_values;


    int m_input_height;
    int m_input_width;
    int m_number_input_channels;
    int m_filter_height;
    int m_filter_width;
    int m_number_output_channels;
    int m_output_img_width;
    int m_output_img_height;
    int m_output_img_size;


    int m_stride;
    int m_padding;

    std::unique_ptr<ActivationFunction> m_activation_function;
    std::unique_ptr<RandomInitialization> m_random_initialization;
private:
    std::shared_ptr<spdlog::logger> m_convlayer_logger;
};