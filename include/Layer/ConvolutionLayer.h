#include "Layer.h"

#include "ActivationFunction/ActivationFunction.h"
#include "OptimizationFunction/OptimizationFunction.h"
#include "RandomInitialization/RandomInitialization.h"

#include "spdlog/spdlog.h"

#include <vector>
#include <memory>

/**
 * Convolution layer.
 * Implements the convolution of a filter with the input values.
 */
class ConvolutionLayer : public Layer {
public:
    /**
     * The standard constructor for adding this layer to the network.
     *
     * The filters of the convolution layer are the only custom matrix, that is stored "transposed".
     * @param input_height The height of the input.
     * @param input_width The width of the input.
     * @param input_channels The number of channels of the input.
     * @param filter_height The height of the filter.
     * @param filter_width The width of the filter.
     * @param output_channels The number of output/filter channels.
     * @param stride The size of the stride of the filter.
     * @param padding The padding of the input. (NOT IMPLEMENTED)
     * @param activation_function The activation function for the layer.
     * @param random_initialization The random initialization function for the layer.
     * @param optimization_function The optimization function for the layer.
     */
    ConvolutionLayer(
            int input_height, int input_width, int input_channels, int filter_height,
            int filter_width, int output_channels, int stride, int padding,
            std::unique_ptr<ActivationFunction> activation_function,
            std::unique_ptr<RandomInitialization> random_initialization,
            std::unique_ptr<OptimizationFunction> optimization_function);

    /**
     * [See abstract base class](@ref Layer)
     */
    void feed_forward(const Eigen::MatrixXf &input) override;

    /**
     * Backpropagation operation of the convolution layer.
     *
     * The operation is implemented using im2col for faster computation.
     * Instead of a classic convolution, the matrices are reshaped and a matrix multiplication is performed.
     * dC_dw=conv(a_prev, (dC_da coeffmult* da_dz)
     * da_dz: derivative of activation function of layer
     * @param a_prev Feed forward result of the previous layer
     * @param dC_da Derivative of the next layer
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
     * Transforms input matrix to a im2col matrix for matrix multiplication with he kernel.
     *
     * The two implementations im2col and im2col_switched differ in the position of the channels and samples.
     * In im2col the image channels are placed downwards and the samples to the right.
     * In im2col_switched it is the other way around. The image channels are to the right and the samples are placed downwards.
     * This determines which dimension (sample or channel) gets summed up in the convolution operation.
     * @param input_matrix The matrix to transform
     */
    [[nodiscard]] std::unique_ptr<Eigen::MatrixXf>
    im2col(const Eigen::MatrixXf &input_matrix, int img_height, int img_width,
           int number_img_channels, int filter_height, int filter_width,
           int stride, int padding) const;

    /**
 * [See other im2col](@ref ConvolutionLayer)
 */
    [[nodiscard]] std::unique_ptr<Eigen::MatrixXf>
    im2col_switched(const Eigen::MatrixXf &input_matrix, int img_height, int img_width,
                    int number_img_channels, int filter_height, int filter_width,
                    int stride, int padding) const;

    /**
     * Returns the number of possible filter positions in the width.
     * @return The number of filter positions
     */
    [[nodiscard]] int row_filter_positions(int img_width, int filter_width,
                                           int stride, int padding) const;

    /**
 * Returns the number of possible filter positions in the height.
 * @return The number of filter positions
 */
    [[nodiscard]] int col_filter_positions(int img_height, int filter_height,
                                           int stride, int padding) const;

    /**
     * Reshapes the computed im2col matrix to the correct format.
     *
     * The im2col matrix has a row per output channel followed by the rows of the
     * next sample. Therefore the matrix has to be transposed and the output
     * channel columns have to be put in their corresponding sample column.
     * The normal matrix format is restored.
     * @param num_samples The number of samples in the input.
     */
    [[nodiscard]] std::unique_ptr<Eigen::MatrixXf>
    reshape_im2col_result(const Eigen::MatrixXf &input, int input_height,
                          int input_width, int filter_height, int filter_width,
                          int number_output_channels, int stride, int padding,
                          long num_samples) const;

    /**
* [See abstract base class](@ref Layer)
*/
    void set_weights(const Eigen::MatrixXf &new_filter) override;

    /**
* [See abstract base class](@ref Layer)
*/
    void set_bias(const Eigen::VectorXf &new_bias) override;

    /**
* [See abstract base class](@ref Layer)
*/
    [[nodiscard]] const Eigen::VectorXf &get_bias() const override;

    /**
* [See abstract base class](@ref Layer)
*/
    [[nodiscard]] const Eigen::MatrixXf &get_weights() const override;

    /**
     * A helper function for tests.
     */
    [[nodiscard]] const Eigen::VectorXf &get_bias_derivative() const;

    /**
 * A helper function for tests.
 */
    [[nodiscard]] const Eigen::MatrixXf &get_weights_derivative() const;

    /**
 * A helper function for tests.
 */
    [[nodiscard]] const Eigen::MatrixXf &get_input_derivative() const;

    /**
* [See abstract base class](@ref Layer)
*/
    [[nodiscard]] int get_number_inputs() const override;

    /**
* [See abstract base class](@ref Layer)
*/
    [[nodiscard]] int get_number_outputs() const override;

    /**
     * Performs the backpropagation in respect to the bias.
     *
     * Compute backpropagation of bias by summing over the channels of all samples and normalizing by dividing by the number of samples.
     * @param dC_dz dC/dz
     */
    void backpropagate_bias(const Eigen::MatrixXf &dC_dz);

    /**
     * Performs the backproagation in respect to the weights.
     * @param a_prev a_prev
     * @param dC_dz dC/dz
     * @param dC_dz_height dC/dz height
     * @param dC_dz_width dC/dz width
     */
    void backpropagate_weights(const Eigen::MatrixXf &a_prev,
                               const Eigen::MatrixXf &dC_dz, int dC_dz_height,
                               int dC_dz_width);

    /**
     * Performs the backpropagation in respect to the input.
     * @param dC_dz dC/dz
     * @param dC_dz_height dC/dz height
     * @param dC_dz_width dC/dz width
     */
    void backpropagate_input(const Eigen::MatrixXf &dC_dz, int dC_dz_height,
                             int dC_dz_width);

    /**
     * Adds zero padding to the input matrix.
     *
     * Generates a matrix with +2*padding cols and rows and copies the old matrix
     * in the middle. Because a tensor is embedded in the matrix, the padding is a
     * bit more complex.
     * new_size=((height+2*padding)*(width+2*padding))*number_channels
     * @param input The input matrix that should be padded.
     * @param padding The size of the padding. A ring of 0 would be a padding of
     * size 1.
     * @return A pointer to the padded new matrix.
     */
    [[nodiscard]] std::unique_ptr<Eigen::MatrixXf>
    pad_matrix(const Eigen::MatrixXf &input, int img_height, int img_width,
               int number_channels, int padding) const;

    /**
     * Flips the filter colwise 180 and rowwise 180.
     * This operation is required to backpropagate the inputs.
     * @return The flipped filter.
     */
    [[nodiscard]] std::unique_ptr<Eigen::MatrixXf> flip_filter() const;

    /**
     * Dilates the matrix for the backpropagation.
     *
     * This is required for filter with stride bigger than 1.
     * Depending on the number of dilations 0 are added between rows and columns of the input matrix.
     * @param input The matrix to dilate.
     * @param img_height The image height encoded in the input.
     * @param img_width The image width encoded in the input.
     * @param img_channels The number of image channels encoded in the input.
     * @param dilation The number of dilation elements to add between rows and columns.
     * @return A pointer to the dilated matrix.
     */
    [[nodiscard]] std::unique_ptr<Eigen::MatrixXf>
    dilate_matrix(const Eigen::MatrixXf &input, int img_height, int img_width,
                  int img_channels, int dilation) const;

private:
    Eigen::MatrixXf m_w;
    Eigen::VectorXf m_b;
    Eigen::MatrixXf m_a;

    Eigen::MatrixXf m_dC_dw;
    Eigen::VectorXf m_dC_db;
    Eigen::MatrixXf m_dC_da_prev;

    int m_input_height;
    int m_input_width;
    int m_input_channels;
    int m_filter_height;
    int m_filter_width;
    int m_output_channels;
    int m_output_width;
    int m_output_height;
    int m_number_output_values;

    int m_stride;
    int m_padding;

    std::unique_ptr<ActivationFunction> m_activation_function;
    std::unique_ptr<RandomInitialization> m_random_initialization;
    std::unique_ptr<OptimizationFunction> m_optimization_function;
    std::shared_ptr<spdlog::logger> m_convlayer_logger;
};