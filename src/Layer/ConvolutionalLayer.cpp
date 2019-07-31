//
// Created by michaelb on 23/06/2019.
//

#include "Layer/ConvolutionalLayer.h"
#include <iostream>

#include "omp.h"

#include "HelperFunctions.h"

const Eigen::MatrixXf &ConvolutionalLayer::get_forward_output() const {
    return m_a;
}

const Eigen::MatrixXf &ConvolutionalLayer::get_backward_output() const {
    return m_dC_da_prev;
}

void ConvolutionalLayer::initialize_parameter() {
    m_b.setZero();
    m_random_initialization->initialize(m_w);
    //m_w <<   0.680375,  -0.211234,   0.566198, 0.59688, 0.823295,-0.604897,  -0.329554  ,0.536459,-0.444451 ,0.10794 ,  -0.0452059,0.257742,  -0.270431, 0.0268018,   0.904459,0.83239;
    //m_w << 1,1,1,0,0,0,0,1,1,0,1,1,0,1,-1,1,0,0,0,1,-1,0,0,1;
}

void ConvolutionalLayer::update_parameter() {
    //m_w = m_w - 0.1 * m_dC_dw;
    //m_b = m_b - 0.1 * m_dC_db;
    m_optimization_function->optimize_weights(m_w,m_dC_dw);
    m_optimization_function->optimize_bias(m_b,m_dC_db);
}

//filters are stored transposed
ConvolutionalLayer::ConvolutionalLayer(int input_height, int input_width, int input_channels, int filter_height,
                                       int filter_width,
                                       int output_channels, int stride, int padding,
                                       std::unique_ptr<ActivationFunction> activation_function,
                                       std::unique_ptr<RandomInitialization> random_initialization,
                                       std::unique_ptr<OptimizationFunction> optimization_function)
        : m_input_height(input_height), m_input_width(input_width),
          m_input_channels(input_channels),
          m_filter_height(filter_height), m_filter_width(filter_width),
          m_output_channels(output_channels), m_stride(stride), m_padding(padding),
          m_activation_function(std::move(activation_function)),
          m_random_initialization(std::move(random_initialization)),
          m_optimization_function(std::move(optimization_function)){

    m_convlayer_logger = spdlog::get("convlayer");
    m_convlayer_logger->info("Start initialization of convlayer");

    if (m_filter_height!=m_filter_width) {
       m_convlayer_logger->error("Filter is not quadratic. Non quadratic filter are currently not supported.");
       m_convlayer_logger->flush();
       throw std::invalid_argument("Filter is not quadratic. Abort");
    }
    if (m_filter_height> m_input_height+2*padding || m_filter_width > m_input_width+2*padding) {
        m_convlayer_logger->error("One filter dimension is bigger than the input + padding dimension.");
        m_convlayer_logger->flush();
        throw std::invalid_argument("Filter dimension is bigger than input + padding dimension. Abort");
    }
    if (m_padding <0) {
        m_convlayer_logger->error("Negative padding is not allowed");
        m_convlayer_logger->flush();
        throw std::invalid_argument("Negative padding is not allowed. Abort");
    }
    if (m_stride <1) {
        m_convlayer_logger->error("A stride smaller than 1 is not allowed");
        m_convlayer_logger->flush();
        throw std::invalid_argument("A stride smaller than 1 is not allowed. Abort");
    }
    if (m_input_height < 1 || m_input_width<1 || m_filter_height < 1 || m_filter_width < 1 || m_input_channels < 1 || m_output_channels < 1) {
        m_convlayer_logger->error("Input and filter dimensions have to be greater than 0");
        m_convlayer_logger->flush();
        throw std::invalid_argument("Input and filter dimensions have to be greater than 0. Abort");
    }
    if (m_padding >0) {
        m_convlayer_logger->error("Padding is currently not supported");
        m_convlayer_logger->flush();
        throw std::invalid_argument("Padding is currently not supported. Abort");
    }

    m_output_height = row_filter_positions(m_input_width, m_filter_width, m_stride, m_padding);
    m_output_width = col_filter_positions(m_input_height, m_filter_height, m_stride, m_padding);
    m_output_size = m_output_height * m_output_width;
    m_number_output_values = m_output_size * m_output_channels;

    m_input_size=m_input_height*m_input_width;

    m_w.resize(m_output_channels, m_filter_height * m_filter_width * m_input_channels);

    m_dC_dw.resize(m_w.rows(), m_w.cols());

    m_b.resize(m_output_channels);

    m_dC_db.resize(m_b.size());

    initialize_parameter();
    //std::cout << "init_Randos\n" << m_w << std::endl;
    m_convlayer_logger->info("End initialization of convlayer");
}

void ConvolutionalLayer::feed_forward(const Eigen::MatrixXf &input) {
    m_convlayer_logger->info("Feed forward convolution");
    assert(input.rows()==m_input_size*m_input_channels && "Feed forward input dimension doesn't match defined one");

    auto im2col_input = im2col(input, m_input_height, m_input_width, m_input_channels, m_filter_height,
                               m_filter_width, m_stride, m_padding);
    //std::cout << "im2col\n" << *im2col_input << std::endl;
    //std::cout << HelperFunctions::print_tensor(input,m_input_height,m_input_width,m_input_channels);

    //std::cout << "m_w\n" << HelperFunctions::print_tensor(m_w.transpose(),m_filter_height,m_filter_width,m_input_channels) << "\nRaw\n" << m_w << std::endl;

    //std::cout << "empt\n" << temp << std::endl;

    m_a = *reshape_im2col_result((m_w * (*im2col_input)).transpose(), m_input_height, m_input_width, m_filter_height, m_filter_width,
                                 m_output_channels, m_stride, m_padding, input.cols());

    //std::cout << "Reshaped:\n" << m_z << std::endl;

    //Add corresponding bias to all output channels
    for (int i =0; i < m_output_channels; i++) {
        m_a.block(i * m_output_size, 0, m_output_size, m_a.cols()).array()+=m_b(i);
    }

    //Apply activation function
    m_activation_function->apply_function(m_a);

    //std::cout << "Conv Feed\n" << "\ninput:\n" << input << "\nOutput\n" << m_a<< std::endl;
}

void ConvolutionalLayer::backpropagation(const Eigen::MatrixXf &a_prev,
                                         const Eigen::MatrixXf &dC_da) {
    m_convlayer_logger->info("Begin backpropagation");
    assert(dC_da.rows()==m_number_output_values && "Backpropagation dC_da input dimension doesn't match defined one");
    //std::cout << "a_prev: " << HelperFunctions::print_tensor(a_prev, m_input_height,m_input_width,m_input_channels) << std::endl;
    //std::cout << "dC_da: " << HelperFunctions::print_tensor(dC_da, m_output_height,m_output_width,m_output_channels) << std::endl;

    //Compute intermidiate result dC/dz=(da/dz)*(dC/da)
    auto dC_dz = m_activation_function->apply_derivative(m_a, dC_da);

    //std::cout << "dC_dz: " << HelperFunctions::print_tensor(dC_dz, m_output_height,m_output_width,m_output_channels) << std::endl;

    backpropagate_bias(dC_dz);

    auto dilation = m_stride-1;

    auto dC_dz_height=m_output_height;
    auto dC_dz_width=m_output_width;

    if (dilation>0) {
        auto dC_dz_dilated=dilate_matrix(dC_dz,dC_dz_height,dC_dz_width,m_output_channels,dilation);
        dC_dz_height+=dilation*(dC_dz_height-1);
        dC_dz_width+=dilation*(dC_dz_width-1);
        backpropagate_weights(a_prev, *dC_dz_dilated, dC_dz_height, dC_dz_width);

        backpropagate_input(*dC_dz_dilated, dC_dz_height, dC_dz_width);
    }
    else {
        backpropagate_weights(a_prev, dC_dz, dC_dz_height, dC_dz_width);

        backpropagate_input(dC_dz, dC_dz_height, dC_dz_width);
    }
    //std::cout << "Conv Backprop: a_prev:\n" << a_prev << "\ndC_da\n" << dC_da << "\nm_dC_dw\n" << m_dC_dw << "\nm_dC_da_prev\n" << m_dC_da_prev << "\ndC_db\n" << m_dC_db << std::endl;
    m_convlayer_logger->info("End backpropagation");
}

void ConvolutionalLayer::backpropagate_bias(const Eigen::MatrixXf &dC_dz) {
    // Compute backpropagation of bias by summing over the channels of all samples and normalizing by dividing by the number of samples
    for (int i=0; i < m_output_channels; i++) {
        m_dC_db(i)=(float) (dC_dz.block(m_output_size * i, 0, m_output_size, dC_dz.cols()).sum() / ((double)dC_dz.cols()));
    }
}

void
ConvolutionalLayer::backpropagate_weights(const Eigen::MatrixXf &a_prev, const Eigen::MatrixXf &dC_dz, int dC_dz_height,
                                          int dC_dz_width) {
    assert(dC_dz.rows()==dC_dz_height*dC_dz_width*m_output_channels && "dC_dz dimensions do not match the one specified by the parameters");
    //std::cout << "a_prev\n" << a_prev << std::endl;
    //std::cout << "dC_dz\n" << dC_dz << std::endl;
    //auto dC_dz_im2col=im2col(dC_dz, m_output_height, m_output_width, m_output_channels, m_output_height, m_output_width, m_stride, m_padding);
    //std::cout << "dC_dz_im2col=\n" << *dC_dz_im2col << std::endl;
    //auto a_prev_im2col=im2col(a_prev,m_input_height,m_input_width,m_input_channels,m_output_height,m_output_width,m_stride,m_padding);
    //std::cout << "a_prev_im2col\n" << *a_prev_im2col << std::endl;

    auto dC_dz_im2col2=im2col2(dC_dz, dC_dz_height, dC_dz_width, m_output_channels, dC_dz_height, dC_dz_width, 1, m_padding);
    //std::cout << "dC_dz_im2col2=\n" << *dC_dz_im2col2 << std::endl;
    auto a_prev_im2col2=im2col2(a_prev, m_input_height, m_input_width, m_input_channels, dC_dz_height, dC_dz_width, 1, m_padding);
    //std::cout << "a_prev_im2col2\n" << *a_prev_im2col2 << std::endl;

    //dC/dw=Conv(a_prev,dC/dz) and normalization

    //std::cout << "res\n" << res << std::endl;
    m_dC_dw.noalias()=((a_prev_im2col2->transpose()*(*dC_dz_im2col2))/dC_dz.cols()).transpose();
    //m_dC_dw.noalias()=*reshape_im2col_result(res,m_input_height,m_input_width,m_output_height,m_output_width,m_output_channels,m_stride,m_padding,m_output_channels);
    //std::cout << "m_dC_dw\n" << m_dC_dw << std::endl;
    //TODO Fix weights test
}

void ConvolutionalLayer::backpropagate_input(const Eigen::MatrixXf &dC_dz, int dC_dz_height, int dC_dz_width) {
    auto flipped_filter=im2col2(flip_filter()->transpose(), m_filter_height, m_filter_width, m_input_channels, m_filter_height, m_filter_width, m_stride, m_padding);
    assert(dC_dz.rows()==dC_dz_height*dC_dz_width*m_output_channels && "dC_dz dimensions do not match the one specified by the parameters");
    //std::cout << "Filter\n" << m_w << std::endl;
    //std::cout << "Flipped filter\n" << *flipped_filter << std::endl;
    //std::cout << HelperFunctions::print_tensor(flipped_filter->transpose(),2,2,1) << std::endl;

    auto padding=m_filter_height-1;

    auto dC_dz_padded= pad_matrix(dC_dz, dC_dz_height, dC_dz_width , m_output_channels, padding);

    auto dC_dz_padded_height=dC_dz_height+2*padding;
    auto dC_dz_padded_width=dC_dz_width+2*padding;

    //std::cout << "undPadded Matrix\n" << dC_dz << std::endl;
    //std::cout << "Padded Matrix\n" << *dC_dz_padded << std::endl;
    //std::cout << HelperFunctions::print_tensor(*dC_dz_padded,m_output_height+2*padding,m_output_width+2*padding,m_output_channels) << std::endl;

    auto dC_dz_padded_im2col=im2col(*dC_dz_padded, dC_dz_padded_height, dC_dz_padded_width, m_output_channels, m_filter_height, m_filter_width, 1, m_padding);
    //std::cout << "im2col Matrix\n" << *dC_dz_padded_im2col << std::endl;
    //std::cout << HelperFunctions::print_tensor(*dC_dz_padded,m_output_height+2*m_padding,m_output_width+2*m_padding,m_output_channels) << std::endl;

    m_dC_da_prev.noalias()=*reshape_im2col_result(dC_dz_padded_im2col->transpose()*(*flipped_filter), dC_dz_padded_height, dC_dz_padded_width, m_filter_height, m_filter_width, m_input_channels, 1, m_padding, dC_dz.cols());
    //std::cout << "res\n" << m_dC_da_prev << std::endl;
    //std::cout << HelperFunctions::print_tensor_comma(m_dC_da_prev);
    //std::cout << HelperFunctions::print_tensor(m_dC_da_prev,m_input_height,m_input_width,m_input_channels);
}


//TODO im2col without segments
std::unique_ptr<Eigen::MatrixXf>
ConvolutionalLayer::im2col(const Eigen::MatrixXf &input_matrix, int img_height, int img_width,
                           int number_img_channels, int filter_height, int filter_width, int stride,
                           int padding) const {
    m_convlayer_logger->info("Start im2col");
    m_convlayer_logger->debug(
            "{} size input matrix; {} rows input matrix; {} cols input matrix; {} filter height; {} filter width",
            input_matrix.size(), input_matrix.rows(), input_matrix.cols(), filter_height, filter_width);
    assert(input_matrix.rows()==img_height*img_width*number_img_channels && "Input dimensions and parameters do not match");

    auto num_row_filter_positions = row_filter_positions(img_width, filter_width, stride, padding);
    auto num_col_filter_positions = col_filter_positions(img_height, filter_height, stride, padding);
    auto num_filter_positions = num_col_filter_positions * num_row_filter_positions;

    auto im2col_matrix = std::make_unique<Eigen::MatrixXf>(filter_width * filter_height * number_img_channels,
                                                           num_filter_positions * input_matrix.cols());

    int filter_size = filter_width * filter_height;

    #pragma omp parallel for
    for (long s = 0; s < input_matrix.cols(); s++) {
        for (int k = 0; k < number_img_channels; k++) {
            for (int j = 0; j < num_row_filter_positions; j++) {
                for (int i = 0; i < num_col_filter_positions; i++) {
                    for (int m = 0; m < filter_width; m++) {
                        auto col = input_matrix.col(s);
                        auto segment = col.segment(
                                img_height * m + i * stride + img_height * img_width * k +
                                img_height * j * stride,
                                filter_height);
                        im2col_matrix->col(i + j * num_col_filter_positions +
                                           s * num_filter_positions).segment(
                                m * filter_height + k * filter_size, filter_height) = segment;
                    }
                }
            }
        }
    }
    m_convlayer_logger->debug("{} size output matrix; {} rows output matrix; {} cols output matrix",
                              im2col_matrix->size(), im2col_matrix->rows(), im2col_matrix->rows());
    m_convlayer_logger->info("End im2col");

    return im2col_matrix;
}

//switched number_img_channel and sample variables
std::unique_ptr<Eigen::MatrixXf>
ConvolutionalLayer::im2col2(const Eigen::MatrixXf &input_matrix, int img_height, int img_width,
                           int number_img_channels, int filter_height, int filter_width, int stride,
                           int padding) const {
    m_convlayer_logger->info("Start im2col");
    m_convlayer_logger->debug(
            "{} size input matrix; {} rows input matrix; {} cols input matrix; {} filter height; {} filter width",
            input_matrix.size(), input_matrix.rows(), input_matrix.cols(), filter_height, filter_width);
    assert(input_matrix.rows()==img_height*img_width*number_img_channels && "Input dimensions and parameters do not match");

    auto num_row_filter_positions = row_filter_positions(img_width, filter_width, stride, padding);
    auto num_col_filter_positions = col_filter_positions(img_height, filter_height, stride, padding);
    auto num_filter_positions = num_col_filter_positions * num_row_filter_positions;

    auto im2col_matrix = std::make_unique<Eigen::MatrixXf>(filter_width * filter_height * input_matrix.cols(),
                                                           num_filter_positions * number_img_channels);

    int filter_size = filter_width * filter_height;

    #pragma omp parallel for
    for (long s = 0; s < input_matrix.cols(); s++) {
        for (int k = 0; k < number_img_channels; k++) {
            for (int j = 0; j < num_row_filter_positions; j++) {
                for (int i = 0; i < num_col_filter_positions; i++) {
                    for (int m = 0; m < filter_width; m++) {
                        auto col = input_matrix.col(s);
                        auto segment = col.segment(
                                img_height * m + i * stride + img_height * img_width * k +
                                img_height * j * stride,
                                filter_height);
                        im2col_matrix->col(i + j * num_col_filter_positions +
                                           k * num_filter_positions).segment(
                                m * filter_height + s * filter_size, filter_height) = segment;
                    }
                }
            }
        }
    }
    m_convlayer_logger->debug("{} size output matrix; {} rows output matrix; {} cols output matrix",
                              im2col_matrix->size(), im2col_matrix->rows(), im2col_matrix->rows());
    m_convlayer_logger->info("End im2col");

    return im2col_matrix;
}

int ConvolutionalLayer::row_filter_positions(int img_width, int filter_width, int stride, int padding) const {
    return (img_width - filter_width + 2 * padding) / stride + 1;
}

int ConvolutionalLayer::col_filter_positions(int img_height, int filter_height, int stride, int padding) const {
    return (img_height - filter_height + 2 * padding) / stride + 1;
}

//const Eigen::MatrixXf &ConvolutionalLayer::get_m_z() const {
//    return Eigen::MatrixXf::Zero(1,1);
//}

//TODO Try resize instead
std::unique_ptr<Eigen::MatrixXf>
ConvolutionalLayer::reshape_im2col_result(const Eigen::MatrixXf &input, int input_height, int input_width,
                                          int filter_height, int filter_width, int number_output_channels, int stride,
                                          int padding, long num_samples) const {
    m_convlayer_logger->debug("Start reshape forward propagation");

    auto output_img_height = row_filter_positions(input_width, filter_width, stride, padding);
    auto output_img_width = col_filter_positions(input_height, filter_height, stride, padding);
    auto output_img_size = output_img_height * output_img_width;
    auto num_output_values = output_img_size * number_output_channels;

    auto im2col_reshaped = std::make_unique<Eigen::MatrixXf>(num_output_values, num_samples);
    m_convlayer_logger->debug("Dimensions reshaped im2col: {} {}", num_output_values, num_samples);

   #pragma omp parallel for
    for (long i = 0; i < num_samples; i++) {
        for (int j = 0; j < number_output_channels; j++) {
            im2col_reshaped->col(i).segment(output_img_size * j, output_img_size) = input.col(j).segment(
                        output_img_size * i, output_img_size);
        }
    }
    m_convlayer_logger->debug("End reshape forward propagation");
    return im2col_reshaped;
}



std::unique_ptr<Eigen::MatrixXf>
ConvolutionalLayer::pad_matrix(const Eigen::MatrixXf &input, int img_height, int img_width, int number_channels,
                               int padding) const {
    m_convlayer_logger->info("Start padding matrix");
    m_convlayer_logger->debug("{} size input matrix; {} rows input matrix; {} cols input matrix; {} img height; {} img width; {} number_channels; {} size padding", input.size(), input.rows(), input.cols(), img_height,img_width,number_channels,padding);
    assert(input.rows()==img_height*img_width*number_channels && "Input dimensions and parameters do not match");
    auto img_size=img_height*img_width;
    auto img_height_padded=img_height+2*padding;
    auto img_width_padded=img_width+2*padding;
    auto img_size_padded=img_height_padded*img_width_padded;

    // new_size=((height+2*padding)*(width+2*padding))*number_channels
    auto input_padded=std::make_unique<Eigen::MatrixXf>(img_size_padded * number_channels, input.cols());

    m_convlayer_logger->debug("{} size output matrix; {} rows output matrix; {} cols output matrix", input_padded->size(), input_padded->rows(), input_padded->cols());

    input_padded->setZero();

    #pragma omp parallel for
    for (int i=0; i<number_channels; i++) {
        for (int j=0; j<img_width; j++) {
            // Added rows of padding + skip already processed img + skip already processed cols + skip first pads of rows
            input_padded->block(img_height_padded * padding + img_size_padded * i + img_height_padded * j + padding, 0, img_height, input.cols())=input.block(img_size * i + img_height * j, 0, img_height, input.cols());
        }
    }
    m_convlayer_logger->info("Finished padding matrix");
    return input_padded;
}

// For padding of a backpropagation matrix with stride (https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-8137e4fc2710)
// dilation factor is stride -1
std::unique_ptr<Eigen::MatrixXf>
ConvolutionalLayer::dilate_matrix(const Eigen::MatrixXf &input, int img_height, int img_width, int img_channels,
                                  int dilation) const {
    assert(input.rows()==img_height*img_width*img_channels && "Input dimensions and parameters do not match");
    auto img_size=img_height*img_width;
    auto img_height_dilated=img_height+dilation*(img_height-1);
    auto img_width_dilated=img_width+dilation*(img_width-1);
    auto img_size_dilated=img_height_dilated*img_width_dilated;
    auto input_padded=std::make_unique<Eigen::MatrixXf>(img_size_dilated * img_channels, input.cols());
    input_padded->setZero();

    #pragma omp parallel for
    for (long l=0;l<input.cols(); l++) {
        for (int i = 0; i < img_channels; i++) {
            for (int j = 0; j < img_width; j++) {
                for (int k = 0; k < img_height; k++) {
                    input_padded->operator()(i * img_size_dilated + j * img_height_dilated*(dilation+1) + k * (dilation + 1),l) = input(i*img_size+j*img_height+k,l);
                }
            }
        }
    }
    return input_padded;


    return std::unique_ptr<Eigen::MatrixXf>();
}

std::unique_ptr<Eigen::MatrixXf> ConvolutionalLayer::flip_filter() const {
    m_convlayer_logger->info("Start flipping filter");
    auto flipped_filter=std::make_unique<Eigen::MatrixXf>(m_w.rows(), m_w.cols());
    for (int i=0; i < m_input_channels; i++) {
        flipped_filter->block(0,i*m_filter_height*m_filter_width,m_w.rows(),m_filter_height*m_filter_width)=m_w.block(0,i*m_filter_height*m_filter_width,m_w.rows(),m_filter_height*m_filter_width).rowwise().reverse();
    }
    m_convlayer_logger->info("Finished flipping filter");
    return flipped_filter;
}

void
ConvolutionalLayer::set_weights(const Eigen::MatrixXf &new_filter) {
    if (new_filter.rows() != m_w.rows() || new_filter.cols() != m_w.cols()) {
        throw std::invalid_argument("New Filter size does not match the old filter size");
    }
    m_w = new_filter;
}

void ConvolutionalLayer::set_bias(const Eigen::VectorXf &new_bias) {
    if (new_bias.rows() != m_b.rows() || new_bias.cols() != m_b.cols()) {
        throw std::invalid_argument("New bias size does not match the old bias size");
    }
    m_b = new_bias;
}

const Eigen::VectorXf &ConvolutionalLayer::get_bias() const {
    return m_b;
}

const Eigen::VectorXf& ConvolutionalLayer::get_bias_derivative() const {
    return m_dC_db;
}

const Eigen::MatrixXf &ConvolutionalLayer::get_weights() const {
    return m_w;
}

const Eigen::MatrixXf &ConvolutionalLayer::get_weights_derivative() const {
    return m_dC_dw;
}

const Eigen::MatrixXf &ConvolutionalLayer::get_input_derivative() const {
    return m_dC_da_prev;
}

int ConvolutionalLayer::get_number_inputs() const {
    return m_input_height * m_input_width * m_input_channels;
}

int ConvolutionalLayer::get_number_outputs() const {
    return m_number_output_values;
}

