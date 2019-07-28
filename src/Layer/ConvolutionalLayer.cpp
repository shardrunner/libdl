//
// Created by michaelb on 23/06/2019.
//

#include "Layer/ConvolutionalLayer.h"
#include <iostream>

#include "HelperFunctions.h"

/*
 * (Input ImgCol x ImgRowx x Channel) x number_samples
 */
void ConvolutionalLayer::feed_forward_old(const Eigen::MatrixXf &input) {
    // std::cout << "input:\n" << input[0] << "\nsize: " <<input.size();
    // std::cout << "rows:" << input[0].rows() << " m_w rows(): " <<m_w.rows();

    // std::stringstream ss;
    // ss << input;
    // temp << input;
    // spdlog::error("Input:\n{}", ss.str());

    m_a.resize(m_number_output_values, input.cols());
    m_z.resize(m_number_output_values, input.cols());
    m_z.setZero();

    // TODO fix multiple input and output channels, for 1x1 the result is correct
    // go through all samples
    for (int k = 0; k < input.cols(); k++) {
        // one filter per output channel
        for (int n = 0; n < m_number_output_channels; n++) {
            // use filter once per input channel
            for (int m = 0; m < m_number_input_channels; m++) {
                // traverse height
                for (int i = 0; i < m_output_img_height; i++) {
                    // traverse width
                    for (int j = 0; j < m_output_img_width; j++) {
                        // for each row of the filter
                        for (int l = 0; l < m_filter_height; l++) {
                            // input column=k
                            // input row =
                            // j+(m_output_img_width*i)+(m_output_img_width*m_output_img_height*m)+(m_output_img_width*m_output_img_height*m_number_input_channels*n)
                            int row_input =
                                    j + (m_input_width * (i + l)) +
                                    (m_input_width * m_input_height * m); // input channels
                            int row_output =
                                    j + (m_output_img_width * i) +
                                    (m_output_img_width * m_output_img_height * n); // output channels
                            // take a block with length of a filter column apply one column of
                            // filter
                            int filter_channel =
                                    m_filter_width * l + (m_filter_height * m_filter_width) * m +
                                    (m_filter_height * m_filter_width * m_number_input_channels) *
                                    n;

                            auto block1 = input.block(row_input, k, m_filter_width, 1);
                            auto block2 = m_w.block(filter_channel, 0, m_filter_width, 1);

                            std::cout << "row input: " << row_input << " row output "
                                      << row_output << " column " << k << "\nblock1:\n"
                                      << block1 << "\nblock2\n"
                                      << block2 << "\nsum\n"
                                      << m_z(row_output, k) << std::endl;

                            m_z(row_output, k) += (block1.array() * block2.array()).sum();

                            // add bias
                        }
                    }
                }
            }
        }
    }
    // spdlog::debug("before segfault");
    // m_z.emplace_back(Eigen::MatrixXf(input[k].rows() - m_w.rows() +
    // 1,input[k].cols() - m_w.cols() + 1)); spdlog::debug("after segfault");

    // std::cout << "k: " << k << " size m_z: " <<m_z.size() << "\nelem0\n"
    // <<m_z[0] <<std::endl;

    // std::cout << "Grenze: " <<input.cols()-m_w.cols() << std::endl;

    // std::cout << "Input block: \n" <<
    // input.block(i,j,m_w.rows(),m_w.cols()).array() << "\nfilter: \n" << m_w
    // << "\n\n" << "Sum: " <<
    // (input.block(i,j,m_w.rows(),m_w.cols()).array()*m_w.array()).sum()<<
    // std::endl;
    // std::cout << "input\n" << input[k] << std::endl;
    // std::cout << "inputBlock\n" << input[k].block(i, j, m_w.rows(),
    // m_w.cols()) << std::endl; std::cout << "Window\n" << m_w << std::endl;

    // std::cout << "mz: \n" << m_z << std::endl;
    // std::cout << "ma: \n" << m_a << std::endl;

    m_a = m_activation_function->apply_function(m_z);

    std::cout << "Conv Feed Forward: Rando weights\n"
              << m_w << "\ninput:\n"
              << input << "\nOutput\n"
              << m_a << std::endl;
}

void ConvolutionalLayer::backpropagation_old(const Eigen::MatrixXf &a_prev,
                                         const Eigen::MatrixXf &dC_da) {
    m_dC_da_prev.resize(a_prev.rows(), a_prev.cols());
    m_dC_dw.setZero();
    m_dC_da_prev.setZero();

    // derivative activation
    // std::cout << "m_a :\n" <<m_a << "\ndC_da: \n" << dC_da << std::endl;
    // std::cout << "Der Activation:\n" <<
    // m_activation_function->apply_derivative(m_a) << std::endl;
    Eigen::MatrixXf dC_dz = m_activation_function->apply_derivative(m_a, dC_da);

    // for (int i =0; i<a_prev.cols()-m_w.cols(); i++) {
    //  for (int j=0; j<a_prev.rows()-m_w.rows(); j++) {

    // std::cout << "a_prev:\n" << a_prev << "\ndC_dz\n" << dC_dz << std::endl;
    // go through all samples
    for (long k = 0; k < a_prev.cols(); k++) {
        // one filter per output channel
        for (int n = 0; n < m_number_output_channels; n++) {
            // use filter once per input channel
            for (int m = 0; m < m_number_input_channels; m++) {
                // traverse height
                for (int i = 0; i < m_filter_height; i++) {
                    // traverse width
                    for (int j = 0; j < m_filter_width; j++) {
                        // for each row of the filter
                        for (int l = 0; l < m_output_img_height; l++) {
                            int row_input = j + (m_input_width * (i + l)) +
                                            (m_input_width * m_input_height * m);
                            int row_filter = j + (m_filter_width * i) +
                                             (m_filter_width * m_filter_height * m) +
                                             (m_filter_width * m_filter_height *
                                              m_number_input_channels * n);

                            int filter_channel = m_output_img_width * l +
                                                 (m_output_img_height * m_output_img_width) * n;

                            auto block1 = a_prev.block(row_input, k, m_output_img_width, 1);
                            auto block2 = dC_dz.block(filter_channel, k, m_output_img_width, 1);

                            m_dC_dw(row_filter, 0) += (block1.array() * block2.array()).sum();

                            // std::cout << "row input: " <<row_input << " row filter " <<
                            // row_filter << " column " << k << "\nblock1:\n" << block1 <<
                            // "\nblock2\n" << block2 << "\nsum\n" << m_dC_dw(row_filter,0) <<
                            // std::endl;

                            // add bias
                        }
                    }
                }
            }
        }
    }

    // average it out by dividing by number of samples
    m_dC_dw = m_dC_dw / a_prev.cols();

    /*  for (int k = 0; k < a_prev.size(); k++) {


        for (int i = 0; i < m_w.cols(); i++) {
          for (int j = 0; j < m_w.rows(); j++) {
            m_dC_dw[k](i, j) =
                (a_prev[k].block(i, j, dC_da[k].rows(), dC_da[k].cols()).array() *
                 dC_dz.array())
                    .sum();

            //
       m_dC_dw.block(i,j,m_w.rows(),m_w.cols())=(m_dC_dw.block(i,j,m_w.rows(),m_w.cols()).array()+m_w.array()*dC_da(i,j)).matrix();

            //
       m_dC_dw(i,j)=(input.block(i,j,m_w.rows(),m_w.cols()).array()*m_w.array()).sum()
            // + m_b(0);
          }
        }*/
    // TODO implement for multi in and output channel
    Eigen::MatrixXf filter_flip;
    filter_flip.resize(m_filter_width, m_filter_height);
    // std::cout << "filter\n" <<m_w<<std::endl;
    for (int i = 0; i < m_filter_height; i++) {
        filter_flip.row(i) =
                m_w.block(i * m_filter_width, 0, m_filter_width, 1).transpose();
        // std::cout <<  "\nfilter flip\n" << filter_flip << "\nblock\n"
        // <<m_w.block(i*m_filter_width,0,m_filter_width,1)<<std::endl;
    }
    filter_flip = (filter_flip.colwise().reverse()).eval();
    // std::cout << "After first reverse\n" << filter_flip << std::endl;
    filter_flip = (filter_flip.rowwise().reverse()).eval();

    // std::cout << "flipped filter "<<filter_flip << std::endl;

    // TODO make more pretty
    // construct bigger matrix with zero padding size 2*(m_dC_da_prev_dim-1) for
    // easier full convolution
    Eigen::MatrixXf temp_filter(filter_flip.rows() + 2 * (m_output_img_height - 1),
                                filter_flip.cols() + 2 * (m_output_img_width - 1));
    temp_filter.setZero();

    temp_filter.block(m_output_img_height - 1, m_output_img_width - 1,
                      filter_flip.rows(), filter_flip.cols()) = filter_flip;

    // std::cout << "dC_dz: \n" <<filter_flip << "\ntemp_filter: \n" <<
    // temp_filter << std::endl;

    // std::cout << "base\n" << m_w << "\nstage1:\n" << m_w.colwise().reverse()
    // << "\nflipped filter: \n" <<filter_flip << "\nm_dC_dw: \n" <<m_dC_dw <<
    // std::endl;

    // std::cout << "dC_dz\n" << dC_dz << "\nfilter_temp\n" << temp_filter <<
    // std::endl;//"\nfilter\n" << m_w << std::endl;

    for (long k = 0; k < a_prev.cols(); k++) {
        for (int i = 0; i < m_input_height; i++) {
            for (int j = 0; j < m_input_width; j++) {
                // std::cout << "block\n"
                // <<temp_filter.block(i,j,dC_da.rows(),dC_da.cols())<<"\nWindow\n"
                // <<dC_dz<<"\nsum: "
                // <<(temp_filter.block(i,j,dC_da.rows(),dC_da.cols()).array()*dC_dz.array()).sum()
                // << "\naddress: " << i << "&" <<j << std::endl;
                for (int l = 0; l < m_output_img_height; l++) {
                    int row_dC_da_prev = j + i * m_input_width;
                    int row_dC_dz = l * m_output_img_width;

                    auto block1 =
                            temp_filter.block(i + l, j, 1, m_output_img_width).transpose();
                    auto block2 = dC_dz.block(row_dC_dz, k, m_output_img_width, 1);

                    m_dC_da_prev(m_dC_da_prev.rows() - row_dC_da_prev - 1, k) +=
                            (block1.array() * block2.array()).sum();

                    // std::cout << "row_dC_da_prev " <<row_dC_da_prev << " row dC_dz " <<
                    // row_dC_dz << " column " << k << "\nblock1:\n" << block1.transpose()
                    // << "\nblock2\n" << block2 << "\nsum\n" <<
                    // m_dC_da_prev(row_dC_da_prev, k) << std::endl;
                }

                // m_dC_da_prev[k](i, j) =
                //   (temp_filter.block(i, j, dC_dz.rows(), dC_dz.cols()).array() *
                //   dC_dz.array())
                //    .sum();
            }
        }
    }

    /*std::cout << "Conv Backprop: a_prev:\n"
              << a_prev << "\ndC_da\n"
              << dC_da << "\nm_dC_dw\n"
              << m_dC_dw << "\nm_dC_da_prev\n"
              << m_dC_da_prev << std::endl;*/
    // spdlog::set_level(spdlog::level::warn);
}

const Eigen::MatrixXf &ConvolutionalLayer::get_forward_output() { return m_a; }

const Eigen::MatrixXf &ConvolutionalLayer::get_backward_output() {
    return m_dC_da_prev;
}

void ConvolutionalLayer::initialize_parameter() {
    m_b.setZero();
    m_random_initialization->initialize(m_w.block(0, 0, 1, m_w.cols()));
    //m_convlayer_logger->debug("Initialized weights:\n{}", HelperFunctions::toString(m_w));
}

void ConvolutionalLayer::update_parameter() {
    /*Eigen::VectorXf temp = Eigen::VectorXf(m_dC_dw[0].rows(),
    m_dC_dw[0].cols()); temp.setZero(); for (auto dw : m_dC_dw) { temp = temp +
    dw;
    }
    temp = (temp.array() / m_dC_dw.size()).matrix();*/
    m_w = m_w - 0.3 * m_dC_dw;
    // m_b = m_b - 0.3 * m_dC_db;
}

ConvolutionalLayer::ConvolutionalLayer(int input_height, int input_width, int number_input_channel,
                                       int filter_height, int filter_width, int number_output_channel, int stride,
                                       int padding, std::unique_ptr<ActivationFunction> activation_function,
                                       std::unique_ptr<RandomInitialization> random_initialization)
        : m_input_height(input_height), m_input_width(input_width),
          m_number_input_channels(number_input_channel),
          m_filter_height(filter_height), m_filter_width(filter_width),
          m_number_output_channels(number_output_channel), m_stride(stride), m_padding(padding),
          m_activation_function(std::move(activation_function)),
          m_random_initialization(std::move(random_initialization)) {

    assert(filter_height==filter_width && "Filter has to be quadratic");

    m_convlayer_logger = spdlog::get("convlayer");
    m_convlayer_logger->info("Start initialization of convlayer");

    m_output_img_height = row_filter_positions(input_width, filter_width, stride, padding);
    m_output_img_width = col_filter_positions(input_height, filter_height, stride, padding);
    m_output_img_size = m_output_img_height * m_output_img_width;
    m_number_output_values = m_output_img_size * m_number_output_channels;

    m_w.resize(m_number_output_channels, m_filter_height * m_filter_width * m_number_input_channels);

    m_dC_dw.resize(m_w.rows(), m_w.cols());

    m_b.resize(m_number_output_channels);

    m_dC_db.resize(m_b.size());

    initialize_parameter();
    m_convlayer_logger->info("End initialization of convlayer");
}

void ConvolutionalLayer::backpropagation(const Eigen::MatrixXf &a_prev,
                                         const Eigen::MatrixXf &dC_da) {
    m_convlayer_logger->info("Begin backpropagation");
    std::cout << "a_prev: " << HelperFunctions::print_tensor(a_prev, m_input_height,m_input_width,m_number_input_channels) << std::endl;
    std::cout << "dC_da: " << HelperFunctions::print_tensor(dC_da, m_output_img_height,m_output_img_width,m_number_output_channels) << std::endl;

    //m_dC_da_prev.resize(a_prev.rows(), a_prev.cols());
    //m_dC_dw.setZero();
    //m_dC_da_prev.setZero();

    // derivative activation
    // std::cout << "m_a :\n" <<m_a << "\ndC_da: \n" << dC_da << std::endl;
    // std::cout << "Der Activation:\n" <<
    // m_activation_function->apply_derivative(m_a) << std::endl;

    //Compute intermidiate result dC/dz=(da/dz)*(dC/da)
    auto dC_dz = m_activation_function->apply_derivative(m_a, dC_da);

    std::cout << "dC_dz: " << HelperFunctions::print_tensor(dC_dz, m_output_img_height,m_output_img_width,m_number_output_channels) << std::endl;

    backpropagate_bias(dC_dz);

    backpropagate_weights(a_prev,dC_dz);

    backpropagate_input(dC_dz);


    //std::cout << "Reshaped\n" << dC_dz.resize(m_number_output_channels,) << std::endl;

    // use largest image between filter and dC/dz as filter
    //int filter_size=std::min(m_filter_height*m_filter_width,m_output_img_size);



//TODO Resize output
//TODO omp parallel for im2col?
//TODO async logging
//TODO python -c "import psutil; psutil.cpu_count(logical=False)"
//TODO Test openmp cores deeper



    //m_z = *reshape_im2col_result(m_w * (*im2col_input), m_input_height, m_input_width, m_filter_height, m_filter_width,
    //                             m_number_output_channels, m_stride, m_padding, input.cols());



/*    for (int i = 0; i < m_filter_height; i++) {
        filter_flip.row(i) =
                m_w.block(i * m_filter_width, 0, m_filter_width, 1).transpose();
        // std::cout <<  "\nfilter flip\n" << filter_flip << "\nblock\n"
        // <<m_w.block(i*m_filter_width,0,m_filter_width,1)<<std::endl;
    }
    filter_flip = (filter_flip.colwise().reverse()).eval();
    // std::cout << "After first reverse\n" << filter_flip << std::endl;
    filter_flip = (filter_flip.rowwise().reverse()).eval();*/

    // std::cout << "flipped filter "<<filter_flip << std::endl;
    m_convlayer_logger->info("End backpropagation");
}

void ConvolutionalLayer::backpropagate_bias(const Eigen::MatrixXf &dC_dz) {
    // Compute backpropagation of bias by summing over the channels of all samples and normalizing by dividing by the number of samples
    for (int i=0; i<m_number_output_channels; i++) {
        m_dC_db(i)=(float) (dC_dz.block(m_output_img_size*i,0,m_output_img_size,dC_dz.cols()).sum()/((double)dC_dz.cols()));
    }

    //std::cout << "m_dC_db:\n " << m_dC_db << std::endl;
}

void ConvolutionalLayer::backpropagate_weights(const Eigen::MatrixXf& a_prev, const Eigen::MatrixXf &dC_dz) {
    auto dC_dz_im2col=im2col(dC_dz, m_output_img_height, m_output_img_width, m_number_output_channels, m_output_img_height, m_output_img_width, m_stride, m_padding);
    std::cout << "dC_dz_im2col=\n" << *dC_dz_im2col << std::endl;
    auto a_prev_im2col=im2col(a_prev,m_input_height,m_input_width,m_number_input_channels,m_output_img_height,m_output_img_width,m_stride,m_padding);
    std::cout << "a_prev_im2col\n" << *a_prev_im2col << std::endl;

    auto dC_dz_im2col2=im2col2(dC_dz, m_output_img_height, m_output_img_width, m_number_output_channels, m_output_img_height, m_output_img_width, m_stride, m_padding);
    std::cout << "dC_dz_im2col2=\n" << *dC_dz_im2col2 << std::endl;
    auto a_prev_im2col2=im2col2(a_prev,m_input_height,m_input_width,m_number_input_channels,m_output_img_height,m_output_img_width,m_stride,m_padding);
    std::cout << "a_prev_im2col2\n" << *a_prev_im2col2 << std::endl;

    // dC/dw=Conv(a_prev,dC/dz) and normalization
    auto res=(a_prev_im2col2->transpose()*(*dC_dz_im2col2))/dC_dz.cols();
    std::cout << "res\n" << res << std::endl;
    m_dC_dw.noalias()=*reshape_im2col_result(res,m_input_height,m_input_width,m_output_img_height,m_output_img_width,m_number_output_channels,m_stride,m_padding,dC_dz.cols());
    std::cout << "m_dC_dw\n" << m_dC_dw << std::endl;
}

void ConvolutionalLayer::backpropagate_input(const Eigen::MatrixXf &dC_dz) {
    auto flipped_filter=flip_filter();

    auto dC_dz_padded= pad_matrix(dC_dz, m_filter_width - 1, m_output_img_height, m_output_img_width, m_number_output_channels);

    auto dC_dz_padded_im2col=im2col(*dC_dz_padded,m_output_img_height+2*m_padding,m_output_img_width+2*m_padding,m_number_output_channels,m_filter_height,m_filter_width,m_stride,m_padding);

    m_dC_da_prev.noalias()=(*flipped_filter)*(*dC_dz_padded_im2col);
}


void ConvolutionalLayer::feed_forward(const Eigen::MatrixXf &input) {
    m_convlayer_logger->info("Feed forward convolution");

    auto im2col_input = im2col(input, m_input_height, m_input_width, m_number_input_channels, m_filter_height,
                               m_filter_width, m_stride, m_padding);

    m_z = *reshape_im2col_result((m_w * (*im2col_input)).transpose(), m_input_height, m_input_width, m_filter_height, m_filter_width,
                                 m_number_output_channels, m_stride, m_padding, input.cols());

    // Add corresponding bias to all output channels
    for (int i =0; i<m_number_output_channels; i++) {
        m_z.block(i*m_output_img_size,0,m_output_img_size,m_z.cols()).array()+=m_b(i);
    }

    //Apply activation function
    m_a = m_activation_function->apply_function(m_z);
}

std::unique_ptr<Eigen::MatrixXf>
ConvolutionalLayer::im2col(const Eigen::MatrixXf &input_matrix, int img_height, int img_width,
                           int number_img_channels, int filter_height, int filter_width, int stride,
                           int padding) const {
    //TODO Add padding logic
    m_convlayer_logger->info("Start im2col");
    //m_convlayer_logger->debug("Input Matrix:\n{}", HelperFunctions::toString(input_matrix));
    m_convlayer_logger->debug(
            "{} size input matrix; {} rows input matrix; {} cols input matrix; {} filter height; {} filter width",
            input_matrix.size(), input_matrix.rows(), input_matrix.cols(), filter_height, filter_width);

    auto num_row_filter_positions = row_filter_positions(img_width, filter_width, stride, padding);
    auto num_col_filter_positions = col_filter_positions(img_height, filter_height, stride, padding);
    auto num_filter_positions = num_col_filter_positions * num_row_filter_positions;

    auto im2col_matrix = std::make_unique<Eigen::MatrixXf>(filter_width * filter_height * number_img_channels,
                                                           num_filter_positions * input_matrix.cols());

    int filter_size = filter_width * filter_height;

    for (long s = 0; s < input_matrix.cols(); s++) {
        for (int k = 0; k < number_img_channels; k++) {
            for (int j = 0; j < num_row_filter_positions; j++) {
                for (int i = 0; i < num_col_filter_positions; i++) {
                    for (int m = 0; m < filter_width; m++) {
                        auto col = input_matrix.col(s);
                        //assert(i == 0 && "LOLOLOLOL");
                        auto segment = col.segment(
                                img_height * m + i * stride + img_height * img_width * k +
                                img_height * j * stride,
                                m_filter_height);
                        //std::cout << "s:" << s << " k:" << k << " i:" << i << " j:" << j << " m:" << m  << " t:" << s * num_col_filter_positions * num_row_filter_positions<< "\nsegment: " << segment.transpose() << std::endl;
                        im2col_matrix->col(i + j * num_col_filter_positions +
                                           s * num_filter_positions).segment(
                                m * filter_height + k * filter_size, filter_height) = segment;
                        //traverse filter field, traverse filter selection row, traverse filter selection height, traverse channels
                        //int start_pos = m * m_input_height + m_input_height * j + i + m_input_height * m_input_width *
                        //Eigen::MatrixXf flat_block = input_matrix.block(0, 0);


                    }
                }
            }
        }
    }
    //std::cout << "output: \n" << m_im2col_matrix << std::endl;
    //m_convlayer_logger->error("stop");
    //m_convlayer_logger->debug("Output Matrix:\n{}",HelperFunctions::toString(m_im2col_matrix));
    m_convlayer_logger->debug("{} size output matrix; {} rows output matrix; {} cols output matrix",
                              im2col_matrix->size(), im2col_matrix->rows(), im2col_matrix->rows());
    m_convlayer_logger->info("End im2col");

    //output_matrix.row(0) = Eigen::Map<const VectorXd>(A.data(), A.size())
    //VectorXd v =
    return im2col_matrix;
}

//switched number_img_channel and sample variables
std::unique_ptr<Eigen::MatrixXf>
ConvolutionalLayer::im2col2(const Eigen::MatrixXf &input_matrix, int img_height, int img_width,
                           int number_img_channels, int filter_height, int filter_width, int stride,
                           int padding) const {
    //TODO Add padding logic
    m_convlayer_logger->info("Start im2col");
    //m_convlayer_logger->debug("Input Matrix:\n{}", HelperFunctions::toString(input_matrix));
    m_convlayer_logger->debug(
            "{} size input matrix; {} rows input matrix; {} cols input matrix; {} filter height; {} filter width",
            input_matrix.size(), input_matrix.rows(), input_matrix.cols(), filter_height, filter_width);

    auto num_row_filter_positions = row_filter_positions(img_width, filter_width, stride, padding);
    auto num_col_filter_positions = col_filter_positions(img_height, filter_height, stride, padding);
    auto num_filter_positions = num_col_filter_positions * num_row_filter_positions;

    auto im2col_matrix = std::make_unique<Eigen::MatrixXf>(filter_width * filter_height * input_matrix.cols(),
                                                           num_filter_positions * number_img_channels);

    int filter_size = filter_width * filter_height;

    for (long s = 0; s < input_matrix.cols(); s++) {
        for (int k = 0; k < number_img_channels; k++) {
            for (int j = 0; j < num_row_filter_positions; j++) {
                for (int i = 0; i < num_col_filter_positions; i++) {
                    for (int m = 0; m < filter_width; m++) {
                        auto col = input_matrix.col(s);
                        //assert(i == 0 && "LOLOLOLOL");
                        auto segment = col.segment(
                                img_height * m + i * stride + img_height * img_width * k +
                                img_height * j * stride,
                                m_filter_height);
                        //std::cout << "s:" << s << " k:" << k << " i:" << i << " j:" << j << " m:" << m  << " t:" << s * num_col_filter_positions * num_row_filter_positions<< "\nsegment: " << segment.transpose() << std::endl;
                        im2col_matrix->col(i + j * num_col_filter_positions +
                                           k * num_filter_positions).segment(
                                m * filter_height + s * filter_size, filter_height) = segment;
                        //traverse filter field, traverse filter selection row, traverse filter selection height, traverse channels
                        //int start_pos = m * m_input_height + m_input_height * j + i + m_input_height * m_input_width *
                        //Eigen::MatrixXf flat_block = input_matrix.block(0, 0);


                    }
                }
            }
        }
    }
    //std::cout << "output: \n" << m_im2col_matrix << std::endl;
    //m_convlayer_logger->error("stop");
    //m_convlayer_logger->debug("Output Matrix:\n{}",HelperFunctions::toString(m_im2col_matrix));
    m_convlayer_logger->debug("{} size output matrix; {} rows output matrix; {} cols output matrix",
                              im2col_matrix->size(), im2col_matrix->rows(), im2col_matrix->rows());
    m_convlayer_logger->info("End im2col");

    //output_matrix.row(0) = Eigen::Map<const VectorXd>(A.data(), A.size())
    //VectorXd v =
    return im2col_matrix;
}

int ConvolutionalLayer::row_filter_positions(int img_width, int filter_width, int stride, int padding) const {
    return (img_width - filter_width + 2 * padding) / stride + 1;
}

int ConvolutionalLayer::col_filter_positions(int img_height, int filter_height, int stride, int padding) const {
    return (img_height - filter_height + 2 * padding) / stride + 1;
}

const Eigen::MatrixXf &ConvolutionalLayer::get_m_z() const {
    return m_z;
}

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

    for (long i = 0; i < num_samples; i++) {
        for (int j = 0; j < number_output_channels; j++) {
            //m_convlayer_logger->debug("Transferred column\n{}",HelperFunctions::toString(im2col_transpose.col(i)));
            im2col_reshaped->col(i).segment(output_img_size * j, output_img_size) = input.col(j).segment(
                        output_img_size * i, output_img_size);
        }
    }
    //std::cout << HelperFunctions::print_tensor(m_im2col_reshaped, m_output_img_height, m_output_img_width, m_number_output_channels) << std::endl;
    //m_convlayer_logger->debug("Reshaped im2col:\n", HelperFunctions::toString(m_z));
    m_convlayer_logger->debug("End reshape forward propagation");
    return im2col_reshaped;
}

void
ConvolutionalLayer::set_filter(const Eigen::MatrixXf &new_filter) {
    //assert(filter_height*filter_width*filter_channels==new_filter.size() &&  "Filter");

    if (new_filter.rows() != m_w.rows() || new_filter.cols() != m_w.cols()) {
        throw std::invalid_argument("New Filter size does not match the old filter size");
    }
    m_w = new_filter;
}

const Eigen::MatrixXf &ConvolutionalLayer::get_dC_dw() const {
    return m_dC_dw;
}

std::unique_ptr<Eigen::MatrixXf>
ConvolutionalLayer::pad_matrix(const Eigen::MatrixXf &input, int padding, int img_height, int img_width,
                               int number_channels) const {
    m_convlayer_logger->info("Start padding matrix");
    m_convlayer_logger->debug("{} size input matrix; {} rows input matrix; {} cols input matrix; {} img height; {} img width; {} number_channels; {} size padding", input.size(), input.rows(), input.cols(), img_height,img_width,number_channels,padding);
    auto img_size=img_height*img_width;
    auto img_height_padded=img_height+2*padding;
    auto img_width_padded=img_width+2*padding;
    auto img_size_padded=img_height_padded*img_width_padded;

    // new_size=((height+2*padding)*(width+2*padding))*number_channels
    auto input_padded=std::make_unique<Eigen::MatrixXf>(img_size_padded * number_channels, input.cols());

    m_convlayer_logger->debug("{} size output matrix; {} rows output matrix; {} cols output matrix", input_padded->size(), input_padded->rows(), input_padded->cols());

    input_padded->setZero();

    for (int i=0; i<number_channels; i++) {
        for (int j=0; j<img_width; j++) {
            // Added rows of padding + skip already processed img + skip already processed cols + skip first pads of rows
            input_padded->block(img_height_padded * padding + img_size_padded * i + img_height_padded * j + padding, 0, img_height, input.cols())=input.block(img_size * i + img_height * j, 0, img_height, input.cols());
        }
    }
    m_convlayer_logger->info("Finished padding matrix");
    return input_padded;
}

std::unique_ptr<Eigen::MatrixXf> ConvolutionalLayer::flip_filter() const {
    m_convlayer_logger->info("Start flipping filter");
    auto flipped_filter=std::make_unique<Eigen::MatrixXf>(m_w.rows(), m_w.cols());
    // std::cout << "filter\n" <<m_w<<std::endl;
    for (int i=0; i< m_number_input_channels; i++) {
        flipped_filter->block(0,i*m_filter_height*m_filter_width,m_w.rows(),m_filter_height*m_filter_width)=m_w.block(0,i*m_filter_height*m_filter_width,m_w.rows(),m_filter_height*m_filter_width).rowwise().reverse();
    }
    m_convlayer_logger->info("Finished flipping filter");
    return flipped_filter;
}

void ConvolutionalLayer::set_bias(const Eigen::VectorXf &new_bias) {
    //assert(filter_height*filter_width*filter_channels==new_filter.size() &&  "Filter");

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

const Eigen::MatrixXf &ConvolutionalLayer::get_filter() const {
    return m_w;
}

const Eigen::MatrixXf &ConvolutionalLayer::get_weights_derivative() const {
    return m_dC_dw;
}

const Eigen::MatrixXf &ConvolutionalLayer::get_input_derivative() const {
    return m_dC_da_prev;
}
