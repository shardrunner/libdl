//
// Created by michaelb on 23/06/2019.
//

#include "Layer/ConvolutionalLayer.h"
#include <iostream>

/*
 * (Input ImgCol x ImgRowx x Channel) x number_samples
 */
void ConvolutionalLayer::feed_forward(const Eigen::MatrixXf &input) {
    // std::cout << "input:\n" << input[0] << "\nsize: " <<input.size();
    // std::cout << "rows:" << input[0].rows() << " m_w rows(): " <<m_w.rows();

    // std::stringstream ss;
    // ss << input;
    // temp << input;
    // spdlog::error("Input:\n{}", ss.str());

    m_a.resize(output_values, input.cols());
    m_z.resize(output_values, input.cols());
    m_z.setZero();

    // TODO fix multiple input and output channels, for 1x1 the result is correct
    // go through all samples
    for (int k = 0; k < input.cols(); k++) {
        // one filter per output channel
        for (int n = 0; n < m_number_output_channel; n++) {
            // use filter once per input channel
            for (int m = 0; m < m_number_input_channel; m++) {
                // traverse height
                for (int i = 0; i < output_img_height; i++) {
                    // traverse width
                    for (int j = 0; j < output_img_width; j++) {
                        // for each row of the filter
                        for (int l = 0; l < m_filter_height; l++) {
                            // input column=k
                            // input row =
                            // j+(output_img_width*i)+(output_img_width*output_img_height*m)+(output_img_width*output_img_height*m_number_input_channel*n)
                            int row_input =
                                    j + (m_input_width * (i + l)) +
                                    (m_input_width * m_input_height * m); // input channels
                            int row_output =
                                    j + (output_img_width * i) +
                                    (output_img_width * output_img_height * n); // output channels
                            // take a block with length of a filter column apply one column of
                            // filter
                            int filter_channel =
                                    m_filter_width * l + (m_filter_height * m_filter_width) * m +
                                    (m_filter_height * m_filter_width * m_number_input_channel) *
                                    n;

                            auto block1 = input.block(row_input, k, m_filter_width, 1);
                            auto block2 = m_w.block(filter_channel, 0, m_filter_width, 1);

                            /*std::cout << "row input: " << row_input << " row output "
                                      << row_output << " column " << k << "\nblock1:\n"
                                      << block1 << "\nblock2\n"
                                      << block2 << "\nsum\n"
                                      << m_z(row_output, k) << std::endl;*/

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

    /*std::cout << "Conv Feed Forward: Rando weights\n"
              << m_w << "\ninput:\n"
              << input << "\nOutput\n"
              << m_a << std::endl;*/
}

void ConvolutionalLayer::backpropagation(const Eigen::MatrixXf &a_prev,
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
    for (int k = 0; k < a_prev.cols(); k++) {
        // one filter per output channel
        for (int n = 0; n < m_number_output_channel; n++) {
            // use filter once per input channel
            for (int m = 0; m < m_number_input_channel; m++) {
                // traverse height
                for (int i = 0; i < m_filter_height; i++) {
                    // traverse width
                    for (int j = 0; j < m_filter_width; j++) {
                        // for each row of the filter
                        for (int l = 0; l < output_img_height; l++) {
                            int row_input = j + (m_input_width * (i + l)) +
                                            (m_input_width * m_input_height * m);
                            int row_filter = j + (m_filter_width * i) +
                                             (m_filter_width * m_filter_height * m) +
                                             (m_filter_width * m_filter_height *
                                              m_number_input_channel * n);

                            int filter_channel = output_img_width * l +
                                                 (output_img_height * output_img_width) * n;

                            auto block1 = a_prev.block(row_input, k, output_img_width, 1);
                            auto block2 = dC_dz.block(filter_channel, k, output_img_width, 1);

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
    Eigen::MatrixXf temp_filter(filter_flip.rows() + 2 * (output_img_height - 1),
                                filter_flip.cols() + 2 * (output_img_width - 1));
    temp_filter.setZero();

    temp_filter.block(output_img_height - 1, output_img_width - 1,
                      filter_flip.rows(), filter_flip.cols()) = filter_flip;

    // std::cout << "dC_dz: \n" <<filter_flip << "\ntemp_filter: \n" <<
    // temp_filter << std::endl;

    // std::cout << "base\n" << m_w << "\nstage1:\n" << m_w.colwise().reverse()
    // << "\nflipped filter: \n" <<filter_flip << "\nm_dC_dw: \n" <<m_dC_dw <<
    // std::endl;

    // std::cout << "dC_dz\n" << dC_dz << "\nfilter_temp\n" << temp_filter <<
    // std::endl;//"\nfilter\n" << m_w << std::endl;

    for (int k = 0; k < a_prev.cols(); k++) {
        for (int i = 0; i < m_input_height; i++) {
            for (int j = 0; j < m_input_width; j++) {
                // std::cout << "block\n"
                // <<temp_filter.block(i,j,dC_da.rows(),dC_da.cols())<<"\nWindow\n"
                // <<dC_dz<<"\nsum: "
                // <<(temp_filter.block(i,j,dC_da.rows(),dC_da.cols()).array()*dC_dz.array()).sum()
                // << "\naddress: " << i << "&" <<j << std::endl;
                for (int l = 0; l < output_img_height; l++) {
                    int row_dC_da_prev = j + i * m_input_width;
                    int row_dC_dz = l * output_img_width;

                    auto block1 =
                            temp_filter.block(i + l, j, 1, output_img_width).transpose();
                    auto block2 = dC_dz.block(row_dC_dz, k, output_img_width, 1);

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
    m_random_initialization->initialize(m_w);
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

ConvolutionalLayer::ConvolutionalLayer(
        int input_height, int input_width, int number_input_channel,
        int number_output_channel, int filter_height, int filter_width, int stride,
        std::unique_ptr<ActivationFunction> activation_function,
        std::unique_ptr<RandomInitialization> random_initialization)
        : m_input_height(input_height), m_input_width(input_width),
          m_number_input_channel(number_input_channel),
          m_number_output_channel(number_output_channel),
          m_filter_height(filter_height), m_filter_width(filter_width), m_stride(stride),
          m_activation_function(std::move(activation_function)),
          m_random_initialization(std::move(random_initialization)) {

  output_img_width = input_width - filter_width + 1;
  output_img_height = input_height - m_filter_height + 1;
  output_values = output_img_width * output_img_height * number_output_channel;

    m_w.resize(m_filter_height * m_filter_width * m_number_input_channel *
               m_number_output_channel,
               1);
    m_dC_dw.resize(m_w.rows(), m_w.cols());

    m_b.resize(number_output_channel);
    m_dC_db.resize(m_b.size());

    initialize_parameter();
}
