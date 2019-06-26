//
// Created by michaelb on 23/06/2019.
//

#include "Layer/ConvolutionalLayer.h"
#include <iostream>

void ConvolutionalLayer::feed_forward(const std::vector<Eigen::MatrixXf> &input) {
  //std::cout << "input:\n" << input[0] << "\nsize: " <<input.size();
  //std::cout << "rows:" << input[0].rows() << " m_w rows(): " <<m_w.rows();
  m_z.clear();
  m_a.clear();
  for (int k =0; k< input.size(); k++) {
    spdlog::debug("before segfault");
    m_z.emplace_back(Eigen::MatrixXf(input[k].rows() - m_w.rows() + 1,input[k].cols() - m_w.cols() + 1));
    spdlog::debug("after segfault");

    //std::cout << "k: " << k << " size m_z: " <<m_z.size() << "\nelem0\n" <<m_z[0] <<std::endl;

    // std::cout << "Grenze: " <<input.cols()-m_w.cols() << std::endl;
    for (int i = 0; i < input[k].cols() - m_w.cols() + 1; i++) {
      for (int j = 0; j < input[k].rows() - m_w.rows() + 1; j++) {
        // std::cout << "Input block: \n" << input.block(i,j,m_w.rows(),m_w.cols()).array() << "\nfilter: \n" << m_w << "\n\n" << "Sum: " << (input.block(i,j,m_w.rows(),m_w.cols()).array()*m_w.array()).sum()<< std::endl;
        //std::cout << "input\n" << input[k] << std::endl;
        //std::cout << "inputBlock\n" << input[k].block(i, j, m_w.rows(), m_w.cols()) << std::endl;
        //std::cout << "Window\n" << m_w << std::endl;


        auto temp1=input[k].block(i, j, m_w.rows(), m_w.cols()).array() * m_w.array();
        auto temp2=temp1.sum();
        m_z[k](i, j)=temp2;// + m_b;
      }
    }
    // std::cout << "mz: \n" << m_z << std::endl;
    m_a.push_back(m_activation_function->apply_function(m_z[k]));
    // std::cout << "ma: \n" << m_a << std::endl;
  }
}
void ConvolutionalLayer::backpropagation(const std::vector<Eigen::MatrixXf> &a_prev,
                                         const std::vector<Eigen::MatrixXf> &dC_da) {
  m_dC_dw.clear();
  m_dC_da_prev.clear();
  for (int k =0; k< a_prev.size(); k++) {
    // spdlog::set_level(spdlog::level::debug);
    m_dC_dw.emplace_back(Eigen::MatrixXf(m_w.rows(), m_w.cols()));
    // m_dC_dw.setZero();
    //m_dC_db[k] = Eigen::MatrixXf(m_b.rows(), m_b.cols());
    m_dC_da_prev.emplace_back(Eigen::MatrixXf(a_prev[k].rows(), a_prev[k].cols()));
    // TODO only set necessary parts zero
    m_dC_da_prev[k].setZero();

    // derivative activation
    // std::cout << "m_a :\n" <<m_a << "\ndC_da: \n" << dC_da << std::endl;
    // std::cout << "Der Activation:\n" << m_activation_function->apply_derivate(m_a) << std::endl;
    Eigen::MatrixXf dC_dz =m_activation_function->apply_derivate(m_a[k], dC_da[k]);

    // for (int i =0; i<a_prev.cols()-m_w.cols(); i++) {
    //  for (int j=0; j<a_prev.rows()-m_w.rows(); j++) {

    for (int i = 0; i < m_w.cols(); i++) {
      for (int j = 0; j < m_w.rows(); j++) {
        m_dC_dw[k](i, j) =
            (a_prev[k].block(i, j, dC_da[k].rows(), dC_da[k].cols()).array() *
             dC_dz.array())
                .sum();

        // m_dC_dw.block(i,j,m_w.rows(),m_w.cols())=(m_dC_dw.block(i,j,m_w.rows(),m_w.cols()).array()+m_w.array()*dC_da(i,j)).matrix();

        // m_dC_dw(i,j)=(input.block(i,j,m_w.rows(),m_w.cols()).array()*m_w.array()).sum() + m_b(0);
      }
    }

    Eigen::MatrixXf filter_flip = m_w.colwise().reverse();
    filter_flip = (filter_flip.rowwise().reverse()).eval();

    // TODO make more pretty
    // construct bigger matrix with zero padding size 2*(m_dC_da_prev_dim-1) for easier full convolution
    Eigen::MatrixXf temp_filter(m_w.rows() + 2 * (dC_dz.rows() - 1),
                                m_w.cols() + 2 * (dC_dz.cols() - 1));
    temp_filter.setZero();

    temp_filter.block(dC_dz.rows() - 1, dC_dz.cols() - 1, m_w.rows(),
                      m_w.cols()) = filter_flip;
    // std::cout << "dC_dz: \n" <<filter_flip << "\ntemp_filter: \n" << temp_filter << std::endl;

    // std::cout << "base\n" << m_w << "\nstage1:\n" << m_w.colwise().reverse() << "\nflipped filter: \n" <<filter_flip << "\nm_dC_dw: \n" <<m_dC_dw << std::endl;

    for (int i = 0; i < m_dC_da_prev[k].rows(); i++) {
      for (int j = 0; j < m_dC_da_prev[k].cols(); j++) {
        // std::cout << "block\n" <<temp_filter.block(i,j,dC_da.rows(),dC_da.cols())<<"\nWindow\n" <<dC_dz<<"\nsum: " <<(temp_filter.block(i,j,dC_da.rows(),dC_da.cols()).array()*dC_dz.array()).sum() << "\naddress: " << i << "&" <<j << std::endl;
        m_dC_da_prev[k](i, j) =
            (temp_filter.block(i, j, dC_dz.rows(), dC_dz.cols()).array() *
             dC_dz.array())
                .sum();
      }
    }
    // spdlog::set_level(spdlog::level::warn);
  }
}
const std::vector<Eigen::MatrixXf> &ConvolutionalLayer::get_forward_output() {
  return m_a;
}
const std::vector<Eigen::MatrixXf> &ConvolutionalLayer::get_backward_output() {
  return m_dC_da_prev;
}
void ConvolutionalLayer::initialize_parameter() {
  m_b=0.0;
  m_random_initialization->initialize(m_w);
}
void ConvolutionalLayer::update_parameter() {
  Eigen::VectorXf temp=Eigen::VectorXf(m_dC_dw[0].rows(),m_dC_dw[0].cols());
  temp.setZero();
  for (auto dw: m_dC_dw) {
    temp=temp+dw;
  }
  temp=(temp.array()/m_dC_dw.size()).matrix();
  m_w = m_w - 0.3 * temp;
  //m_b = m_b - 0.3 * m_dC_db;
}
ConvolutionalLayer::ConvolutionalLayer(
    int number_input_channel, int number_output_channel, int filter_width,
    int filter_heigth, std::unique_ptr<ActivationFunction> activation_function,
    std::unique_ptr<RandomInitialization> random_initialization) : m_number_input_channel(number_input_channel), m_number_output_channel(number_output_channel), m_activation_function(std::move(activation_function)), m_random_initialization(std::move(random_initialization)) {
  m_w.resize(filter_heigth, filter_width);
  initialize_parameter();
}
