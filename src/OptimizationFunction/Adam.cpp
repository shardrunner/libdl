//
// Created by michib on 30.07.19.
//

#include "OptimizationFunction/Adam.h"

void Adam::optimize_weights(Eigen::MatrixXf &values,
                            const Eigen::MatrixXf &derivatives) {
  assert(values.rows() == m_mt.rows() && values.cols() == m_mt.cols() &&
         derivatives.rows() == m_mt.rows() &&
         derivatives.cols() == m_mt.cols() &&
         "Defined input and real input dimensions do not match!");
  // First order momentum
  m_mt *= m_beta_1;
  m_mt += (1 - m_beta_1) * derivatives;

  // Second order momentum
  m_vt *= m_beta_2;
  m_vt += (1 - m_beta_2) * derivatives.array().pow(2).matrix();

  // bias correction
  Eigen::MatrixXf m_mt_corrected = m_mt / (1 - m_beta_1);
  Eigen::MatrixXf m_vt_corrected = m_vt / (1 - m_beta_2);

  values.array() -= m_learning_rate * m_mt_corrected.array() /
                    (m_vt_corrected.array().sqrt() + m_epsilon);
}

Adam::Adam(int filter_height, int filter_width, int bias_size)
    : Adam(0.001, 1e-8, 0.9, 0.999, filter_height, filter_width, bias_size) {}

Adam::Adam(float learning_rate, float epsilon, float beta_1, float beta_2,
           int filter_height, int filter_width, int bias_size)
    : m_learning_rate(learning_rate), m_epsilon(epsilon), m_beta_1(beta_1),
      m_beta_2(beta_2) {
  m_mt = Eigen::MatrixXf::Zero(filter_height, filter_width);
  m_vt = Eigen::MatrixXf::Zero(filter_height, filter_width);
  m_mt_bias = Eigen::VectorXf::Zero(bias_size);
  m_vt_bias = Eigen::VectorXf::Zero(bias_size);
}

void Adam::optimize_bias(Eigen::VectorXf &values,
                         const Eigen::VectorXf &derivatives) {
  assert(values.rows() == m_mt_bias.rows() &&
         values.cols() == m_mt_bias.cols() &&
         derivatives.rows() == m_mt_bias.rows() &&
         derivatives.cols() == m_mt_bias.cols() &&
         "Defined input and real input dimensions do not match!");
  // First order momentum
  m_mt_bias *= m_beta_1;
  m_mt_bias += (1 - m_beta_1) * derivatives;

  // Second order momentum
  m_vt_bias *= m_beta_2;
  m_vt_bias += (1 - m_beta_2) * derivatives.array().pow(2).matrix();

  // bias correction
  Eigen::MatrixXf m_mt_corrected = m_mt_bias / (1 - m_beta_1);
  Eigen::MatrixXf m_vt_corrected = m_vt_bias / (1 - m_beta_2);

  values.array() -= m_learning_rate * m_mt_corrected.array() /
                    (m_vt_corrected.array().sqrt() + m_epsilon);
}