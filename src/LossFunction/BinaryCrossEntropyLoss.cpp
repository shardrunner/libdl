#include "LossFunction/BinaryCrossEntropyLoss.h"

#include <iostream>

float BinaryCrossEntropyLoss::calculate_loss(
    const Eigen::MatrixXf &a_prev, const Eigen::VectorXi &label) const {
  // const Eigen::MatrixXf &a_prev, const Eigen::RowVectorXf &label) const {

  int num_samples = label.size();

  // std::cout << "Prev: " << a_prev << std::endl;
  // std::cout << "label: " << label << std::endl;
  // spdlog::set_level(spdlog::level::debug);
  //  spdlog::debug("In calculate loss");
  //  spdlog::debug("Dimension prev x: {}, y: {}", a_prev.rows(),a_prev.cols());
  //  spdlog::debug("Dimension label x: {}, y: {}", label.rows(),label.cols());

  Eigen::RowVectorXf row_label = label.transpose().cast<float>();

  auto left_side = row_label.cwiseProduct((a_prev.array().log()).matrix());
  auto right_side = ((1 - row_label.array()).matrix())
                        .cwiseProduct(((1 - a_prev.array()).log()).matrix());
  float cost = (left_side + right_side).sum();
  cost = (-1) * cost / float(num_samples);

  return cost;

  // spdlog::set_level(spdlog::level::warn);

  // return cost;

  // return temp_loss.array().abs().log().sum() / temp_loss.cols();
}
void BinaryCrossEntropyLoss::backpropagate(const Eigen::MatrixXf &a_prev,
                                           const Eigen::VectorXi &label) {

  const int nobs = a_prev.cols();
  const int nvar = a_prev.rows();

  int t1 = label.cols();
  int t2 = label.rows();

  Eigen::RowVectorXi label_row = label.transpose();

  if ((label_row.cols() != nobs) || (label_row.rows() != nvar)) {
    throw std::invalid_argument("Target data have incorrect dimension");
  }
  //  spdlog::debug("1");
  /*  if (a_prev.rows() != 1) {
      throw std::invalid_argument("only one last line");
    }*/

  // std::cout << "a_prev " << a_prev << "\nlabel: " << label << std::endl;

  //  spdlog::debug("2");
  // Eigen::RowVectorXf temp=label.cast<float>().transpose();
  // Eigen::MatrixXf temp1=(-1)*(temp.array()/a_prev.array()).matrix();
  //  spdlog::debug("3");
  // Eigen::MatrixXf temp2=((1-temp.array())/(1-a_prev.array())).matrix();
  //  spdlog::debug("4");
  //
  // temp_loss= temp1+temp2;

  temp_loss = (label_row.array() == 0)
                  .select((float(1) - a_prev.array()).cwiseInverse(),
                          -a_prev.cwiseInverse())
                  .matrix();

  // temp_loss.array() = (label_row.array() < 0.5).select((1 -
  // a_prev.array()).cwiseInverse(),-a_prev.cwiseInverse());
}
const Eigen::MatrixXf &BinaryCrossEntropyLoss::get_backpropagate() const {
  return temp_loss;
}
