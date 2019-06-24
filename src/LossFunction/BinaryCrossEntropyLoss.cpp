#include "LossFunction/BinaryCrossEntropyLoss.h"

#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"
#include <iostream>

float BinaryCrossEntropyLoss::calculate_loss() const {
    //const Eigen::MatrixXf &a_prev, const Eigen::RowVectorXf &label) const {

  //int num_samples = label.size();

  //std::cout << "Prev: " << a_prev << std::endl;
  //std::cout << "label: " << label << std::endl;
  //spdlog::set_level(spdlog::level::debug);
/*  spdlog::debug("In calculate loss");
  spdlog::debug("Dimension prev x: {}, y: {}", a_prev.rows(),a_prev.cols());
  spdlog::debug("Dimension label x: {}, y: {}", label.rows(),label.cols());

  auto left_side = label.cwiseProduct((a_prev.array().log()).matrix());
  auto right_side = ((1 - label.array()).matrix()).cwiseProduct(((1 - a_prev.array()).log()).matrix());
  float cost = (left_side + right_side).sum();
  cost = (-1) * cost / float(num_samples);*/

  //spdlog::set_level(spdlog::level::warn);

  //return cost;


  return temp_loss.array().abs().log().sum() / temp_loss.cols();
}
void BinaryCrossEntropyLoss::backpropagate(const Eigen::MatrixXf &a_prev,
                                      const Eigen::MatrixXf &label) {

  const int nobs = a_prev.cols();
  const int nvar = a_prev.rows();

  if ((label.cols() != nobs) || (label.rows() != nvar))
  {
    throw std::invalid_argument("Target data have incorrect dimension");
  }
  spdlog::debug("1");
/*  if (a_prev.rows() != 1) {
    throw std::invalid_argument("only one last line");
  }*/

  //std::cout << "a_prev " << a_prev << "\nlabel: " << label << std::endl;

//  spdlog::debug("2");
//  Eigen::MatrixXf temp1=-(label.array()/a_prev.array());
//  spdlog::debug("3");
//  Eigen::MatrixXf temp2=((1-label.array())/(1-a_prev.array())).matrix();
//  spdlog::debug("4");
//
//  temp_loss= temp1+temp2;

/*  temp_loss= (label.array() == 0).select((float(1) -
                                       a_prev.array()).cwiseInverse(),
                                      -a_prev.cwiseInverse()).matrix();*/

  temp_loss.array() = (label.array() < 0.5).select((1 - a_prev.array()).cwiseInverse(),-a_prev.cwiseInverse());
}
const Eigen::MatrixXf &BinaryCrossEntropyLoss::get_backpropagate() const {
  return temp_loss;
}
