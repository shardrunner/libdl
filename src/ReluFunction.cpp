#include <ActivationFunction/ReluFunction.h>

#include <iostream>

ReluFunction::ReluFunction(float leak_factor) {
  this->leak_factor=leak_factor;
}

Eigen::MatrixXf ReluFunction::apply_function(const Eigen::MatrixXf &input) const {
  //spdlog::set_level(spdlog::level::debug);
  spdlog::debug("Leak factor: {}",leak_factor);
  //spdlog::debug("Input: {}",input);
  //std::cout << "Input: " << input << std::endl;
  //Mult with -1, because otherwise -0 is the result
  auto output=(input.unaryExpr([this](float x) { if (x>=0.0) {return x;} else {return (-1)*leak_factor*x; }})).eval();
  //spdlog::debug("Output: {}",output);
  //std::cout << "Output: " << output << std::endl;
  //std::cout << "Custom: " << std::endl;
  //spdlog::set_level(spdlog::level::warn);
  return output;
}

Eigen::MatrixXf
ReluFunction::apply_derivate(const Eigen::MatrixXf &m_a,
                             const Eigen::MatrixXf &dC_da) const {
  //-1 same as in apply_function
  return (m_a.unaryExpr([this](float x) -> float { if (x>0.0) {return 1.0;} else {return (-1)*leak_factor*x; }}).array()*dC_da.array()).matrix();
}