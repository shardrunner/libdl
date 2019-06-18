#include <ReluFunction.h>

ReluFunction::ReluFunction(float leak_factor) {
  this->leak_factor=leak_factor;
}

void ReluFunction::apply_function(Eigen::MatrixXf &input) {
  input=input.unaryExpr([this](float x) { if (x<0.0) {return x;} else {return leak_factor*x; }});
}

void ReluFunction::apply_derivate(Eigen::MatrixXf &input) {
  input=input.unaryExpr([this](float x) -> float { if (x<0.0) {return 1.0;} else {return leak_factor*x; }});
}