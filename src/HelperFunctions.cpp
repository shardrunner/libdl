#include "HelperFunctions.h"

std::string HelperFunctions::toString(const Eigen::MatrixXf &mat) {
        std::stringstream ss;
        ss << mat;
        return ss.str();
}