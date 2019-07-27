#include "HelperFunctions.h"
#include <iostream>


std::string HelperFunctions::toString(const Eigen::MatrixXf &mat) {
        std::stringstream ss;
        ss << mat;
        return ss.str();
}

std::string HelperFunctions::print_tensor(const Eigen::MatrixXf &input, int img_height, int img_width, int num_channels) {
    std::stringstream ss;
    ss << "Matrix of size: " << input.size() << "; Number of samples: " <<input.cols() << "; Images of height: " << img_height << ", width: " << img_width << ", channels: " << num_channels << std::endl;
    for (long i =0; i < input.cols(); i++) {
        ss << "\nSample: " << i <<std::endl;
        for (int j=0; j< num_channels; j++) {
            ss << "Channel: " << j << std::endl;
            for (int k=0; k < img_height; k++) {
                for (int l=0; l<img_width; l++) {
                    ss << input(k+l*img_height+img_height*img_width*j,i) << " ";
                }
                ss << std::endl;
            }
        }
    }
    return ss.str();
}

std::string HelperFunctions::print_tensor_first(const Eigen::MatrixXf &input, int img_height, int img_width, int num_channels) {
    std::stringstream ss;
    ss << "Matrix of size: " << input.size() << "; Number of samples: " <<input.cols() << "; Images of height: " << img_height << ", width: " << img_width << ", channels: " << num_channels << std::endl;

    for (int j=0; j< num_channels; j++) {
        ss << "Channel: " << j << std::endl;
        for (int k=0; k < img_height; k++) {
            for (int l=0; l<img_width; l++) {
                ss << input(k+l*img_height+img_height*img_width*j,0) << " ";
            }
            ss << std::endl;
        }
    }
    return ss.str();
}
