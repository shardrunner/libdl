#include "NN.h"

#include <iostream>

NN::NN(int dim_x, int dim_h, int dim_y) {
    this->dim_x = dim_x;
    this->dim_h = dim_h;
    this->dim_y = dim_y;
}

void NN::sigmoid(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &input) {
    input = (1 / (1 + Eigen::exp((-1) * input.array()))).matrix();
}

Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>
NN::compute_layer(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &input,
                  const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &weights,
                  const Eigen::Matrix<float, Eigen::Dynamic, 1> &bias) {
    auto result = (weights * input).colwise() + bias;
    return result;
}


float NN::cost_func(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &NN_result,
                    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &y_output) {

    int num_samples = y_output.size();

    auto left_side = y_output.cwiseProduct((NN_result.array().log()).matrix());
    auto right_side = ((1 - y_output.array()).matrix()).cwiseProduct(((1 - NN_result.array()).log()).matrix());
    float cost = (left_side + right_side).sum();
    cost = (-1) * cost / num_samples;

    return cost;
}

void NN::backprop(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &x_input,
                  const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &y_output,
                  const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &o_h_layer,
                  const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &o_o_layer,
                  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &h_layer,
                  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &o_layer,
                  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &h_bias,
                  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &o_bias,
                  float learning_rate) {

    int size = y_output.size();

    // backprop
    auto o_temp = o_o_layer - y_output;
    auto diff_o_layer = o_temp * o_h_layer.transpose() / size;
    auto diff_o_bias = o_temp.rowwise().sum() / size;
    auto left_side = o_layer.transpose() * o_temp;
    auto right_side = (1 - (o_h_layer.array().pow(2))).matrix();
    auto h_temp = left_side.cwiseProduct(right_side);
    auto diff_h_layer = h_temp * x_input.transpose() / size;
    auto diff_h_bias = h_temp.rowwise().sum() / size;

    // Update weights
    h_layer = h_layer - learning_rate * diff_h_layer;
    h_bias = h_bias - learning_rate * diff_h_bias;
    o_layer = o_layer - learning_rate * diff_o_layer;
    o_bias = o_bias - learning_rate * diff_o_bias;

}

void NN::train_net(int iterations, Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &x_input,
                   const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &y_output, float learning_rate) {

    srand((unsigned int) time(0));
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> h_layer(dim_h, dim_x);
    h_layer.setRandom();
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> o_layer(dim_y, dim_h);
    o_layer.setRandom();

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> h_bias(dim_h, 1);
    h_bias.setZero();
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> o_bias(dim_y, 1);
    o_bias.setZero();


    for (int i = 0; i < iterations; i++) {
        auto h_res_temp = compute_layer(x_input, h_layer, h_bias);
        h_res_temp = Eigen::tanh(h_res_temp.array()).matrix();

        auto o_res_temp = compute_layer(h_res_temp, o_layer, o_bias);
        sigmoid(o_res_temp);


        float cost = cost_func(o_res_temp, y_output);

        backprop(x_input, y_output, h_res_temp, o_res_temp, h_layer, o_layer, h_bias, o_bias, learning_rate);

        if (i % 50 == 0) {
            std::cout << "Loss at iteration number:" << i << " " << cost << std::endl;
        }
    }


    Eigen::Matrix<float, 2, 4> test_x;
    Eigen::Matrix<float, 1,4> test_y;

    test_x << 0,0,1,1,0,1,0,1;
    test_y << 0,1,1,0;

    auto lambda = [](float val) { return val >0.5; };

    for (int i=0; i<4; i++) {
        auto h_res_temp = compute_layer(test_x.block(0,i,2,1), h_layer, h_bias);
        h_res_temp = Eigen::tanh(h_res_temp.array()).matrix();
        auto o_res_temp = compute_layer(h_res_temp, o_layer, o_bias);
        sigmoid(o_res_temp);
        std::cout << "Result for: "<<test_x.block(0,i,2,1).transpose()<< " -> predicted: " << lambda(o_res_temp(0,0)) << std::endl;
    }
}