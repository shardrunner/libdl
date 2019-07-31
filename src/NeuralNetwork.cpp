#pragma once

#include "NeuralNetwork.h"
#include <algorithm>
#include <random>
#include <memory>
#include "omp.h"
#include <thread>
#include <chrono>

#include "LossFunction/MultiCrossEntropyLoss.h"
#include "LossFunction/BinaryCrossEntropyLoss.h"
#include "Layer/ConvolutionalLayer.h"
#include "OptimizationFunction/Adam.h"
#include "Layer/FullyConnectedLayer.h"
#include "ActivationFunction/SoftmaxFunction.h"
#include "ActivationFunction/TanhFunction.h"
#include "ActivationFunction/ReluFunction.h"
#include "RandomInitialization/HetalInitialization.h"
#include "RandomInitialization/UniformHeInitialization.h"
#include "RandomInitialization/XavierInitialization.h"
#include "RandomInitialization/UniformXavierInitialization.h"
#include "OptimizationFunction/SimpleOptimizer.h"

#include "ManageLoggers.h"
#include "HelperFunctions.h"

void NeuralNetwork::add_layer(std::unique_ptr<Layer> layer) {
    m_layer_list.push_back(std::move(layer));
}

void NeuralNetwork::feed_forward(const Eigen::MatrixXf &input) {
    m_layer_list[0]->feed_forward(input);

    for (unsigned long i = 1; i < m_layer_list.size(); i++) {
        m_layer_list[i]->feed_forward(m_layer_list[i - 1]->get_forward_output());
    }
    // std::cout << "\nOutput:
    // \n"<<m_layer_list[m_layer_list.size()-1]->get_forward_output() <<
    // std::endl;
}

void NeuralNetwork::backpropagation(const Eigen::MatrixXf &input,
                                    const Eigen::MatrixXi &label) {
    const unsigned long number_layers = m_layer_list.size();

    //  spdlog::debug("rows output: {}, columns output: {}",
    //                m_layer_list[number_layers -
    //                1]->get_forward_output().rows(), m_layer_list[number_layers
    //                - 1]->get_forward_output().cols());
    //  spdlog::debug("rows label: {}, columns label: {}", label.rows(),
    //                label.cols());
    //  spdlog::debug("before segfault");
    auto temp = (m_layer_list[number_layers - 1]->get_forward_output()).eval();
    // std::cout << label.eval();
    // std::cout << temp.eval();
    // spdlog::debug("temp");
    m_loss_function->backpropagate(temp, label);
    // spdlog::debug("after segfault");

    if (number_layers == 1) {
        m_layer_list[0]->backpropagation(input,
                                         m_loss_function->get_backpropagate());
        return;
    }

    m_layer_list[number_layers - 1]->backpropagation(
            m_layer_list[number_layers - 2]->get_forward_output(),
            m_loss_function->get_backpropagate());

    for (unsigned long i = number_layers - 2; i > 0; i--) {
        m_layer_list[i]->backpropagation(
                m_layer_list[i - 1]->get_forward_output(),
                m_layer_list[i + 1]->get_backward_output());
    }

    m_layer_list[0]->backpropagation(input,
                                     m_layer_list[1]->get_backward_output());
}

void NeuralNetwork::update() {
    for (auto &i : m_layer_list) {
        i->update_parameter();
    }
}

NeuralNetwork::NeuralNetwork(std::unique_ptr<LossFunction> loss_function)
        : m_loss_function(std::move(loss_function)) {
    //std::cout << "C++ Test: " << std::thread::hardware_concurrency() << std::endl;
    //std::cout << "OpenMP setting:" << omp_get_num_procs() << std::endl;
    //Try to set thread number to the physical core number (without HT), because Eigen is slower otherwise (https://eigen.tuxfamily.org/dox-devel/TopicMultiThreading.html)
    omp_set_num_threads(omp_get_num_procs() / 2);

/*    //omp_set_num_threads(2);
    std::cout << "C++ Test: " << std::thread::hardware_concurrency() << std::endl;
    std::cout << "OpenMP setting:" << omp_get_num_procs() << std::endl;
    std::cout << "EigenSetting" << Eigen::nbThreads( ) << std::endl;

    //int num_threads =4;
    //omp_set_num_threads ( num_threads );
    # pragma omp parallel for
    for (int i = 0; i < 8; i++) {
        # pragma omp critical
        std::cout << "My id is: "
                  << omp_get_thread_num() << std::endl;
    }*/


    //init loggers
    ManageLoggers loggers;
    loggers.initLoggers();
    m_nn_logger = spdlog::get("nn");
    m_nn_logger->info("Initialized neural network");
}

void NeuralNetwork::train_network(const Eigen::MatrixXf &input,
                                  const Eigen::MatrixXi &label, int batch_size,
                                  int iterations, int divisor) {
    m_nn_logger->info("Started training network");
    m_nn_logger->debug(
            "{} size input matrix; {} rows input matrix; {} cols input matrix;; {} batch size; {} iterations",
            input.size(), input.rows(), input.cols(), batch_size, iterations);

    check_network(input.rows());

    //auto t1=std::chrono::high_resolution_clock::now();

    auto perm_input = input;
    auto perm_label = label;

    // std::cout << "Dim input row" << input.rows() << " cols " <<input.cols() <<
    // "label rows" << label.rows() << " cols " <<label.cols() << std::endl;

    // https://stackoverflow.com/questions/15858569/randomly-permute-rows-columns-of-a-matrix-with-eigen
    std::random_device rd;
    std::mt19937 g(rd());
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(input.cols());
    perm.setIdentity();
    std::shuffle(perm.indices().data(),
                 perm.indices().data() + perm.indices().size(), g);
    perm_input = perm_input * perm; // permute columns
    // TODO change if label becomes row

    perm_label = perm * perm_label;
    // A_perm = perm * A; // permute rows

    // std::cout << "orig input\n" << input.colwise().sum() << "\nnew input\n" <<
    // perm_input.colwise().sum() << "\norig label\n" << label << "\nnew label\n"
    // << perm_label <<std::endl;

    int num_batches = (int) perm_input.cols() / batch_size;
    if (num_batches < 0) {
        num_batches = 1;
    }
    Eigen::MatrixXf input_batch;
    Eigen::MatrixXi label_batch;

    //float loss = 0.0;
    for (int i = 0; i < iterations; i++) {
        for (int j = 0; j < num_batches; j++) {
            if (num_batches != 1) {
                input_batch =
                        perm_input.block(0, j * batch_size, perm_input.rows(), batch_size);
                label_batch =
                        perm_label.block(j * batch_size, 0, batch_size, perm_label.cols());
                // std::cout << "input \n" << input.colwise().sum() << "\ninput batch\n"
                // << input_batch.colwise().sum() <<"\nlabel\n" << label << "\nlabel
                // batch\n" << label_batch << std::endl; std::cout << "Batch " << j << "
                // of " <<  num_batches<<" in iter " << i << std::endl;
            } else {
                input_batch = input;
                label_batch = label;
            }
            feed_forward(input_batch);
            backpropagation(input_batch, label_batch);
            update();
            if (divisor >= 0) {
                if (i % divisor == 0) {
                    auto temp1 =
                            m_layer_list[m_layer_list.size() - 1]->get_forward_output();
                    // std::cout << /*"dim: " <<temp1.rows() << " & " << temp1.cols() <<*/
                    // "\nForward_output\n" << temp1 << "\n";
                    //        spdlog::debug("test");
                    auto temp2 =
                            m_loss_function->calculate_loss(temp1, label_batch); //(temp1, label);
                    m_nn_logger->warn("Loss batch {} of iteration number {}: {} ", j, i, temp2);
                    //m_nn_logger->warn("Forward output {}",HelperFunctions::toString(temp1));
                }
            }
        }
    }
    m_nn_logger->info("Ended training network");
}

const Eigen::MatrixXf &
NeuralNetwork::test_network(const Eigen::MatrixXf &input) {
    feed_forward(input);
    return m_layer_list[m_layer_list.size() - 1]->get_forward_output();
}

double NeuralNetwork::get_loss(const Eigen::MatrixXf &feed_forward_input, const Eigen::VectorXi &label) const {
    return m_loss_function->calculate_loss(feed_forward_input, label);
}

Eigen::MatrixXi NeuralNetwork::calculate_accuracy(const Eigen::MatrixXf &input,
                                                  const Eigen::MatrixXi &label) {
    feed_forward(input);
    auto result = m_layer_list[m_layer_list.size() - 1]->get_forward_output();

    Eigen::MatrixXi predictions(label.rows(), label.cols());

    int correct = 0;
    Eigen::MatrixXf::Index pos_max;
    for (long i = 0; i < result.cols(); i++) {
        result.col(i).maxCoeff(&pos_max);
        predictions(i) = (int) pos_max;
        if ((int) pos_max == label(i)) {
            correct += 1;
        }
    }
    std::cout << "\naccuracy " << (double) correct / (double) label.size() << std::endl;
    return predictions;
}

void NeuralNetwork::check_network(long input_size) {
    auto output = input_size;
    for (unsigned long i = 0; i < m_layer_list.size(); i++) {
        auto input = m_layer_list[i]->get_number_inputs();
        if (output != input) {
            m_nn_logger->error("Mismatch of layer dimensions!");
            m_nn_logger->error("Layer {} has output {} and next layer {} has input {}.", i - 1, output, i, input);
            m_nn_logger->flush();
            throw std::invalid_argument("Layer dimensions mismatch. See log for more information");
        }
        output = m_layer_list[i]->get_number_outputs();
    }
    m_nn_logger->info("Layer dimensions match");
}

std::vector<std::tuple<Eigen::MatrixXf, Eigen::VectorXf>> NeuralNetwork::get_layer_parameter() const {
    std::vector<std::tuple<Eigen::MatrixXf, Eigen::VectorXf>> parameter_list;
    for (const auto &layer: m_layer_list) {
        parameter_list.emplace_back(std::make_tuple(layer->get_weights(), layer->get_bias()));
    }
    return parameter_list;
}

void
NeuralNetwork::set_layer_parameter(const std::vector<std::tuple<Eigen::MatrixXf, Eigen::VectorXf>> &parameter_list) {
    for (unsigned long i = 0; i < m_layer_list.size(); i++) {
        auto[weights, bias] =parameter_list[i];
        m_layer_list[i]->set_weights(weights);
        m_layer_list[i]->set_bias(bias);
    }
}

NeuralNetwork::NeuralNetwork() {
    omp_set_num_threads(omp_get_num_procs() / 2);

/*    //omp_set_num_threads(2);
    std::cout << "C++ Test: " << std::thread::hardware_concurrency() << std::endl;
    std::cout << "OpenMP setting:" << omp_get_num_procs() << std::endl;
    std::cout << "EigenSetting" << Eigen::nbThreads( ) << std::endl;

    //int num_threads =4;
    //omp_set_num_threads ( num_threads );
    # pragma omp parallel for
    for (int i = 0; i < 8; i++) {
        # pragma omp critical
        std::cout << "My id is: "
                  << omp_get_thread_num() << std::endl;
    }*/


    //init loggers
    ManageLoggers loggers;
    loggers.initLoggers();
    m_nn_logger = spdlog::get("nn");
    m_nn_logger->info("Initialized neural network");
}

void NeuralNetwork::use_multiclass_loss() {
    m_loss_function=std::make_unique<MultiCrossEntropyLoss>();
}

void NeuralNetwork::add_conv_layer(int input_height, int input_width, int input_channels, int filter_height,
                                   int filter_width, int output_channels, int stride, int padding) {
    m_layer_list.emplace_back(std::make_unique<ConvolutionalLayer>(input_height, input_width, input_channels, filter_height,filter_width,output_channels,stride,padding,std::make_unique<ReluFunction>(),std::make_unique<UniformHeInitialization>(),std::make_unique<Adam>(output_channels,filter_height*filter_width*input_channels,output_channels)));
}

void NeuralNetwork::add_fc_layer(int input_size, int output_size) {
    m_layer_list.emplace_back(std::make_unique<FullyConnectedLayer>(input_size, output_size, std::make_unique<TanhFunction>(), std::make_unique<UniformXavierInitialization>(), std::make_unique<Adam>(input_size, output_size, output_size)));
}

void NeuralNetwork::add_output_layer(int input_size, int output_size) {
    m_layer_list.emplace_back(std::make_unique<FullyConnectedLayer>(input_size, output_size, std::make_unique<SoftmaxFunction>(), std::make_unique<UniformXavierInitialization>(), std::make_unique<Adam>(input_size, output_size, output_size)));
}

void NeuralNetwork::add_conv_layer_simple(int input_height, int input_width, int input_channels, int filter_height,
                                          int filter_width, int output_channels, int stride, int padding) {
    m_layer_list.emplace_back(std::make_unique<ConvolutionalLayer>(input_height, input_width, input_channels, filter_height,filter_width,output_channels,stride,padding,std::make_unique<ReluFunction>(),std::make_unique<HetalInitialization>(),std::make_unique<SimpleOptimizer>(0.1)));

}

void NeuralNetwork::add_fc_layer_relu(int input_size, int output_size) {
    m_layer_list.emplace_back(std::make_unique<FullyConnectedLayer>(input_size, output_size, std::make_unique<ReluFunction>(), std::make_unique<XavierInitialization>(), std::make_unique<SimpleOptimizer>(0.1)));
}

void NeuralNetwork::add_output_layer_simple(int input_size, int output_size) {
    m_layer_list.emplace_back(std::make_unique<FullyConnectedLayer>(input_size, output_size, std::make_unique<SoftmaxFunction>(), std::make_unique<XavierInitialization>(), std::make_unique<SimpleOptimizer>(0.1)));
}


void NeuralNetwork::train_batch(Eigen::Ref<const Eigen::MatrixXf> &input_batch, Eigen::Ref<const Eigen::VectorXi> &label_batch) {
    m_nn_logger->warn("Started training");
    feed_forward(input_batch);
    m_nn_logger->warn("Ended feed forward");
    backpropagation(input_batch, label_batch);
    m_nn_logger->warn("Ended backpro");
    update();
    m_nn_logger->warn("Ended training");
}

void NeuralNetwork::set_layer_weights(Eigen::Ref<const Eigen::MatrixXf> &param, unsigned long position) {
    m_layer_list[position]->set_weights(param);
}

const Eigen::MatrixXf &NeuralNetwork::get_layer_weights(unsigned long position) const {
    return m_layer_list[position]->get_weights();
}

void NeuralNetwork::set_layer_bias(Eigen::Ref<const Eigen::VectorXf> &param, unsigned long position) {
    m_layer_list[position]->set_bias(param);
}

const Eigen::VectorXf &NeuralNetwork::get_layer_bias(unsigned long position) const {
    return m_layer_list[position]->get_bias();
}

float NeuralNetwork::get_current_accuracy(const Eigen::VectorXi &labels) const {

    const Eigen::MatrixXf &feed_forward = m_layer_list[m_layer_list.size() - 1]->get_forward_output();

    int correct = 0;
    Eigen::MatrixXf::Index pos_max;
    for (long i = 0; i < feed_forward.cols(); i++) {
        feed_forward.col(i).maxCoeff(&pos_max);
        if ((int) pos_max == labels(i)) {
            correct += 1;
        }
    }
    return (float) correct/labels.size();
}

float NeuralNetwork::get_current_error(const Eigen::VectorXi &labels) const {

    const Eigen::MatrixXf &feed_forward = m_layer_list[m_layer_list.size() - 1]->get_forward_output();
    return (float) m_loss_function->calculate_loss(feed_forward, labels);
}

const Eigen::MatrixXf &NeuralNetwork::get_current_prediction() const {
    return m_layer_list[m_layer_list.size() - 1]->get_forward_output();
}

int NeuralNetwork::layer_size() const {
    std::cout << "Size:" << m_layer_list.size() << std::endl;
    m_nn_logger->info("Size");
    return m_layer_list.size();
}

void NeuralNetwork::feed_forward_py(Eigen::Ref<const Eigen::MatrixXf> &input_batch) {
    m_layer_list[0]->feed_forward(input_batch);

    for (unsigned long i = 1; i < m_layer_list.size(); i++) {
        m_layer_list[i]->feed_forward(m_layer_list[i - 1]->get_forward_output());
    }
}

Eigen::VectorXi NeuralNetwork::get_predicted_classes(const Eigen::MatrixXf &prediction) const {
    Eigen::VectorXi predictions(prediction.cols());

    Eigen::MatrixXf::Index pos_max;
    for (long i = 0; i < prediction.cols(); i++) {
        prediction.col(i).maxCoeff(&pos_max);
        predictions(i) = (int) pos_max;
    }
    return predictions;
}


