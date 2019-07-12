#pragma once

#include "NeuralNetwork.h"
#include <algorithm>
#include <random>

void NeuralNetwork::add_layer(std::unique_ptr<BaseLayer> layer) {
  m_layer_list.push_back(std::move(layer));
}
void NeuralNetwork::feed_forward(const Eigen::MatrixXf &input) {
  spdlog::error("Test");
  m_layer_list[0]->feed_forward(input);

  for (int i = 1; i < m_layer_list.size(); i++) {
    m_layer_list[i]->feed_forward(m_layer_list[i - 1]->get_forward_output());
  }
  // std::cout << "\nOutput:
  // \n"<<m_layer_list[m_layer_list.size()-1]->get_forward_output() <<
  // std::endl;
}

void NeuralNetwork::backpropagation(const Eigen::MatrixXf &input,
                                    const Eigen::MatrixXi &label) {
  int number_layers = m_layer_list.size();

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

  for (int i = number_layers - 2; i > 0; i--) {
    m_layer_list[i]->backpropagation(
        m_layer_list[i - 1]->get_forward_output(),
        m_layer_list[i + 1]->get_backward_output());
  }

  m_layer_list[0]->backpropagation(input,
                                   m_layer_list[1]->get_backward_output());
}
void NeuralNetwork::update() {
  for (int i = 0; i < m_layer_list.size(); i++) {
    m_layer_list[i]->update_parameter();
  }
}
NeuralNetwork::NeuralNetwork(std::unique_ptr<LossFunction> loss_function)
    : m_loss_function(std::move(loss_function)) {}
void NeuralNetwork::train_network(const Eigen::MatrixXf &input,
                                  const Eigen::MatrixXi &label, int batch_size,
                                  int iterations, int divisor) {

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

  int num_batches = (int)perm_input.cols() / batch_size;
  if (num_batches < 0) {
    num_batches = 1;
  }

  float loss = 0.0;
  for (int i = 0; i < iterations; i++) {
    for (int j = 0; j < num_batches; j++) {
      if (num_batches != 1) {
        auto input_batch =
            perm_input.block(0, j * batch_size, perm_input.rows(), batch_size);
        auto label_batch =
            perm_label.block(j * batch_size, 0, batch_size, perm_label.cols());
        // std::cout << "input \n" << input.colwise().sum() << "\ninput batch\n"
        // << input_batch.colwise().sum() <<"\nlabel\n" << label << "\nlabel
        // batch\n" << label_batch << std::endl; std::cout << "Batch " << j << "
        // of " <<  num_batches<<" in iter " << i << std::endl;
      } else {
        auto input_batch = input;
        auto label_batch = label;
      }
      feed_forward(input);
      backpropagation(input, label);
      update();
      if (i % divisor == 0) {
        auto temp1 =
            m_layer_list[m_layer_list.size() - 1]->get_forward_output();
        // std::cout << /*"dim: " <<temp1.rows() << " & " << temp1.cols() <<*/
        // "\nForward_output\n" << temp1 << "\n";
        //        spdlog::debug("tesp");
        auto temp2 =
            m_loss_function->calculate_loss(temp1, label); //(temp1, label);
        std::cout << "Loss batch " << j << " of iteration number " << i << ": "
                  << temp2 << std::endl;
      }
    }
  }
}

const Eigen::MatrixXf &
NeuralNetwork::test_network(const Eigen::MatrixXf &input) {
  feed_forward(input);
  return m_layer_list[m_layer_list.size() - 1]->get_forward_output();
}

Eigen::MatrixXi NeuralNetwork::calc_accuracy(const Eigen::MatrixXf &input,
                                             const Eigen::MatrixXi &label) {
  feed_forward(input);
  auto result = m_layer_list[m_layer_list.size() - 1]->get_forward_output();

  Eigen::MatrixXi predictions(label.rows(), label.cols());

  int correct = 0;
  Eigen::MatrixXf::Index pos_max;
  for (int i = 0; i < result.cols(); i++) {
    result.col(i).maxCoeff(&pos_max);
    predictions(i) = (int)pos_max;
    if ((int)pos_max == label(i)) {
      correct += 1;
    }
  }
  std::cout << "\naccuracy " << (float)correct / label.size() << std::endl;
  return predictions;
}