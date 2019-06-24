#include <iostream>

#include "NN.h"
#include "NeuralNetwork.h"
#include "ActivationFunction/SigmoidFunction.h"
#include "ActivationFunction/ReluFunction.h"
#include "LossFunction/BinaryCrossEntropyLoss.h"
#include "ActivationFunction/TanFunction.h"
#include "Layer/FullyConnectedLayer.h"
#include "RandomInitialization/SimpleRandomInitialization.h"
#include <Eigen/Core>
#include <memory>
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"


int main()
{
  Eigen::MatrixXf inp;
  inp.resize(2,4);
  Eigen::MatrixXf out;
  out.resize(1,4);
  out << 0.0,1.0,1.0,0.0;
  inp << 0.0,0.0,1.0,1.0,0.0,1.0,0.0,1.0;

  auto bin_loss = std::make_unique<BinaryCrossEntropyLoss>();
  auto sigmoid = std::make_unique<SigmoidFunction>();
  auto init = std::make_unique<SimpleRandomInitialization>();

  //init->print(0);

  auto mnet=NeuralNetwork(std::move(bin_loss));

  auto hid_layer=std::make_unique<FullyConnectedLayer>(2,4,std::make_unique<TanFunction>(), std::make_unique<SimpleRandomInitialization>());
  auto out_layer=std::make_unique<FullyConnectedLayer>(4,1,std::make_unique<SigmoidFunction>(), std::make_unique<SimpleRandomInitialization>());
  mnet.add_layer(std::move(hid_layer));
  mnet.add_layer(std::move(out_layer));

  //spdlog::error("Here 1");
  //init->print(1);
  //spdlog::error("Here 2");

  mnet.train_network(inp,out);

  NN NeN=NN(2,4,1);


  NeN.train_net(1,inp, out, 1.3);

}

