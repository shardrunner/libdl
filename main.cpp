#include <iostream>

//#include "NN.h"
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
#include "LossFunction/MultiClassLoss.h"

#include "extern/mnist/include/mnist/mnist_reader.hpp"
#include "extern/mnist/include/mnist/mnist_utils.hpp"


Eigen::MatrixXf flatten_vec(std::vector<Eigen::MatrixXf> &vec) {
  Eigen::MatrixXf input(vec[0].cols()*vec[0].rows(),vec.size());
  for (int i=0; i< vec.size(); i++) {
    Eigen::VectorXf v  = Eigen::Map<const Eigen::VectorXf>(vec[i].data(), vec[i].cols()*vec[i].rows());
    //std::cout << "size: "<< v.size() <<"dims: " << input.rows() << " " << input.cols() << std::endl;
    input.col(i)=v;
    //std::cout << "size image: " << v.size() << std::endl;
  }
  return input;
}

Eigen::MatrixXf flatten(Eigen::MatrixXf &vec) {
  Eigen::MatrixXf input(vec.cols()*vec.rows(),1);
    Eigen::VectorXf v  = Eigen::Map<const Eigen::VectorXf>(vec.data(), vec.cols()*vec.rows());
    //std::cout << "size: "<< v.size() <<"dims: " << input.rows() << " " << input.cols() << std::endl;
    input.col(0)=v;
    //std::cout << "size image: " << v.size() << std::endl;
  return input;
}

int main()
{
  // MNIST_DATA_LOCATION set by MNIST cmake config
  //std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

  // Load MNIST data
  mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
      mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

//  std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
//  std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
//  std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
//  std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;

  mnist::normalize_dataset(dataset);
  mnist::binarize_dataset(dataset);

  std::vector<Eigen::MatrixXf> images;
  Eigen::Matrix<float,10,100> labels;
  labels.setZero();

  for (int i =0; i< 100; i++) {
    Eigen::Matrix<float, 28,28> image;
    image.setZero();
    for (int j =0; j< 28; j++) {
      for (int k =0; k< 28; k++) {
        //std::cout << (float) unsigned(dataset.training_images[i][j*28+k]) << " ";
        image(j,k) = (float) unsigned(dataset.training_images[i][j*28+k]);
      }
      //std::cout << "\n";
    }
    images.push_back(image);
    int num_rep=(int) unsigned(dataset.training_labels[i]);
    //std::cout << "num: " << num_rep <<std::endl;
    labels(num_rep,i)=1;
  }
  //labels=labels.transpose();

  std::cout << "Rep:\n " << labels << std::endl;

  //std::cout << "labels\n" <<labels <<std::endl;



  //std::cout << "\nis: " <<unsigned(dataset.training_labels[0]) << std::endl;

  //std::cout << images[0] << "\nsize: " << images.size() << std::endl;

  //Eigen::MatrixXf f = d.cast <float> ();

  Eigen::MatrixXf inp;
  inp.resize(2,4);
  Eigen::MatrixXf out;
  out.resize(1,4);
  out << 0.0,1.0,1.0,0.0;
  inp << 0.0,0.0,1.0,1.0,0.0,1.0,0.0,1.0;



  inp=flatten_vec(images);
  //std::cout << "dim inp: " << inp.rows() << " & " << inp.cols() << "\ndim label " << labels.rows() << " & " << labels.cols();

  //std::vector<Eigen::MatrixXf> test;
  //test.push_back(inp);
  //std::cout << "flatten: \n" << flatten(test) <<std::endl;

  auto bin_loss = std::make_unique<MultiClassLoss>();
  auto sigmoid = std::make_unique<SigmoidFunction>();
  auto init = std::make_unique<SimpleRandomInitialization>();

  //init->print(0);

  auto mnet=NeuralNetwork(std::move(bin_loss),1000,50);

  auto hid_layer=std::make_unique<FullyConnectedLayer>(784,16,std::make_unique<ReluFunction>(), std::make_unique<SimpleRandomInitialization>());
  auto hid2_layer=std::make_unique<FullyConnectedLayer>(16,16,std::make_unique<ReluFunction>(), std::make_unique<SimpleRandomInitialization>());
  auto out_layer=std::make_unique<FullyConnectedLayer>(16,10,std::make_unique<SigmoidFunction>(), std::make_unique<SimpleRandomInitialization>());
  mnet.add_layer(std::move(hid_layer));
  mnet.add_layer(std::move(hid2_layer));
  mnet.add_layer(std::move(out_layer));

  //spdlog::error("Here 1");
  //init->print(1);
  //spdlog::error("Here 2");

  mnet.train_network(inp,labels);

  //NN NeN=NN(2,4,1);


  //NeN.train_net(1,inp, out, 1.3);

  Eigen::Matrix<float, 2, 2> test_x;
  Eigen::Matrix<float, 1,4> test_y;

  test_x << 0,0,0,1;
  test_y << 0,1,1,0;

  auto lambda = [](float val) { return val >0.5; };
  auto res = mnet.test_network(flatten(images[1]));
  std::cout << "rows " <<res.rows() << "cols "<< res.cols() << "\nres\n " << res;

  //for (int i=0; i<4; i++) {
    //std::cout << "Result for: "<<test_x.block(0,i,2,1).transpose()<< " -> predicted: " << lambda(o_res_temp(0,0)) << std::endl;
  //}

}

