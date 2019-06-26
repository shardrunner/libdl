#include <iostream>

//#include "NN.h"
#include "NeuralNetwork.h"
#include "ActivationFunction/SigmoidFunction.h"
#include "ActivationFunction/ReluFunction.h"
#include "LossFunction/BinaryCrossEntropyLoss.h"
#include "ActivationFunction/TanFunction.h"
#include "Layer/FullyConnectedLayer.h"
#include "RandomInitialization/SimpleRandomInitialization.h"
#include "RandomInitialization/XavierInitialization.h"
#include "RandomInitialization/HetalInitialization.h"
#include "RandomInitialization/DeterministicInitialization.h"
#include <Eigen/Core>
#include <memory>
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "LossFunction/MultiClassLoss.h"
#include "ActivationFunction/IdentityFunction.h"
#include "ActivationFunction/SoftmaxFunction.h"
#include <vector>
#include "Layer/ConvolutionalLayer.h"


#include "extern/mnist/include/mnist/mnist_reader.hpp"
#include "extern/mnist/include/mnist/mnist_utils.hpp"
#include "omp.h"


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
  //mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
  //    mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

//  std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
//  std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
//  std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
//  std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;

 // mnist::normalize_dataset(dataset);
 // mnist::binarize_dataset(dataset);

  std::vector<Eigen::MatrixXf> images;
  Eigen::Vector<int, 200> labels;
  labels.setZero();

//  #pragma omp parallel for shared(dataset,images,labels)
  for (int i = 0; i < 200; i++) {
    Eigen::Matrix<float, 28, 28> image;
    image.setZero();
    for (int j = 0; j < 28; j++) {
      for (int k = 0; k < 28; k++) {
        // std::cout << (float) unsigned(dataset.training_images[i][j*28+k]) << " ";
       // image(j, k) = (float)unsigned(dataset.training_images[i][j * 28 + k]);
      }
      // std::cout << "\n";
    }
    images.push_back(image);
    // std::cout << "num: " << num_rep <<std::endl;
    //labels(i) = (int)unsigned(dataset.training_labels[i]);
  }
  //labels=labels.transpose();

  //std::cout << "Rep:\n " << labels << std::endl;

  //std::cout << "labels\n" <<labels <<std::endl;



  //std::cout << "\nis: " <<unsigned(dataset.training_labels[0]) << std::endl;

  //std::cout << images[0] << "\nsize: " << images.size() << std::endl;

  //Eigen::MatrixXf f = d.cast <float> ();

  Eigen::MatrixXf inp2;
  inp2.resize(2,4);
  Eigen::Vector<int,4> labels2;
  labels2 << 0,1,1,0;
  inp2 << 0.0,0.0,1.0,1.0,0.0,1.0,0.0,1.0;

  //std::cout << "input\n"<<inp2<< "\nout\n" << labels2 << std::endl;


  Eigen::MatrixXf inp=flatten_vec(images);
  //std::cout << "dim inp: " << inp.rows() << " & " << inp.cols() << "\ndim label " << labels.rows() << " & " << labels.cols();

  //std::vector<Eigen::MatrixXf> test;
  //test.push_back(inp);
  //std::cout << "flatten: \n" << flatten(test) <<std::endl;

  auto bin_loss = std::make_unique<BinaryCrossEntropyLoss>();
  auto sigmoid = std::make_unique<SigmoidFunction>();
  auto init = std::make_unique<SimpleRandomInitialization>();

  //init->print(0);

  auto mnet=NeuralNetwork(std::move(bin_loss),2,1);


  Eigen::MatrixXf img;
  img.resize(9,2);
  img.col(0) << 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9;
  img.col(1) << 1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8;
  img.col(0) <<0,0,0,0,0,0,0,0,0;
  Eigen::VectorXi labels3;
  labels3.resize(2);
  labels3 << 1,0;




  //hid_layer->feed_forward(img);

  //auto temp=hid_layer->get_forward_output();
  //std::cout << "flatten:\n " << temp << std::endl;

  auto hid_layer=std::make_unique<ConvolutionalLayer>(3,3,1,1,2,2,std::make_unique<ReluFunction>(), std::make_unique<DeterministicInitialization>());
  //auto hid2_layer=std::make_unique<FullyConnectedLayer>(32,32,std::make_unique<ReluFunction>(), std::make_unique<HetalInitialization>());
  auto out_layer=std::make_unique<FullyConnectedLayer>(4,1,std::make_unique<SigmoidFunction>(), std::make_unique<DeterministicInitialization>());
  mnet.add_layer(std::move(hid_layer));
  //mnet.add_layer(std::move(hid2_layer));
  mnet.add_layer(std::move(out_layer));

  //spdlog::error("Here 1");
  //init->print(1);
  //spdlog::error("Here 2");

  //mnet.train_network(inp.block(0,0,2,5),labels.block(0,0,1,5));
  mnet.train_network(img,labels3);

  //NN NeN=NN(2,4,1);


  //NeN.train_net(1,inp, out, 1.3);

  Eigen::Matrix<float, 2, 2> test_x;
  Eigen::Matrix<float, 1,4> test_y;

  test_x << 0,0,0,1;
  test_y << 0,1,1,0;

  auto lambda = [](float val) { return val >0.5; };
  //auto res = mnet.test_network(flatten(images[1]));
  //std::cout << "rows " <<res.rows() << "cols "<< res.cols() << "\nres\n " << res;

  //for (int i=0; i<4; i++) {
    //std::cout << "Result for: "<<test_x.block(0,i,2,1).transpose()<< " -> predicted: " << lambda(o_res_temp(0,0)) << std::endl;
  //}

}

