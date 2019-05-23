#include <iostream>

#include "NN.h"
#include <Eigen/Core>


int main()
{
    NN NeN=NN(2,4,1);

    Eigen::MatrixXf inp;
    inp.resize(2,4);
    Eigen::MatrixXf out;
    out.resize(1,4);
    out << 0.0,1.0,1.0,0.0;
    inp << 0.0,0.0,1.0,1.0,0.0,1.0,0.0,1.0;

    NeN.train_net(1000,inp, out, 0.3);



}

