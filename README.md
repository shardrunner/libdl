# Deep Learning Library

Build instructions:

    git clone git@github.com:shardrunner/libdl.git

    cd libdl

    mkdir build

    cd build

    cmake ..

    make

Then either execute mnist:

    ./Mlib_execute

or the tests

    ./Mlib_test

or the jupyter notebook
    
    // Dependencies
    python -m venv venv
    source /venv/bin/activate
    pip install jupyter
    python -mpip install matplotlib
    pip install Pillow
    
    // Move generated python class to jupyter notebook
    cp MlibWrapper.cpython-37m-x86_64-linux-gnu.so ../pybind
    
    //Execute jupyter notebook
    jupyter notebook ../pybind/PyNotebook.ipynb
    
    //Source for example dataset (extract in data folder and edit jupyter notebook data location accordingly)
    https://www.kaggle.com/puneet6060/intel-image-classification
    
Doxygen documentation

    // In build directory
    doxygen Doxyfile.doxygen

## General structure

The deep learning library is coded in an object oriented structure.
It makes extensive usage of the *Eigen* library for linear algebra operations.

The library itself manages its dependencies itself.
Only the jupyter notebook requires the installation of additional libraries.

There are some asserts active in debug mode, that check for correctness of the configuration and the input. 
Those are however disabled in release mode.

The *NeuralNetwork* class is the heart of every network.
It manages the represented network structure and acts as an interface between network and user.

The abstract classes *ActivationFunction*, *Layer*, *LossFunction*, *OptimizationFunction* and *RandomInitialization* define the interface between the network components.

The *HelperFunction* file provides useful functions for printing and logging.

The test functions written in Catch2 are in the tests folder.

There is also a graphical interface in the form of a *jupyter notebook* with pybindings. This allows more flexibility with training data sources and it also handles the project.

A rudimentary logging is implemented with *spdlog*.
The log files are piped to the std::out and written execution directory.
The log level and handling can be changed in the *HelperFunction* file.

## Design choices

### No wrapper for Tensor

I decided to not use a wrapper or the Eigen::Tensor implementation for the representation of tensors.
I liked the structure with one sample/image per column.
I'm not sure if this was the correct decision in the end, because managing and remembering the embedded image size was really tedious and prone to error.
It also made implementing functions on those custom tensors a lot harder.
On the other side, it's probably pretty fast with the column major storage layout and I didn't have to use the unsupported Eigen tensors.

Format of internally stored matrices:

    normal Matrix:
    Channel 1
    1   4
    2   5
    3   6
                     
    Channel 2
    7   10
    8   11
    9   12
    
    Own format:
    (1,2,3,4,5,6,7,8,9,10,11,12).transpose()

### A lot of work in CMake file

I put a lot of work in the CMake file in order to be able to build the library with the submodules and no dependencies.
In addition, I activated every optimization and warning flag I found. 

### No maxpool layer

I didn't feel like they were necessary, because I have filters with stride.

### Object oriented design

Aside from the manager class in the form of 'NeuralNetwork.h' and the helper classes 'ManageLogger.h' and 'NeuralNetwork.h', every class either inherits from a abstract class or is one themselves.

The handling of the classes is done with unique_ptr.

This structure allows for a flexible and modular library.
New features can be easily integrated, as long as they follow already existing base classes.

The represented network itself is easily extensible and flexible in its form. The child classes are all interchangeable and there are aside from the typedefs no hardcoded limits in the network size.

### Matrix operations

I tried to realise most of the function in Eigen matrix operations for performance reasons.

Unfortunately, I didn't manage to do this for the convolution layer

## Current state

* some floating point conversion warnings from test initializations and unused parameters from identity function
* all basic deep learning library necessities are covered
* padding isn't implemented
* everything else should work
* convolution and im2col is slow
* logging is sporadic
* tests are lacking for the convolution layer and for "non-math" stuff
* those really ugly network initialization functions for pybind are in NeuralNetwork, because pybind doesn't work with unique_ptr in function parameter

## Notation    

    From 3blue1brown Neural Network video series:
    C: Loss
    w: weights
    z: result before activation, [z=w*a(prev)+b]
    a: activated z, [a=sigma(z)]
    d: derivative
    a_prev: a of previous layer

### Extra:
Use 'git submodule update --remote --merge' to update the submodules.

### TODO:
- [ ] Padding
- [x] Make proper CI Pipeline Setup
- [x] Smart Pointers vs References/management of objects to hand over
- [x] Fix Doxygen Setup
- [x] Add logging setup with asserts!
- [x] Fix tests
- [x] Fix wonky error and convergence
- [x] Implement more optimization functions
- [x] Finish convolution layer
- [x] Code cleanup and commenting
- [x] More test coverage
- [ ] Add regularization
- [ ] Add dropout
- [x] Add visualisation error, convergence, ...
- [ ] Add automatism (grid search)
- [x] Add interactivity + feedback
- [ ] Add convergence checks + on-the-fly adjustments
- [x] Add more flexibility/options (cost function, initialization values, activation functions, gradient descent alternatives, network layout)
- [x] Compute derivative manually
- [x] Move constructors?
- [x] Clarify forward -> save -> backward structure
- [x] Convert more to matrix operations
- [x] Use noalias()
- [x] Activation function remove multiple copies with either pointers, in place or pass result reference
- [x] Check reference passing, if they get assigned
- [x] omp parallel for im2col?
- [x] Test OpenMP cores deeper
- [ ] more parallel im2col investigation
