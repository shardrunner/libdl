# Deep Learning Library from Brunner Michael

## How to use:
`#include "NN.h"`

Initialize NN with the network dimensions. `NN(2,4,1)` did work for me and makes the most sense for XOR.
You can change the middle value, the dimension of the hidden layer, but it didn't reliably converge for me for <4.
If you use less than 4 neurons for the hidden layer, sometimes it doesn't converge or very slowly. It depends on the seed.

Train and test the network with the `train_net()` function. It takes four arguments.
1. The number of iterations in int (I used 1000)
2. The input matrix as 2x4 Eigen float matrix with the four different input possibilities.
3. The output labels to train against. This has to be a matching 1x4 float Eigen matrix.
4. The learning rate. I only used 0.3 so far.

To execute it like this, you can use the `testmain` from the main.cpp.

### TODO:
#### Setup:
- [ ] Make proper CI Pipline Setup
- [x] Smart Pointers vs References/management of objects to hand over
- [ ] Fix Doxygen Setup
- [ ] Add logging setup with asserts!
- [x More test coverage
#### Library:
- [ ] Add interactivity + feedback
- [ ] Add convergence checks + on-the-fly adjustments
- [x] Add more flexibility/options (cost function, initialization values, activation functions, gradient descent alternatives, network layout)