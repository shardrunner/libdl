# Deep Learning Library from Brunner Michael

Build instructions:

git clone git@gitlab.lrz.de:brunner/libdl.git
cd libdl
mkdir build
cd build
cmake ../.
make
./testmain

The network is also executed and shown on the clang CI runner and the result with the default configuration can be seen there as well (e.g. https://gitlab.lrz.de/brunner/libdl/-/jobs/550453).

The network currently takes 2000 training mnist images and 100 images for testing, It goes through 75 iterations. The training takes under 2 minutes for me.
The accuracy and the error is sometimes a bit wonky, goes up and down or is on a plateau, and it might need some tries (<5) to get a good seed and a accuracy > 0.6. The accuracy can be found above the printed testing images.
The cause for the wonky convergence might be the SGD as gradient descent function or the simple network architecture.

The library itself is somewhat modular and most of the implementations already work. The convolution layer however is still limited to one input and output channel.

The code is still very wip, not commented and all over the place due to time constraints.

The tests are unfortunately broken as well.

### TODO:
- [ ] Make proper CI Pipline Setup
- [x] Smart Pointers vs References/management of objects to hand over
- [ ] Fix Doxygen Setup
- [ ] Add logging setup with asserts!
- [ ] Fix tests
- [ ] Fix wonky error and convergence
- [ ] Implement more optimization functions
- [ ] Finish convolution layer
- [ ] Code cleanup and commenting
- [x] More test coverage
- [ ] Add regularization
- [ ] Add dropout
- [ ] Add visualisation error, convergence, ...
- [ ] Add automatisms (grid search)
- [ ] Add interactivity + feedback
- [ ] Add convergence checks + on-the-fly adjustments
- [x] Add more flexibility/options (cost function, initialization values, activation functions, gradient descent alternatives, network layout)