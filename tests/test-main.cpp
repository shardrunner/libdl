#define CATCH_CONFIG_RUNNER

#include "HelperFunctions.h"
#include "catch2/catch.hpp"
#include "omp.h"

// from the catch2 wiki
int main(int argc, char *argv[]) {
  // global setup...
  // init loggers
  HelperFunctions::init_loggers();

  // Try to set thread number to the physical core number (without HT), because
  // Eigen is slower otherwise
  // (https://eigen.tuxfamily.org/dox-devel/TopicMultiThreading.html)
  // HT / SMT cores should not be used if openmp is enabled
  #if defined _OPENMP
    omp_set_num_threads(omp_get_num_procs() / 2);
  #endif

  int result = Catch::Session().run(argc, argv);

  // global clean-up...

  return result;
}