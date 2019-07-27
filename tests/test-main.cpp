#define CATCH_CONFIG_RUNNER
#include "catch2/catch.hpp"
#include "ManageLoggers.h"
#include "omp.h"

int main( int argc, char* argv[] ) {
    // global setup...
    //init loggers
    ManageLoggers loggers;
    loggers.initLoggers();

    //Try to set thread number to the physical core number (without HT), because Eigen is slower otherwise (https://eigen.tuxfamily.org/dox-devel/TopicMultiThreading.html)
    omp_set_num_threads(omp_get_num_procs()/2);

    int result = Catch::Session().run( argc, argv );

    // global clean-up...

    return result;
}

/*
 * Scenario: vectors can be sized and resized
     Given: A vector with some items
      When: more capacity is reserved
      Then: the capacity changes but not the size
 */