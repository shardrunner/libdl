#include <catch2/catch.hpp>

#include "RandomInitialization/XavierInitialization.h"

SCENARIO("Test random initialization") {
    GIVEN("Xavier initialization") {
        XavierInitialization xav_init = XavierInitialization();
        Eigen::MatrixXf input(50, 50);
        WHEN("A matrix is initialized") {
            xav_init.initialize(input);
            THEN("The mean of the output should be correct") {
                REQUIRE(std::abs(input.mean()) < 0.01);
            }
        }
    }
}