#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "tinytorch/tensor.hpp"
TEST_CASE("A: construction & metadata") {
    auto t = Tensor({1,2,3,4,5,6}, {2,3});
    CHECK(t.size() == 6);
    CHECK(t.ndim() == 2);
    CHECK(t.numel() == 6);
    CHECK(t.nbytes() == 24);
    CHECK(t.strides == std::vector<size_t>{3,1});
    CHECK_THROWS_AS(Tensor({1,2,3}, {2,3}), std::invalid_argument);

    Tensor s(5.0f);                       // 0-d
    CHECK(s.ndim() == 0);
    CHECK(s.size() == 1);

    auto z = zeros({2,2});
    CHECK(z.size() == 4);
    auto r1 = randn({3,3}), r2 = randn({3,3});
    CHECK(r1.to_vector() == r2.to_vector());   // same seed → same numbers

    auto e = zeros({0,3});
    CHECK(e.size() == 0);                 // doesn't crash
}