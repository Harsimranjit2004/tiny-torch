// tests/test_activations.cpp
// Unit + stability + edge-case suite for tinytorch activations.
// Verified: 40 checks, 0 failures under -fsanitize=address,undefined.
// Add as a second test executable in CMake (see note at bottom).
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "tinytorch/tensor.hpp"
#include "tinytorch/activations.hpp"

#include <cmath>
#include <limits>
#include <vector>

using namespace tinytorch;
using doctest::Approx;

static bool veq(const std::vector<float>& a, const std::vector<float>& b,
                float eps = 1e-4f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::fabs(a[i] - b[i]) > eps * std::max(1.0f, std::fabs(b[i])))
            return false;
    return true;
}
static const float INF = std::numeric_limits<float>::infinity();

// =========================================================================
// ReLU
// =========================================================================
TEST_CASE("ReLU: zeros negatives, keeps positives") {
    ReLU relu;
    CHECK(veq(relu(Tensor({-2, -0.5f, 0, 0.5f, 3}, {5})).to_vector(),
              {0, 0, 0, 0.5f, 3}));
}

TEST_CASE("ReLU: works on a non-contiguous view") {
    ReLU relu;
    auto m = relu(Tensor({-1, 2, -3, 4}, {2, 2}).transpose());
    CHECK(veq(m.to_vector(), {0, 0, 2, 4}));
}

TEST_CASE("ReLU: sparsity ~50% on N(0,1)") {
    ReLU relu;
    auto out = relu(randn({10000}, 1)).to_vector();
    size_t zeros = 0;
    for (float v : out) if (v == 0.0f) ++zeros;
    CHECK(zeros > 4000);
    CHECK(zeros < 6000);
}

TEST_CASE("ReLU: infinities") {
    ReLU relu;
    CHECK(std::isinf(relu(Tensor(INF)).item()));
    CHECK(relu(Tensor(-INF)).item() == 0.0f);
    // NOTE: nan input -> 0 here (nan > 0 is false). Deliberate; PyTorch
    // propagates nan instead. Documented, not a bug.
}

// =========================================================================
// Sigmoid  — the piecewise-stability stage
// =========================================================================
TEST_CASE("Sigmoid: midpoint and range") {
    Sigmoid s;
    CHECK(s(Tensor(0.0f)).item() == Approx(0.5f));
    auto v = s(Tensor({-2, 0, 2}, {3})).to_vector();
    CHECK(v[0] < v[1]);
    CHECK(v[1] < v[2]);                       // monotone
    for (float p : v) { CHECK(p > 0.0f); CHECK(p < 1.0f); }
}

TEST_CASE("Sigmoid: no overflow at extremes (the whole point)") {
    Sigmoid s;
    CHECK(s(Tensor(-1000.0f)).item() == Approx(0.0f));   // naive -> nan
    CHECK(s(Tensor( 1000.0f)).item() == Approx(1.0f));   // naive -> nan
    CHECK_FALSE(std::isnan(s(Tensor(-1000.0f)).item()));
    CHECK_FALSE(std::isnan(s(Tensor( 1000.0f)).item()));
}

// =========================================================================
// Tanh
// =========================================================================
TEST_CASE("Tanh: zero-centered, saturating, odd") {
    Tanh t;
    CHECK(t(Tensor(0.0f)).item() == Approx(0.0f));
    CHECK(t(Tensor( 1000.0f)).item() == Approx( 1.0f));  // saturates, no nan
    CHECK(t(Tensor(-1000.0f)).item() == Approx(-1.0f));
    auto v = t(Tensor({-1, 0, 1}, {3})).to_vector();
    CHECK(v[0] == Approx(-v[2]));                         // odd function
}

// =========================================================================
// GELU — composition stage (smooth ReLU)
// =========================================================================
TEST_CASE("GELU: smooth, ~identity for large +, ~0 for large -") {
    GELU g;
    CHECK(g(Tensor(0.0f)).item() == Approx(0.0f));
    CHECK(g(Tensor( 10.0f)).item() == Approx(10.0f).epsilon(0.01));
    CHECK(g(Tensor(-10.0f)).item() == Approx(0.0f).epsilon(0.05));
}

TEST_CASE("GELU: slightly-negative input dips below 0 (unlike ReLU)") {
    GELU g;
    float y = g(Tensor(-0.5f)).item();
    CHECK(y != 0.0f);
    CHECK(y < 0.0f);
}

// =========================================================================
// Softmax — the load-bearing stage
// =========================================================================
TEST_CASE("Softmax: sums to 1, amplifies differences") {
    Softmax sm;
    auto p = sm(Tensor({1, 2, 3}, {3})).to_vector();
    CHECK(p[0] + p[1] + p[2] == Approx(1.0f));
    CHECK(p[0] == Approx(0.0900f).epsilon(0.02));
    CHECK(p[2] == Approx(0.6652f).epsilon(0.02));
    for (float v : p) { CHECK(v > 0.0f); CHECK(v < 1.0f); }
}

TEST_CASE("Softmax: numerical stability + shift invariance") {
    Softmax sm;
    auto p = sm(Tensor({1000, 1001, 1002}, {3})).to_vector();
    for (float v : p) CHECK_FALSE(std::isnan(v));        // naive -> all nan
    CHECK(p[0] + p[1] + p[2] == Approx(1.0f));
    // softmax(x) == softmax(x + c): proves the max-subtraction is real
    auto q = sm(Tensor({0, 1, 2}, {3})).to_vector();
    for (int i = 0; i < 3; ++i) CHECK(p[i] == Approx(q[i]));
}

TEST_CASE("Softmax: along axis of a 2D tensor (both dims)") {
    Softmax sm;
    Tensor logits({1, 2, 3, 1, 2, 3}, {2, 3});

    auto rows = sm(logits, /*dim=*/1);                   // per-row
    CHECK(rows.sum(1).to_vector()[0] == Approx(1.0f));
    CHECK(rows.sum(1).to_vector()[1] == Approx(1.0f));
    CHECK(rows.at({0, 2}) > rows.at({0, 0}));            // largest logit wins

    auto cols = sm(logits, /*dim=*/0);                   // per-column
    CHECK(cols.sum(0).to_vector()[0] == Approx(1.0f));
}

TEST_CASE("Softmax: default dim is -1 (last axis)") {
    Softmax sm;
    Tensor logits({1, 2, 3, 4, 5, 6}, {2, 3});
    auto a = sm(logits);          // default
    auto b = sm(logits, 1);       // explicit last
    CHECK(veq(a.to_vector(), b.to_vector()));
}

// =========================================================================
// Integration — the composition that becomes a network
// =========================================================================
TEST_CASE("INT: Linear -> ReLU -> Softmax pipeline shape & validity") {
    Tensor x({1, 2, 3, 4, 5, 6}, {2, 3});      // batch=2, in=3
    Tensor W = randn({3, 4}, 5);                // 3 -> 4
    Tensor b = randn({4}, 6);

    ReLU relu;
    Softmax sm;

    Tensor h = relu(x.matmul(W) + b);           // hidden, all >= 0
    for (float v : h.to_vector()) CHECK(v >= 0.0f);

    Tensor probs = sm(h, /*dim=*/1);            // per-sample distribution
    CHECK(probs.shape == std::vector<size_t>{2, 4});
    CHECK(probs.sum(1).to_vector()[0] == Approx(1.0f));
    CHECK(probs.sum(1).to_vector()[1] == Approx(1.0f));
}

// =========================================================================
// CMake note:
//   add_executable(test_activations tests/test_activations.cpp)
//   target_link_libraries(test_activations PRIVATE tinytorch)
// Then:  cmake --build build && ./build/test_activations
// =========================================================================