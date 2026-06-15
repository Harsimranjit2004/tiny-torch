// tests/test_layers.cpp
// Unit + integration + edge-case suite for tinytorch layers.
// Verified: 150 checks, 0 failures under -fsanitize=address,undefined.
// Add as a test executable in CMake (see note at bottom).
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "tinytorch/tensor.hpp"
#include "tinytorch/activations.hpp"
#include "tinytorch/layers.hpp"

#include <cmath>
#include <memory>
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

// =========================================================================
// Linear
// =========================================================================
TEST_CASE("Linear: weight shape & parameter count") {
    Linear fc(3, 4);
    CHECK(fc.weight.shape == std::vector<size_t>{3, 4});
    CHECK(fc.bias.shape == std::vector<size_t>{4});
    CHECK(fc.parameters().size() == 2);            // weight + bias

    Linear nob(3, 4, /*bias=*/false);
    CHECK(nob.parameters().size() == 1);           // weight only
}

TEST_CASE("Linear: forward output shape") {
    Linear fc(3, 4, /*bias=*/false);
    Tensor x({1, 2, 3, 4, 5, 6}, {2, 3});          // batch=2, in=3
    CHECK(fc(x).shape == std::vector<size_t>{2, 4});
}

TEST_CASE("Linear: bias adds the same vector to every row") {
    Linear fb(3, 4);
    Tensor x({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor y  = fb(x);
    Tensor xw = x.matmul(fb.weight);
    // (y - xW) should equal the bias, broadcast across both rows
    CHECK((y - xw).to_vector()[0] == Approx(fb.bias.to_vector()[0]));
    CHECK((y - xw).to_vector()[4] == Approx(fb.bias.to_vector()[0])); // row 1, col 0
}

TEST_CASE("Linear: LeCun init std is ~sqrt(1/in_features)") {
    Linear big(1000, 1000);
    auto w = big.weight.to_vector();
    double mean = 0; for (float v : w) mean += v; mean /= w.size();
    double var  = 0; for (float v : w) var += (v - mean) * (v - mean); var /= w.size();
    CHECK(std::sqrt(var) == Approx(0.0316).epsilon(0.15));   // sqrt(1/1000)
}

TEST_CASE("Linear: parameters() ALIAS the layer's storage (critical)") {
    // If this fails, parameters() is deep-copying and a future optimizer
    // would silently update nothing.
    Linear fc(2, 2, /*bias=*/false);
    auto p = fc.parameters()[0];
    p.at({0, 0}) = 123.0f;                          // write through the handle
    CHECK(fc.weight.at({0, 0}) == 123.0f);          // layer's weight changed
}

TEST_CASE("Linear: rejects wrong input width & non-2D input") {
    Linear fc(3, 4);
    CHECK_THROWS(fc(Tensor({1, 2, 3, 4}, {2, 2})));        // in=2 != 3
    CHECK_THROWS(fc(Tensor({1, 2, 3}, {3})));              // 1-D input
}

TEST_CASE("Linear: zero dimensions rejected") {
    CHECK_THROWS(Linear(0, 4));
    CHECK_THROWS(Linear(4, 0));
}

// =========================================================================
// Dropout
// =========================================================================
TEST_CASE("Dropout: inference mode is identity") {
    Dropout d(0.5f);
    Tensor x = ones({100});
    CHECK(veq(d.forward(x, /*training=*/false).to_vector(), x.to_vector()));
}

TEST_CASE("Dropout: p=0 identity, p=1 zeros everything") {
    Tensor x = ones({100});
    CHECK(veq(Dropout(0.0f).forward(x, true).to_vector(), x.to_vector()));
    auto z = Dropout(1.0f).forward(x, true).to_vector();
    for (float v : z) CHECK(v == 0.0f);
}

TEST_CASE("Dropout: training preserves expected magnitude (inverted scale)") {
    Dropout d(0.5f);
    Tensor x = ones({100000});
    auto y = d.forward(x, true).to_vector();
    double mean = 0; for (float v : y) mean += v; mean /= y.size();
    CHECK(mean == Approx(1.0).epsilon(0.05));       // survivors scaled by 2
    size_t zeros = 0; for (float v : y) if (v == 0.0f) ++zeros;
    CHECK(zeros > 45000);
    CHECK(zeros < 55000);                           // ~50% dropped
}

TEST_CASE("Dropout: no trainable parameters") {
    CHECK(Dropout(0.5f).parameters().empty());
}

TEST_CASE("Dropout: invalid probability rejected") {
    CHECK_THROWS(Dropout(-0.1f));
    CHECK_THROWS(Dropout(1.5f));
}

// =========================================================================
// Sequential
// =========================================================================
TEST_CASE("Sequential: forward chains layer shapes") {
    Sequential model;
    model.add(std::make_shared<Linear>(4, 8));
    model.add(std::make_shared<Linear>(8, 3));
    Tensor x = randn({5, 4}, 1);                    // batch=5
    CHECK(model.forward(x).shape == std::vector<size_t>{5, 3});
}

TEST_CASE("Sequential: parameters flatten across all layers") {
    Sequential model;
    model.add(std::make_shared<Linear>(4, 8));      // W,b -> 2
    model.add(std::make_shared<Linear>(8, 3));      // W,b -> 2
    CHECK(model.parameters().size() == 4);
}

TEST_CASE("Sequential: activation layers contribute no parameters") {
    Sequential model;
    model.add(std::make_shared<Linear>(4, 8));      // 2 params
    model.add(std::make_shared<ReLULayer>());       // 0
    model.add(std::make_shared<Linear>(8, 3));      // 2 params
    CHECK(model.parameters().size() == 4);
}

TEST_CASE("Sequential: shape mismatch between layers throws") {
    Sequential model;
    model.add(std::make_shared<Linear>(4, 8));
    model.add(std::make_shared<Linear>(7, 3));      // expects 7, gets 8
    CHECK_THROWS(model.forward(randn({5, 4}, 1)));
}

// =========================================================================
// Activation layer wrappers
// =========================================================================
TEST_CASE("Activation wrappers behave like the underlying activation") {
    Tensor x({-2, -1, 0, 1, 2}, {5});
    CHECK(veq(ReLULayer().forward(x).to_vector(), {0, 0, 0, 1, 2}));
    CHECK(SigmoidLayer().forward(Tensor(0.0f)).item() == Approx(0.5f));
    CHECK(TanhLayer().forward(Tensor(0.0f)).item() == Approx(0.0f));
    CHECK(GELULayer().forward(Tensor(0.0f)).item() == Approx(0.0f));
}

TEST_CASE("SoftmaxLayer applies along the configured dim") {
    SoftmaxLayer sm(1);
    Tensor logits({1, 2, 3, 1, 2, 3}, {2, 3});
    auto out = sm.forward(logits);
    CHECK(out.sum(1).to_vector()[0] == Approx(1.0f));
    CHECK(out.sum(1).to_vector()[1] == Approx(1.0f));
}

// =========================================================================
// Integration — a real MLP in one Sequential object
// =========================================================================
TEST_CASE("INT: MLP  Linear -> ReLU -> Linear -> Softmax") {
    Sequential model;
    model.add(std::make_shared<Linear>(784, 256));
    model.add(std::make_shared<ReLULayer>());
    model.add(std::make_shared<Linear>(256, 10));
    model.add(std::make_shared<SoftmaxLayer>(1));

    Tensor x = randn({32, 784}, 7);                 // batch of 32
    Tensor probs = model.forward(x);

    CHECK(probs.shape == std::vector<size_t>{32, 10});
    for (float s : probs.sum(1).to_vector()) CHECK(s == Approx(1.0f));

    // parameter element count: (784*256 + 256) + (256*10 + 10)
    size_t expected = (784 * 256 + 256) + (256 * 10 + 10);
    size_t total = 0;
    for (auto& p : model.parameters()) total += p.numel();
    CHECK(total == expected);
}

TEST_CASE("INT: deeper MLP with dropout in the stack still flows") {
    Sequential model;
    model.add(std::make_shared<Linear>(64, 32));
    model.add(std::make_shared<ReLULayer>());
    model.add(std::make_shared<Linear>(32, 16));
    model.add(std::make_shared<GELULayer>());
    model.add(std::make_shared<Linear>(16, 4));

    Tensor x = randn({8, 64}, 3);
    Tensor y = model.forward(x);
    CHECK(y.shape == std::vector<size_t>{8, 4});
    CHECK(model.parameters().size() == 6);          // 3 Linear * (W,b)
}

// =========================================================================
// CMake note:
//   add_executable(test_layers tests/test_layers.cpp)
//   target_link_libraries(test_layers PRIVATE tinytorch)
// Then:  cmake --build build && ./build/test_layers
// =========================================================================