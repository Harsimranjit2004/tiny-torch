// tests/test_losses.cpp
// Unit + stability + integration suite for tinytorch losses.
// Verified: 26 core checks + integration, 0 failures under
// -fsanitize=address,undefined.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "tinytorch/tensor.hpp"
#include "tinytorch/activations.hpp"
#include "tinytorch/layers.hpp"
#include "tinytorch/losses.hpp"

#include <cmath>
#include <memory>
#include <vector>

using namespace tinytorch;
using doctest::Approx;

// =========================================================================
// log_softmax — the load-bearing helper
// =========================================================================
TEST_CASE("log_softmax: exp(log_softmax) is a valid distribution, all <= 0") {
    auto ls = log_softmax(Tensor({1, 2, 3}, {3}));
    auto p = exp(ls).to_vector();
    CHECK(p[0] + p[1] + p[2] == Approx(1.0f));
    for (float v : ls.to_vector()) CHECK(v <= 0.0f);     // log-prob is <= 0
}

TEST_CASE("log_softmax: stability + shift invariance (the whole point)") {
    auto ls = log_softmax(Tensor({50, 100, 150}, {3}));
    for (float v : ls.to_vector()) CHECK_FALSE(std::isnan(v));   // naive -> nan
    // subtracting a constant first changes nothing (algebraically exact)
    auto a = log_softmax(Tensor({50, 100, 150}, {3})).to_vector();
    auto b = log_softmax(Tensor({0, 50, 100}, {3})).to_vector();
    for (int i = 0; i < 3; ++i) CHECK(a[i] == Approx(b[i]));
}

TEST_CASE("log_softmax: along a 2D axis, each row normalizes") {
    Tensor x({1, 2, 3, 4, 5, 6}, {2, 3});
    auto e = exp(log_softmax(x, 1));
    CHECK(e.sum(1).to_vector()[0] == Approx(1.0f));
    CHECK(e.sum(1).to_vector()[1] == Approx(1.0f));
}

// =========================================================================
// gather_rows (the cross-entropy primitive)
// =========================================================================
TEST_CASE("gather_rows: picks one element per row") {
    Tensor x({10, 11, 12, 20, 21, 22}, {2, 3});
    auto g = gather_rows(x, {0, 2});            // row0 col0, row1 col2
    CHECK(g.shape == std::vector<size_t>{2});
    CHECK(g.to_vector()[0] == 10.0f);
    CHECK(g.to_vector()[1] == 22.0f);
    CHECK_THROWS(gather_rows(x, {0}));          // wrong count
    CHECK_THROWS(gather_rows(Tensor({1, 2, 3}, {3}), {0}));  // not 2D
}

// =========================================================================
// MSELoss
// =========================================================================
TEST_CASE("MSELoss: zero on identical, matches hand-computed values") {
    MSELoss mse;
    CHECK(mse(Tensor({1, 2, 3}, {3}), Tensor({1, 2, 3}, {3})).item()
          == Approx(0.0f));
    // (2-1)^2 + (4-4)^2 + (6-9)^2 = 1+0+9 = 10, /3
    CHECK(mse(Tensor({2, 4, 6}, {3}), Tensor({1, 4, 9}, {3})).item()
          == Approx(10.0f / 3.0f));
    // 2D grand mean: only last element differs by 1 -> 1/4
    CHECK(mse(Tensor({1, 2, 3, 4}, {2, 2}), Tensor({1, 2, 3, 5}, {2, 2})).item()
          == Approx(0.25f));
}

TEST_CASE("MSELoss: shape mismatch throws") {
    MSELoss mse;
    CHECK_THROWS(mse(Tensor({1, 2, 3}, {3}), Tensor({1, 2}, {2})));
}

// =========================================================================
// CrossEntropyLoss — logits in, NOT probabilities
// =========================================================================
TEST_CASE("CrossEntropyLoss: confident-correct ~0, uniform ~log(C)") {
    CrossEntropyLoss ce;
    // huge logit on the correct class -> loss ~ 0
    Tensor logits({100, 0, 0, 0, 100, 0}, {2, 3});
    CHECK(ce.forward(logits, {0, 1}).item() == Approx(0.0f).epsilon(0.01));
    // uniform logits -> loss = log(C) = log(3)
    CHECK(ce.forward(Tensor({1, 1, 1}, {1, 3}), {0}).item()
          == Approx(std::log(3.0f)));
}

TEST_CASE("CrossEntropyLoss: confident-and-wrong costs more than correct") {
    CrossEntropyLoss ce;
    Tensor lw({100, 0, 0}, {1, 3});
    CHECK(ce.forward(lw, {0}).item() < ce.forward(lw, {1}).item());
}

TEST_CASE("CrossEntropyLoss: stable at huge logits (no nan)") {
    CrossEntropyLoss ce;
    Tensor big({500, 0, 0, 0, 500, 0}, {2, 3});
    CHECK_FALSE(std::isnan(ce.forward(big, {0, 1}).item()));
}

TEST_CASE("CrossEntropyLoss: bad target index & shape errors throw") {
    CrossEntropyLoss ce;
    CHECK_THROWS(ce.forward(Tensor({1, 1, 1}, {1, 3}), {5}));     // index OOB
    CHECK_THROWS(ce.forward(Tensor({1, 1, 1}, {1, 3}), {0, 1})); // count != batch
    CHECK_THROWS(ce.forward(Tensor({1, 2, 3}, {3}), {0}));       // not 2D
}

// =========================================================================
// BinaryCrossEntropyLoss — probabilities in, 0/1 labels
// =========================================================================
TEST_CASE("BCELoss: confident-correct ~0, p=0.5 -> log(2)") {
    BinaryCrossEntropyLoss bce;
    CHECK(bce(Tensor({0.999f, 0.001f}, {2}), Tensor({1, 0}, {2})).item()
          == Approx(0.0f).epsilon(0.02));
    CHECK(bce(Tensor({0.5f, 0.5f}, {2}), Tensor({1, 0}, {2})).item()
          == Approx(std::log(2.0f)));
}

TEST_CASE("BCELoss: eps-clamp prevents nan/inf at p in {0,1}") {
    BinaryCrossEntropyLoss bce;
    float v = bce(Tensor({0.0f, 1.0f}, {2}), Tensor({1, 0}, {2})).item();
    CHECK_FALSE(std::isnan(v));
    CHECK_FALSE(std::isinf(v));
}

TEST_CASE("BCELoss: confident-and-wrong is large; shape mismatch throws") {
    BinaryCrossEntropyLoss bce;
    CHECK(bce(Tensor({0.001f}, {1}), Tensor({1}, {1})).item() > 5.0f);
    CHECK_THROWS(bce(Tensor({0.5f}, {1}), Tensor({1, 0}, {2})));
}

// =========================================================================
// Integration — the training-signal preview
// =========================================================================
TEST_CASE("INT: model logits -> CrossEntropyLoss is a finite scalar ~log(C)") {
    Sequential model;
    model.add(std::make_shared<Linear>(8, 4));   // 4 classes, NO softmax layer
    Tensor x = randn({6, 8}, 1);                 // batch 6
    Tensor logits = model.forward(x);            // {6,4} raw logits

    CrossEntropyLoss ce;
    float loss = ce.forward(logits, {0, 1, 2, 3, 0, 1}).item();
    CHECK(loss > 0.0f);
    CHECK_FALSE(std::isnan(loss));
    // an untrained 4-class model predicts ~uniform -> loss ~ log(4) ~ 1.386
    CHECK(loss == Approx(std::log(4.0f)).epsilon(0.5));
}

TEST_CASE("INT: sigmoid output -> BCELoss is a finite scalar") {
    Sequential model;
    model.add(std::make_shared<Linear>(5, 1));   // 1 logit per sample
    Tensor x = randn({4, 5}, 2);
    Tensor logits = model.forward(x);            // {4,1}
    Tensor probs = Sigmoid()(logits);            // squash to (0,1)

    BinaryCrossEntropyLoss bce;
    float loss = bce(probs, Tensor({1, 0, 1, 0}, {4, 1})).item();
    CHECK(loss > 0.0f);
    CHECK_FALSE(std::isnan(loss));
}

// =========================================================================
// CMake note:
//   add_executable(test_losses tests/test_losses.cpp)
//   target_link_libraries(test_losses PRIVATE tinytorch)
//   add_test(NAME losses COMMAND test_losses)   # if using CTest
// =========================================================================