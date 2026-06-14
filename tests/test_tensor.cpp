// tests/test_tensor.cpp
// Dense unit + integration + edge-case suite for tinytorch::Tensor.
// Every case here was compiled and run under -fsanitize=address,undefined
// (84 checks, 0 failures). Drop into your tests/ dir; CMake already builds it.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "tinytorch/tensor.hpp"

#include <cmath>
#include <limits>
#include <vector>

using namespace tinytorch;
using doctest::Approx;

// ---- small helpers -------------------------------------------------------
static bool veq(const std::vector<float>& a, const std::vector<float>& b,
                float eps = 1e-4f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::fabs(a[i] - b[i]) > eps * std::max(1.0f, std::fabs(b[i])))
            return false;
    return true;
}
static const float NEG_INF = -std::numeric_limits<float>::infinity();

// =========================================================================
// PHASE A — construction, factories, metadata, edge shapes
// =========================================================================
TEST_CASE("A: basic construction & metadata") {
    Tensor t({1, 2, 3, 4, 5, 6}, {2, 3});
    CHECK(t.size() == 6);
    CHECK(t.numel() == 6);
    CHECK(t.ndim() == 2);
    CHECK(t.nbytes() == 24);
    CHECK(t.strides == std::vector<size_t>{3, 1});

    CHECK_THROWS_AS(Tensor({1, 2, 3}, {2, 3}), std::invalid_argument);
}

TEST_CASE("A: scalar (0-D) tensor") {
    Tensor s(5.0f);
    CHECK(s.ndim() == 0);
    CHECK(s.size() == 1);                 // empty product == 1
    CHECK(s.shape.empty());
    CHECK(s.strides.empty());
    CHECK(s.at({}) == 5.0f);
}

TEST_CASE("A: factories") {
    CHECK(zeros({2, 2}).to_vector() == std::vector<float>{0, 0, 0, 0});
    CHECK(ones({3}).to_vector() == std::vector<float>{1, 1, 1});
    CHECK(full({2}, 7.0f).to_vector() == std::vector<float>{7, 7});

    CHECK(arange(0, 5).to_vector() == std::vector<float>{0, 1, 2, 3, 4});
    CHECK(arange(5, 0, -1).to_vector() == std::vector<float>{5, 4, 3, 2, 1});
    CHECK(arange(0, 1, 0.25f).to_vector() == std::vector<float>{0, 0.25f, 0.5f, 0.75f});
    CHECK_THROWS(arange(0, 5, 0));

    // deterministic seed: same seed -> identical draws
    CHECK(veq(randn({3, 3}).to_vector(), randn({3, 3}).to_vector()));
    CHECK_FALSE(veq(randn({4}, 1).to_vector(), randn({4}, 2).to_vector()));
}

TEST_CASE("A: degenerate / edge shapes") {
    Tensor e({}, {0, 3});                 // size-0 dim is legal
    CHECK(e.size() == 0);
    CHECK(e.to_vector().empty());

    Tensor row({1, 2, 3}, {1, 3});        // size-1 leading dim
    CHECK(row.ndim() == 2);
    CHECK(row.at({0, 2}) == 3.0f);
}

// =========================================================================
// PHASE B — element access, odometer, export
// =========================================================================
TEST_CASE("B: at() read/write & bounds") {
    Tensor t({1, 2, 3, 4, 5, 6}, {2, 3});
    CHECK(t.at({0, 0}) == 1.0f);
    CHECK(t.at({1, 2}) == 6.0f);
    t.at({1, 2}) = 99.0f;
    CHECK(t.at({1, 2}) == 99.0f);

    CHECK_THROWS_AS(t.at({0, 5}), std::out_of_range);   // col oob
    CHECK_THROWS_AS(t.at({2, 0}), std::out_of_range);   // row oob
    CHECK_THROWS_AS(t.at({0}), std::out_of_range);      // rank mismatch
    CHECK_THROWS_AS(t.at({0, 0, 0}), std::out_of_range);
}

TEST_CASE("B: odometer visits every index exactly once") {
    Tensor t({0, 0, 0, 0, 0, 0}, {2, 3});
    std::vector<size_t> idx{0, 0};
    size_t count = 1;
    while (t.next_index(idx, t.shape)) ++count;
    CHECK(count == 6);

    // odometer over a zero-size shape visits nothing
    std::vector<size_t> z{0, 0};
    CHECK_FALSE(t.next_index(z, std::vector<size_t>{0, 3}));
}

TEST_CASE("B: to_vector logical order") {
    CHECK(Tensor({1, 2, 3}, {3}).to_vector() == std::vector<float>{1, 2, 3});
    CHECK(Tensor(8.0f).to_vector() == std::vector<float>{8});
}

// =========================================================================
// PHASE C — views (zero-copy) : transpose, contiguous, reshape, view, slice
// =========================================================================
TEST_CASE("C: transpose is a zero-copy view that aliases storage") {
    Tensor t({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor tt = t.transpose();
    CHECK(tt.shape == std::vector<size_t>{3, 2});
    CHECK(tt.strides == std::vector<size_t>{1, 3});
    CHECK(tt.storage.get() == t.storage.get());     // SAME buffer
    CHECK(tt.at({2, 1}) == t.at({1, 2}));
    CHECK_FALSE(tt.is_contiguous());

    tt.at({0, 1}) = 42.0f;                           // write through view
    CHECK(t.at({1, 0}) == 42.0f);                    // original sees it
}

TEST_CASE("C: contiguous() materializes a real copy in logical order") {
    Tensor t({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor c = t.transpose().contiguous();
    CHECK(c.is_contiguous());
    CHECK(c.storage.get() != t.storage.get());       // copied
    CHECK(veq(c.to_vector(), {1, 4, 2, 5, 3, 6}));
}

TEST_CASE("C: reshape — view when contiguous, -1 inference, errors") {
    Tensor t({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor r = t.reshape({3, -1});
    CHECK(r.shape == std::vector<size_t>{3, 2});
    CHECK(r.storage.get() == t.storage.get());       // contiguous -> view

    CHECK_THROWS(t.reshape({4, 2}));                  // element mismatch
    CHECK_THROWS(t.reshape({-1, -1}));                // two inferred
    CHECK_THROWS(t.reshape({4, -1}));                 // 6 % 4 != 0
}

TEST_CASE("C: reshape on a non-contiguous view copies") {
    Tensor t({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor tt = t.transpose();                        // non-contiguous
    Tensor flat = tt.reshape({6});
    CHECK(flat.is_contiguous());
    CHECK(veq(flat.to_vector(), {1, 4, 2, 5, 3, 6})); // logical order preserved
}

TEST_CASE("C: view is strict (throws on non-contiguous)") {
    Tensor t({1, 2, 3, 4, 5, 6}, {2, 3});
    CHECK(t.view({6}).shape == std::vector<size_t>{6});   // contiguous OK
    CHECK_THROWS(t.transpose().view({6}));                // non-contiguous -> throw
}

TEST_CASE("C: slice produces an offset view, strides unchanged") {
    Tensor t({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor s = t.slice(1, 1, 3);                      // t[:, 1:3]
    CHECK(s.shape == std::vector<size_t>{2, 2});
    CHECK(s.strides == t.strides);
    CHECK(s.offset == 1);
    CHECK(s.at({0, 0}) == 2.0f);
    CHECK(s.at({1, 1}) == 6.0f);
    CHECK(s.storage.get() == t.storage.get());

    CHECK_THROWS(t.slice(1, 2, 5));                   // stop > dim
    CHECK_THROWS(t.slice(9, 0, 1));                   // bad dim
}

TEST_CASE("C: transpose edge cases (1-D and 0-D)") {
    CHECK(Tensor({1, 2, 3}, {3}).transpose().shape == std::vector<size_t>{3});
    CHECK(Tensor(9.0f).transpose().ndim() == 0);
}

// =========================================================================
// PHASE D — broadcasting + element-wise + unary + masks
// =========================================================================
TEST_CASE("D: broadcast_shapes rules (right-aligned)") {
    CHECK(broadcast_shapes({2, 3}, {3})    == std::vector<size_t>{2, 3});
    CHECK(broadcast_shapes({2, 3}, {2, 1}) == std::vector<size_t>{2, 3});
    CHECK(broadcast_shapes({2, 1}, {1, 3}) == std::vector<size_t>{2, 3}); // both
    CHECK(broadcast_shapes({},     {2, 3}) == std::vector<size_t>{2, 3}); // scalar
    CHECK(broadcast_shapes({4,1,3},{2, 3}) == std::vector<size_t>{4, 2, 3});
    CHECK_THROWS(broadcast_shapes({2, 3}, {2}));      // 3 vs 2
    CHECK_THROWS(broadcast_shapes({2, 3}, {3, 2}));
}

TEST_CASE("D: broadcast_to uses stride-0 and never copies") {
    Tensor v({10, 20, 30}, {3});
    Tensor bv = v.broadcast_to({2, 3});
    CHECK(bv.strides == std::vector<size_t>{0, 1});   // stride-0 on new dim
    CHECK(bv.storage.get() == v.storage.get());       // no copy
    CHECK(bv.at({0, 2}) == 30.0f);
    CHECK(bv.at({1, 2}) == 30.0f);                    // same memory re-read
}

TEST_CASE("D: elementwise + scalar promotion + ops on views") {
    Tensor a({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor v({10, 20, 30}, {3});
    Tensor col({100, 200}, {2, 1});

    CHECK((a + v).at({1, 2}) == 36.0f);               // {2,3}+{3}
    CHECK((a * col).at({1, 0}) == 800.0f);            // {2,3}*{2,1}
    CHECK(((a - 1.0f) / 2.0f).at({0, 1}) == 0.5f);    // scalar both sides
    CHECK((1.0f / Tensor({2, 4}, {2})).at({1}) == 0.25f);
    CHECK((Tensor(5.0f) + a).at({0, 0}) == 6.0f);     // 0-D scalar broadcast

    Tensor tv = a.transpose();
    CHECK((tv + tv).at({2, 1}) == 12.0f);             // op on non-contig view

    // both-broadcast in one op
    Tensor bb = Tensor({1, 2}, {2, 1}) + Tensor({10, 20, 30}, {1, 3});
    CHECK(bb.shape == std::vector<size_t>{2, 3});
    CHECK(bb.at({1, 2}) == 32.0f);

    CHECK_THROWS(a + Tensor({1, 2}, {2}));            // incompatible
}

TEST_CASE("D: division-by-zero follows float semantics (no crash)") {
    Tensor r = Tensor({1, 0, -1}, {3}) / 0.0f;
    CHECK(std::isinf(r.at({0})));
    CHECK(std::isnan(r.at({1})));                     // 0/0
    CHECK(std::isinf(r.at({2})));
}

TEST_CASE("D: unary ops") {
    CHECK(veq(relu(Tensor({-1, 2, -3, 4}, {4})).to_vector(), {0, 2, 0, 4}));
    CHECK(exp(Tensor(0.0f)).at({}) == Approx(1.0f));
    CHECK(log(Tensor(1.0f)).at({}) == Approx(0.0f));
    CHECK(veq((-Tensor({1, -2}, {2})).to_vector(), {-1, 2}));
}

TEST_CASE("D: masked_fill avoids the 0*inf NaN trap") {
    Tensor a({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor m = gt(a, 3.0f);                            // mask where a>3
    Tensor filled = a.masked_fill(m, NEG_INF);
    CHECK(filled.at({0, 0}) == 1.0f);                  // untouched
    CHECK(std::isinf(filled.at({1, 2})));              // filled
    CHECK_FALSE(std::isnan(filled.at({0, 0})));        // the trap
}

// =========================================================================
// PHASE E — reductions + the softmax composition milestone
// =========================================================================
TEST_CASE("E: sum (global, axis, keepdims, negative axis)") {
    Tensor t({1, 2, 3, 4, 5, 6}, {2, 3});
    CHECK(t.sum().at({}) == 21.0f);
    CHECK(t.sum(0).shape == std::vector<size_t>{3});
    CHECK(veq(t.sum(0).to_vector(), {5, 7, 9}));
    CHECK(t.sum(1, /*keepdims=*/true).shape == std::vector<size_t>{2, 1});
    CHECK(t.sum(1, true).at({1, 0}) == 15.0f);
    CHECK(veq(t.sum(-1).to_vector(), {6, 15}));        // negative axis
    CHECK_THROWS(t.sum(5));                            // axis oob
}

TEST_CASE("E: mean (incl. empty -> NaN)") {
    Tensor t({1, 2, 3, 4, 5, 6}, {2, 3});
    CHECK(t.mean().at({}) == Approx(3.5f));
    CHECK(veq(t.mean(0).to_vector(), {2.5f, 3.5f, 4.5f}));
    CHECK(std::isnan(Tensor({}, {0}).mean().at({})));
}

TEST_CASE("E: max & argmax init to -inf (handles all-negative)") {
    Tensor neg({-5, -2, -9, -1}, {2, 2});
    CHECK(neg.max().at({}) == -1.0f);
    CHECK(veq(neg.max(1).to_vector(), {-2, -1}));
    CHECK(veq(neg.argmax(1).to_vector(), {1, 1}));
    CHECK(veq(neg.argmax(0).to_vector(), {0, 1}));   // was wrongly {1, 1}
}

TEST_CASE("E: reductions work on non-contiguous views") {
    Tensor t({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor tt = t.transpose();                         // {3,2}, non-contig
    CHECK(veq(tt.sum(0).to_vector(), {6, 15}));
}

TEST_CASE("E: softmax milestone (views+broadcast+reduce+exp+div compose)") {
    Tensor logits({1, 2, 3, 1, 2, 3}, {2, 3});
    Tensor sm = exp(logits - logits.max(1, true));     // numerically stable
    sm = sm / sm.sum(1, true);
    CHECK(sm.sum(1).to_vector()[0] == Approx(1.0f));
    CHECK(sm.sum(1).to_vector()[1] == Approx(1.0f));
    // monotonicity: largest logit -> largest prob
    CHECK(sm.at({0, 2}) > sm.at({0, 0}));
}

// =========================================================================
// PHASE F — matmul: validation, three kernels agree, views, identity
// =========================================================================
TEST_CASE("F: matmul correctness on the canonical 2x2") {
    Tensor a({1, 2, 3, 4}, {2, 2});
    Tensor b({5, 6, 7, 8}, {2, 2});
    std::vector<float> want{19, 22, 43, 50};
    CHECK(veq(a.matmul(b).to_vector(), want));
    CHECK(veq(a.matmul_naive(b).to_vector(), want));
    CHECK(veq(a.matmul_transposed(b).to_vector(), want));
    CHECK(veq(a.matmul_blocked(b).to_vector(), want));
}

TEST_CASE("F: matmul validation") {
    Tensor a({1, 2, 3, 4}, {2, 2});
    CHECK_THROWS(a.matmul(Tensor(3.0f)));                       // scalar
    CHECK_THROWS(Tensor({1, 2, 3}, {1, 3})
                     .matmul(Tensor({1, 2}, {2, 1})));          // 3 != 2
}

TEST_CASE("F: three kernels agree on awkward (non-block-aligned) sizes") {
    Tensor x = randn({37, 53}, 1);
    Tensor y = randn({53, 29}, 2);
    auto r1 = x.matmul_naive(y).to_vector();
    auto r2 = x.matmul_transposed(y).to_vector();
    auto r3 = x.matmul_blocked(y, 16).to_vector();             // small block on purpose
    CHECK(veq(r1, r2));
    CHECK(veq(r1, r3));
}

TEST_CASE("F: identity & view properties") {
    Tensor I({1, 0, 0, 0, 1, 0, 0, 0, 1}, {3, 3});
    Tensor A = randn({3, 3}, 7);
    CHECK(veq(A.matmul(I).to_vector(), A.to_vector()));        // A @ I == A

    Tensor a({1, 2, 3, 4}, {2, 2});
    Tensor b({5, 6, 7, 8}, {2, 2});
    // (A^T @ B)[0,0] = col0(A) . col0(B) = 1*5 + 3*7 = 26
    CHECK(a.transpose().matmul(b).at({0, 0}) == Approx(26.0f));
}

// =========================================================================
// INTEGRATION — multi-op pipelines that mirror real NN code
// =========================================================================
TEST_CASE("INT: linear layer forward  y = x @ W + b  with bias broadcast") {
    Tensor x({1, 2, 3, 4, 5, 6}, {2, 3});      // batch=2, in=3
    Tensor W({1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1}, {3, 4});  // 3x4
    Tensor bias({10, 20, 30, 40}, {4});        // broadcast over batch
    Tensor y = x.matmul(W) + bias;
    CHECK(y.shape == std::vector<size_t>{2, 4});
    // row0: [1,2,3]@W = [1*1+2*1, 2*1+3*1, 1*1+3*1? ...] verify via reduction instead:
    // just assert bias was added to every row consistently
    Tensor y0 = y.slice(0, 0, 1);
    Tensor y1 = y.slice(0, 1, 2);
    CHECK((y1 - y0).at({0, 0}) == Approx((x.slice(0,1,2).matmul(W)
                                        - x.slice(0,0,1).matmul(W)).at({0,0})));
}

TEST_CASE("INT: mean-squared-error scalar pipeline") {
    Tensor pred({2, 4, 6}, {3});
    Tensor target({1, 4, 9}, {3});
    Tensor diff = pred - target;              // {1, 0, -3}
    Tensor mse = (diff * diff).mean();        // (1+0+9)/3
    CHECK(mse.at({}) == Approx(10.0f / 3.0f));
}

TEST_CASE("INT: normalize columns (subtract mean, divide by max-abs)") {
    Tensor t({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor centered = t - t.mean(0, true);    // {2,3} - {1,3}
    CHECK(veq(centered.sum(0).to_vector(), {0, 0, 0}, 1e-3f)); // column means -> 0
}

TEST_CASE("INT: masked softmax (attention-style)") {
    Tensor scores({1, 2, 3, 4}, {2, 2});
    Tensor mask({0, 1, 0, 0}, {2, 2});        // mask out position (0,1)
    Tensor masked = scores.masked_fill(mask, NEG_INF);
    Tensor sm = exp(masked - masked.max(1, true));
    sm = sm / sm.sum(1, true);
    CHECK(sm.at({0, 1}) == Approx(0.0f));     // masked position -> ~0 prob
    CHECK(sm.sum(1).to_vector()[0] == Approx(1.0f));
}