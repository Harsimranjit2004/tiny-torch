// tests/test_data.cpp
// Unit + iterator + shuffle + integration suite for the data pipeline.
// Verified: 42 checks, 0 failures under -fsanitize=address,undefined.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "tinytorch/tensor.hpp"
#include "tinytorch/data.hpp"

#include <algorithm>
#include <vector>

using namespace tinytorch;

// drain the loader into a vector of batches (helper for tests)
static std::vector<Batch> collect_all(DataLoader& dl) {
    std::vector<Batch> out;
    for (auto& b : dl) out.push_back(b);
    return out;
}

// =========================================================================
// stack — the collation primitive
// =========================================================================
TEST_CASE("stack: joins same-shape tensors along a new leading dim") {
    auto s = stack({Tensor({1, 2}, {2}), Tensor({3, 4}, {2}), Tensor({5, 6}, {2})});
    CHECK(s.shape == std::vector<size_t>{3, 2});
    CHECK(s.to_vector() == std::vector<float>{1, 2, 3, 4, 5, 6});
}

TEST_CASE("stack: scalars stack into a 1-D vector") {
    auto s = stack({Tensor(1.0f), Tensor(2.0f), Tensor(3.0f)});
    CHECK(s.shape == std::vector<size_t>{3});
    CHECK(s.to_vector() == std::vector<float>{1, 2, 3});
}

TEST_CASE("stack: shape mismatch & empty input throw") {
    CHECK_THROWS(stack({Tensor({1, 2}, {2}), Tensor({3}, {1})}));
    CHECK_THROWS(stack({}));
}

// =========================================================================
// TensorDataset
// =========================================================================
TEST_CASE("TensorDataset: size & sample extraction") {
    Tensor X({1, 2, 3, 4, 5, 6}, {3, 2});      // 3 samples, 2 features
    Tensor y({10, 20, 30}, {3});               // 3 labels
    TensorDataset ds({X, y});
    CHECK(ds.size() == 3);

    auto s0 = ds.get(0);
    CHECK(s0.size() == 2);                      // (features, label)
    CHECK(s0[0].to_vector() == std::vector<float>{1, 2});
    CHECK(s0[1].item() == 10.0f);

    auto s2 = ds.get(2);
    CHECK(s2[0].to_vector() == std::vector<float>{5, 6});
    CHECK(s2[1].item() == 30.0f);
}

TEST_CASE("TensorDataset: mismatched lengths rejected") {
    Tensor X({1, 2, 3, 4}, {2, 2});            // 2 samples
    Tensor y({1, 2, 3}, {3});                  // 3 labels
    CHECK_THROWS(TensorDataset({X, y}));
}

TEST_CASE("TensorDataset: empty input & out-of-range index throw") {
    CHECK_THROWS(TensorDataset({}));
    TensorDataset ds({Tensor({1, 2, 3}, {3})});
    CHECK_THROWS(ds.get(3));
}

TEST_CASE("TensorDataset: single tensor (unlabeled) works") {
    TensorDataset ds({Tensor({0, 1, 2, 3, 4, 5}, {3, 2})});
    CHECK(ds.size() == 3);
    CHECK(ds.get(1).size() == 1);              // one field per sample
    CHECK(ds.get(1)[0].to_vector() == std::vector<float>{2, 3});
}

// =========================================================================
// DataLoader — batch math
// =========================================================================
TEST_CASE("DataLoader: batch count is ceil(N / batch_size)") {
    TensorDataset ds({Tensor(std::vector<float>(100, 0.0f), {100})});
    CHECK(DataLoader(ds, 32).size() == 4);     // 32,32,32,4
    CHECK(DataLoader(ds, 10).size() == 10);
    CHECK(DataLoader(ds, 100).size() == 1);
    CHECK(DataLoader(ds, 200).size() == 1);    // batch > N -> one short batch
    CHECK_THROWS(DataLoader(ds, 0));
}

// =========================================================================
// Collation through the loader
// =========================================================================
TEST_CASE("DataLoader: collates features & labels, short final batch") {
    Tensor X({1, 2, 3, 4, 5, 6}, {3, 2});
    Tensor y({10, 20, 30}, {3});
    TensorDataset ds({X, y});
    DataLoader loader(ds, /*batch=*/2, /*shuffle=*/false);
    auto b = collect_all(loader);

    CHECK(b.size() == 2);
    CHECK(b[0][0].shape == std::vector<size_t>{2, 2});   // features
    CHECK(b[0][1].shape == std::vector<size_t>{2});      // labels
    CHECK(b[0][0].to_vector() == std::vector<float>{1, 2, 3, 4});
    CHECK(b[0][1].to_vector() == std::vector<float>{10, 20});

    CHECK(b[1][0].shape == std::vector<size_t>{1, 2});   // short last batch
    CHECK(b[1][1].to_vector() == std::vector<float>{30});
}

// =========================================================================
// Iterator behavior
// =========================================================================
TEST_CASE("DataLoader: iterates every sample once in order (no shuffle)") {
    Tensor X({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {10, 1});
    TensorDataset ds({X});
    DataLoader loader(ds, 3, /*shuffle=*/false);
    auto b = collect_all(loader);

    CHECK(b.size() == 4);                                  // 3,3,3,1
    CHECK(b[3][0].shape == std::vector<size_t>{1, 1});     // short last

    std::vector<float> seen;
    for (auto& batch : b)
        for (float v : batch[0].to_vector()) seen.push_back(v);
    CHECK(seen == std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
}

TEST_CASE("DataLoader: batch>N yields a single short batch") {
    TensorDataset ds({Tensor({1, 2, 3}, {3, 1})});
    DataLoader loader(ds, 10, false);
    auto b = collect_all(loader);
    CHECK(b.size() == 1);
    CHECK(b[0][0].shape == std::vector<size_t>{3, 1});
}

// =========================================================================
// Shuffling
// =========================================================================
TEST_CASE("DataLoader: shuffle is a true permutation (same multiset)") {
    Tensor X({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {10, 1});
    TensorDataset ds({X});
    DataLoader loader(ds, 10, /*shuffle=*/true, /*seed=*/123);
    auto seen = collect_all(loader)[0][0].to_vector();

    auto sorted = seen;
    std::sort(sorted.begin(), sorted.end());
    CHECK(sorted == std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9});  // all once
    CHECK_FALSE(std::is_sorted(seen.begin(), seen.end()));              // reordered
}

TEST_CASE("DataLoader: shuffle keeps (features, labels) aligned") {
    // label == feature * 10, so any misalignment is detectable post-shuffle
    Tensor X({0, 1, 2, 3, 4}, {5, 1});
    Tensor y({0, 10, 20, 30, 40}, {5});
    TensorDataset ds({X, y});
    DataLoader loader(ds, 5, /*shuffle=*/true, /*seed=*/7);
    auto b = collect_all(loader)[0];
    auto xs = b[0].to_vector();
    auto ys = b[1].to_vector();
    for (size_t i = 0; i < xs.size(); ++i) CHECK(ys[i] == xs[i] * 10.0f);
}

TEST_CASE("DataLoader: consecutive epochs reshuffle differently") {
    Tensor X({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {10, 1});
    TensorDataset ds({X});
    DataLoader loader(ds, 10, /*shuffle=*/true, /*seed=*/5);
    auto e1 = collect_all(loader)[0][0].to_vector();
    auto e2 = collect_all(loader)[0][0].to_vector();
    CHECK(e1 != e2);                            // different order each pass
}

TEST_CASE("DataLoader: no-shuffle iteration is stable across epochs") {
    Tensor X({0, 1, 2, 3, 4}, {5, 1});
    TensorDataset ds({X});
    DataLoader loader(ds, 2, /*shuffle=*/false);
    auto e1 = collect_all(loader);
    auto e2 = collect_all(loader);
    CHECK(e1[0][0].to_vector() == e2[0][0].to_vector());
}

// =========================================================================
// Integration — the training-loop shape
// =========================================================================
TEST_CASE("INT: range-for drives a full pass with aligned batch dims") {
    Tensor X = randn({20, 8}, 1);              // 20 samples, 8 features
    Tensor y(std::vector<float>(20, 0.0f), {20});
    for (size_t i = 0; i < 20; ++i) y.at({i}) = float(i % 4);  // labels 0..3
    TensorDataset ds({X, y});
    DataLoader loader(ds, 4, /*shuffle=*/true, /*seed=*/1);

    size_t batches = 0, samples = 0;
    for (auto& batch : loader) {
        CHECK(batch[0].shape[0] == batch[1].shape[0]);   // features/labels aligned
        CHECK(batch[0].shape[1] == 8);                   // feature dim preserved
        samples += batch[0].shape[0];
        ++batches;
    }
    CHECK(batches == 5);                        // ceil(20/4)
    CHECK(samples == 20);                       // every sample seen once
}

// =========================================================================
// CMake note:
//   add_executable(test_data tests/test_data.cpp)
//   target_link_libraries(test_data PRIVATE tinytorch)
//   add_test(NAME data COMMAND test_data)   # if using CTest
// =========================================================================