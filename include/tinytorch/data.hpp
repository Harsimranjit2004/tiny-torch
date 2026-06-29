#pragma once

#include "tensor.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tinytorch {

using Sample = std::vector<Tensor>;   // one sample  = its per-field tensor slices
using Batch  = std::vector<Tensor>;   // one batch   = per-field stacked tensors

// ---- Dataset interface --------------------------------------------------
class Dataset {
public:
    virtual ~Dataset() = default;
    virtual size_t size() const = 0;            // number of samples
    virtual Sample get(size_t idx) const = 0;   // sample at idx
};

// ---- TensorDataset ------------------------------------------------------
class TensorDataset : public Dataset {
    std::vector<Tensor> tensors;   // each tensor's dim 0 is the sample axis
public:
    explicit TensorDataset(std::vector<Tensor> ts) : tensors(std::move(ts)) {
        if (tensors.empty())
            throw std::invalid_argument("TensorDataset requires >= 1 tensor");
        size_t n = tensors[0].shape[0];
        for (size_t i = 1; i < tensors.size(); i++)
            if (tensors[i].shape[0] != n)
                throw std::invalid_argument(
                    "TensorDataset: all tensors must share dim-0 length");
    }
    size_t size() const override { return tensors[0].shape[0]; }
    Sample get(size_t idx) const override {
        if (idx >= size()) throw std::out_of_range("TensorDataset index oob");
        Sample s; s.reserve(tensors.size());
        for (const auto& t : tensors) s.push_back(t.index_dim(0, idx));
        return s;
    }
};

// ---- stack: join tensors of identical shape along a NEW leading dim ------
inline Tensor stack(const std::vector<Tensor>& tensors) {
    if (tensors.empty()) throw std::invalid_argument("stack needs >= 1 tensor");
    const std::vector<size_t>& base = tensors[0].shape;
    for (const auto& t : tensors)
        if (t.shape != base)
            throw std::invalid_argument("stack: shapes must match");

    std::vector<size_t> out_shape;
    out_shape.push_back(tensors.size());
    for (size_t d : base) out_shape.push_back(d);

    std::vector<float> data;
    data.reserve(tensors.size() * (base.empty() ? 1 : numel_from_shape(base)));
    for (const auto& t : tensors) {
        auto v = t.to_vector();
        data.insert(data.end(), v.begin(), v.end());
    }
    return Tensor(std::move(data), out_shape);
}

// ---- DataLoader ---------------------------------------------------------
class DataLoader {
    const Dataset& dataset;
    size_t batch_size;
    bool shuffle;
    unsigned seed;
    unsigned epoch = 0;     // advanced each begin() so shuffle differs per epoch

    // collate a list of samples into per-field stacked batch tensors
    Batch collate(const std::vector<Sample>& samples) const {
        if (samples.empty()) return {};
        size_t num_fields = samples[0].size();
        std::vector<std::vector<Tensor>> fields(num_fields);
        for (const auto& s : samples) {
            if (s.size() != num_fields)
                throw std::runtime_error("collate: ragged sample field count");
            for (size_t f = 0; f < num_fields; f++) fields[f].push_back(s[f]);
        }
        Batch batch; batch.reserve(num_fields);
        for (size_t f = 0; f < num_fields; f++) batch.push_back(stack(fields[f]));
        return batch;
    }

public:
    DataLoader(const Dataset& ds, size_t bs, bool shuffle_ = false, unsigned seed_ = 42)
        : dataset(ds), batch_size(bs), shuffle(shuffle_), seed(seed_) {
        if (batch_size == 0) throw std::invalid_argument("batch_size must be > 0");
    }

    // number of batches = ceil(N / batch_size)
    size_t size() const {
        size_t n = dataset.size();
        return (n + batch_size - 1) / batch_size;
    }

    // ---- iterator: builds each batch on demand (one batch alive at a time)
    class Iterator {
        const DataLoader* loader = nullptr;
        std::vector<size_t> indices;   // permuted access order
        size_t pos = 0;                // start of the current batch in `indices`
        Batch current;                 // materialized batch at `pos`

        void load() {
            size_t n = indices.size();
            size_t end = std::min(pos + loader->batch_size, n);
            std::vector<Sample> samples;
            samples.reserve(end - pos);
            for (size_t i = pos; i < end; i++)
                samples.push_back(loader->dataset.get(indices[i]));
            current = loader->collate(samples);
        }
    public:
        Iterator() = default;   // end sentinel: loader==nullptr, pos==0
        Iterator(const DataLoader* l, std::vector<size_t> idx)
            : loader(l), indices(std::move(idx)), pos(0) {
            if (!indices.empty()) load();
        }
        Batch& operator*() { return current; }
        Iterator& operator++() {
            pos += loader->batch_size;
            if (pos < indices.size()) load();
            return *this;
        }
        // compare against end(): done when pos has reached the number of samples
        bool operator!=(const Iterator& other) const {
            bool this_done  = (loader == nullptr) || (pos >= indices.size());
            bool other_done = (other.loader == nullptr) || (other.pos >= other.indices.size());
            if (this_done && other_done) return false;   // both at end -> equal
            return pos != other.pos || this_done != other_done;
        }
    };

    Iterator begin() {
        std::vector<size_t> idx(dataset.size());
        std::iota(idx.begin(), idx.end(), 0);            // [0, N)
        if (shuffle) {
            // shuffle INDICES, not data — 8 bytes each, paid once per epoch.
            std::mt19937 rng(seed + epoch);
            std::shuffle(idx.begin(), idx.end(), rng);
        }
        ++epoch;                                         // next pass reshuffles
        return Iterator(this, std::move(idx));
    }
    Iterator end() { return Iterator(); }                // sentinel
};

} // namespace tinytorch