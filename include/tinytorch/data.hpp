#pragma once

#include "tinytorch/tensor.hpp"

#include <algorithm>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace tinytorch {

using Sample = std::vector<Tensor>;
using Batch  = std::vector<Tensor>;

class Dataset {
public:
    virtual ~Dataset() = default;

    virtual size_t size() const = 0;

    virtual Sample get(size_t idx) const = 0;
};


class TensorDataset : public Dataset {
private:
    std::vector<Tensor> tensors;

public:
    explicit TensorDataset(std::vector<Tensor> ts)
        : tensors(std::move(ts))
    {
        if (tensors.empty()) {
            throw std::invalid_argument(
                "TensorDataset requires at least one tensor"
            );
        }

        if (tensors[0].ndim() == 0) {
            throw std::invalid_argument(
                "TensorDataset tensors must have a sample dimension"
            );
        }

        size_t n = tensors[0].shape[0];

        for (size_t i = 1; i < tensors.size(); i++) {
            if (tensors[i].ndim() == 0) {
                throw std::invalid_argument(
                    "TensorDataset tensor is scalar and has no sample dimension"
                );
            }

            if (tensors[i].shape[0] != n) {
                throw std::invalid_argument(
                    "TensorDataset length mismatch\n"
                    "  ❌ Tensor 0 has " + std::to_string(n) +
                    " samples, but tensor " + std::to_string(i) +
                    " has " + std::to_string(tensors[i].shape[0]) + "\n"
                    "  💡 All tensors must share the same first dimension"
                );
            }
        }
    }

    size_t size() const override {
        return tensors[0].shape[0];
    }

    Sample get(size_t idx) const override {
        if (idx >= size()) {
            throw std::out_of_range(
                "TensorDataset index out of range"
            );
        }

        Sample sample;
        sample.reserve(tensors.size());

        for (const auto& t : tensors) {
            sample.push_back(t.index_dim(0, idx));
        }

        return sample;
    }
};
} // namespace tinytorch