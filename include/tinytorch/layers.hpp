#pragma once

#include "tinytorch/tensor.hpp"
#include "tinytorch/activations.hpp"
#include <random>
#include <vector>
#include <memory>
#include <cmath>
#include <stdexcept>

namespace tinytorch {

class Layer {
public:
    virtual ~Layer() = default;

    virtual Tensor forward(const Tensor& x) = 0;

    virtual std::vector<Tensor> parameters() {
        return {};
    }

    Tensor operator()(const Tensor& x) {
        return forward(x);
    }
};


class Linear : public Layer {
public:
    size_t in_features;
    size_t out_features;

    Tensor weight;
    Tensor bias;

    bool has_bias;

    Linear(size_t in_features_,
           size_t out_features_,
           bool use_bias = true,
           unsigned seed = 42)
        : in_features(in_features_),
          out_features(out_features_),
          weight(),
          bias(),
          has_bias(use_bias)
    {
        if (in_features == 0 || out_features == 0) {
            throw std::invalid_argument(
                "Linear layer dimensions must be non-zero"
            );
        }

        float scale =
            std::sqrt(1.0f / static_cast<float>(in_features));

        weight =
            randn({in_features, out_features}, seed) * scale;

        if (has_bias) {
            bias = zeros({out_features});
        }
    }

    Tensor forward(const Tensor& x) override {
        if (x.ndim() != 2) {
            throw std::invalid_argument(
                "Linear forward expects 2D input {batch, in_features}"
            );
        }

        if (x.shape[1] != in_features) {
            throw std::invalid_argument(
                "Linear input feature mismatch"
            );
        }

        Tensor y = x.matmul(weight);

        if (has_bias) {
            y = y + bias;
        }

        return y;
    }

    std::vector<Tensor> parameters() override {
        if (has_bias) {
            return {weight, bias};
        }

        return {weight};
    }
};


class Dropout : public Layer {
public:
    float p;
    std::mt19937 rng;

    explicit Dropout(float p_ = 0.5f,
                     unsigned seed = 42)
        : p(p_),
          rng(seed)
    {
        if (p < 0.0f || p > 1.0f) {
            throw std::invalid_argument(
                "Dropout probability must be between 0 and 1"
            );
        }
    }

    Tensor forward(const Tensor& x) override {
        return forward(x, true);
    }

    Tensor forward(const Tensor& x,
                   bool training) {
        if (!training || p == 0.0f) {
            return x.shallow_copy();
        }

        if (p == 1.0f) {
            return zeros(x.shape);
        }

        float keep = 1.0f - p;
        float scale = 1.0f / keep;

        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        std::vector<float> mask_data;
        mask_data.reserve(x.numel());

        for (size_t i = 0; i < x.numel(); i++) {
            float r = dist(rng);

            if (r < keep) {
                mask_data.push_back(scale);
            } else {
                mask_data.push_back(0.0f);
            }
        }

        Tensor mask(std::move(mask_data), x.shape);

        return x * mask;
    }
};



class Sequential : public Layer {
public:
    std::vector<std::shared_ptr<Layer>> layers;

    Sequential() = default;

    void add(std::shared_ptr<Layer> layer) {
        layers.push_back(std::move(layer));
    }

    Tensor forward(const Tensor& x) override {
        Tensor out = x;

        for (auto& layer : layers) {
            out = layer->forward(out);
        }

        return out;
    }

    std::vector<Tensor> parameters() override {
        std::vector<Tensor> params;

        for (auto& layer : layers) {
            auto layer_params = layer->parameters();

            params.insert(
                params.end(),
                layer_params.begin(),
                layer_params.end()
            );
        }

        return params;
    }
};

class ReLULayer : public Layer {
public:
    ReLU act;

    Tensor forward(
        const Tensor& x
    ) override {
        return act(x);
    }
};

class SigmoidLayer : public Layer {
public:
    Sigmoid act;

    Tensor forward(
        const Tensor& x
    ) override {
        return act(x);
    }
};

class TanhLayer : public Layer {
public:
    Tanh act;

    Tensor forward(
        const Tensor& x
    ) override {
        return act(x);
    }
};

class GELULayer : public Layer {
public:
    GELU act;

    Tensor forward(
        const Tensor& x
    ) override {
        return act(x);
    }
};

class SoftmaxLayer : public Layer {
public:
    int dim;

    explicit SoftmaxLayer(
        int dim_ = -1
    )
        : dim(dim_)
    {}

    Tensor forward(
        const Tensor& x
    ) override {
        Softmax sm;

        return sm(
            x,
            dim
        );
    }
};
} // namespace tinytorch