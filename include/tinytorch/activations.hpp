#pragma once

#include "tinytorch/tensor.hpp"
#include <cmath>

namespace tinytorch {

struct ReLU {
    Tensor forward(const Tensor& x) const {
        return relu(x);
    }

    Tensor operator()(const Tensor& x) const {
        return forward(x);
    }
};

struct Sigmoid {
    Tensor forward(const Tensor& x) const;
    Tensor operator()(const Tensor& x) const {
        return forward(x);
    }
};

struct Tanh {
    Tensor forward(const Tensor& x) const;
    Tensor operator()(const Tensor& x) const {
        return forward(x);
    }
};

struct GELU {
    Tensor forward(const Tensor& x) const;
    Tensor operator()(const Tensor& x) const {
        return forward(x);
    }
};

struct Softmax {
    Tensor forward(const Tensor& x, int dim = -1) const;
    Tensor operator()(const Tensor& x, int dim = -1) const {
        return forward(x, dim);
    }
};


Tensor Sigmoid::forward(
    const Tensor& x
) const {
    return unary_op(
        x,
        [](float v) {

            if (v >= 0.0f) {
                return 1.0f /
                       (1.0f + std::exp(-v));
            }

            float t = std::exp(v);

            return t / (1.0f + t);
        }
    );
}

Tensor Tanh::forward(
    const Tensor& x
) const {
    return unary_op(
        x,
        [](float v) {
            return std::tanh(v);
        }
    );
}
Tensor GELU::forward(
    const Tensor& x
) const {
    Sigmoid sigmoid;

    return x *
           sigmoid(
               x * 1.702f
           );
}
Tensor Softmax::forward(
    const Tensor& x,
    int dim
) const {

    Tensor m =
        x.max(
            dim,
            true
        );

    Tensor shifted =
        x - m;

    Tensor e =
        exp(shifted);

    Tensor s =
        e.sum(
            dim,
            true
        );

    return e / s;
}



} // namespace tinytorch