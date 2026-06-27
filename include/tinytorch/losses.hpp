#pragma once

#include "tinytorch/tensor.hpp"

namespace tinytorch {



inline Tensor log_softmax(
    const Tensor& x,
    int dim = -1
) {

    Tensor m =
        x.max(
            dim,
            true
        );

    Tensor shifted =
        x - m;

    Tensor lse =
        log(
            exp(
                shifted
            ).sum(
                dim,
                true
            )
        );

    return shifted - lse;
}


inline Tensor gather_rows(
    const Tensor& x,
    const std::vector<size_t>& indices
) {
    if (x.ndim() != 2) {
        throw std::invalid_argument(
            "gather_rows expects 2D tensor"
        );
    }

    size_t B = x.shape[0];

    if (indices.size() != B) {
        throw std::invalid_argument(
            "Index count must match batch size"
        );
    }

    std::vector<float> out(B);

    for (size_t i = 0; i < B; i++) {
        out[i] =
            x.at(
                {i, indices[i]}
            );
    }

    return Tensor(
        std::move(out),
        {B}
    );
}


struct MSELoss {
    Tensor forward(const Tensor& pred,
                   const Tensor& target) const {
        if (pred.shape != target.shape) {
            throw std::invalid_argument(
                "MSELoss shape mismatch\n"
                "  ❌ Prediction and target shapes must match"
            );
        }

        Tensor diff = pred - target;

        return (diff * diff).mean();
    }

    Tensor operator()(const Tensor& pred,
                      const Tensor& target) const {
        return forward(pred, target);
    }
};

struct CrossEntropyLoss {

    Tensor forward(
        const Tensor& logits,
        const std::vector<size_t>& targets
    ) const {

        if (logits.ndim() != 2) {
            throw std::invalid_argument(
                "CrossEntropy expects logits shape {batch, classes}"
            );
        }

        size_t B = logits.shape[0];
        size_t C = logits.shape[1];

        if (targets.size() != B) {
            throw std::invalid_argument(
                "Target count must equal batch size"
            );
        }

        for (auto t : targets) {
            if (t >= C) {
                throw std::invalid_argument(
                    "Target index out of bounds"
                );
            }
        }

        Tensor lp =
            log_softmax(
                logits,
                1
            );

        Tensor selected =
            gather_rows(
                lp,
                targets
            );

        return -selected.mean();
    }

    Tensor operator()(
        const Tensor& logits,
        const std::vector<size_t>& targets
    ) const {
        return forward(
            logits,
            targets
        );
    }
};
struct BinaryCrossEntropyLoss {
    Tensor forward(const Tensor& pred,
                   const Tensor& target) const {
        if (pred.shape != target.shape) {
            throw std::invalid_argument(
                "BinaryCrossEntropyLoss shape mismatch\n"
                "  ❌ Prediction and target shapes must match"
            );
        }

        float eps = 1e-7f;

        Tensor p = unary_op(pred, [eps](float v) {
            if (v < eps) return eps;
            if (v > 1.0f - eps) return 1.0f - eps;
            return v;
        });

        Tensor loss =
            -(target * log(p) +
              (1.0f - target) * log(1.0f - p));

        return loss.mean();
    }

    Tensor operator()(const Tensor& pred,
                      const Tensor& target) const {
        return forward(pred, target);
    }
};

}