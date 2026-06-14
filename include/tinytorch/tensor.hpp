#pragma once
#include <algorithm>
#include <cassert>
#include <optional>
#include <cstddef>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <cmath>
#include <limits>
#include <Eigen/Dense>
namespace tinytorch {

static size_t numel_from_shape(const std::vector<size_t>& shape) {
    size_t total = 1;
    for (size_t dim : shape) {
        total *= dim;
    }
    return total;
}
class Tensor;
inline std::vector<size_t> broadcast_shapes(const std::vector<size_t>& a,
                                            const std::vector<size_t>& b);
inline Tensor zeros(const std::vector<size_t>& shape);
inline Tensor ones(const std::vector<size_t>& shape);
inline Tensor full(const std::vector<size_t>& shape, float value);
inline Tensor operator/(const Tensor& a, const Tensor& b);
inline Tensor operator/(const Tensor& a, float b);
// (the full definitions remain below, unchanged)

class Tensor {
private:
    size_t flat_index(const std::vector<size_t>& idx) const {
        if (idx.size() != shape.size()) {
            throw std::out_of_range(
                "Tensor indexing rank mismatch\n"
                "  ❌ Got " + std::to_string(idx.size()) + " indices, but tensor has " +
                std::to_string(shape.size()) + " dimensions\n"
                "  💡 Number of indices must match tensor ndim()\n"
                "  🔧 Fix: provide exactly " + std::to_string(shape.size()) + " indices"
            );
        }

        size_t flat = offset;

        for (size_t d = 0; d < idx.size(); d++) {
            if (idx[d] >= shape[d]) {
                throw std::out_of_range(
                    "Tensor index out of bounds\n"
                    "  ❌ Index " + std::to_string(idx[d]) + " is out of range for dim " +
                    std::to_string(d) + " with size " + std::to_string(shape[d]) + "\n"
                    "  💡 Valid indices are 0 to " + std::to_string(shape[d] - 1) + "\n"
                    "  🔧 Fix: use a smaller index"
                );
            }

            flat += idx[d] * strides[d];
        }

        return flat;
    }

    Tensor(std::shared_ptr<std::vector<float>> storage_,
            std::vector<size_t> shape_,
            std::vector<size_t> strides_,
            size_t offset_)
            : storage(std::move(storage_)),
            shape(std::move(shape_)),
            strides(std::move(strides_)),
            offset(offset_) {
            check_invariants();
        }
        int normalize_axis(int axis) const {
        int n = static_cast<int>(ndim());

        if (axis < 0) {
            axis += n;
        }

        if (axis < 0 || axis >= n) {
            throw std::out_of_range(
                "Axis out of range\n"
                "  ❌ Invalid axis " + std::to_string(axis) + "\n"
                "  💡 Axis must be in [-ndim(), ndim()-1]"
            );
        }

        return axis;
    }
    std::vector<size_t> reduction_shape(int axis, bool keepdims) const {
        std::vector<size_t> out_shape = shape;

        if (keepdims) {
            out_shape[axis] = 1;
        } else {
            out_shape.erase(out_shape.begin() + axis);
        }

        return out_shape;
    }
    std::vector<size_t> reduction_index(
        const std::vector<size_t>& in_idx,
        int axis,
        bool keepdims
    ) const {
        std::vector<size_t> out_idx;

        for (size_t d = 0; d < in_idx.size(); d++) {
            if (static_cast<int>(d) == axis) {
                if (keepdims) {
                    out_idx.push_back(0);
                }
            } else {
                out_idx.push_back(in_idx[d]);
            }
        }

        return out_idx;
    }

public:
    std::shared_ptr<std::vector<float>> storage;
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    size_t offset = 0;

    Tensor() {
        storage = std::make_shared<std::vector<float>>();
        shape = {0};
        strides = {1};
        offset = 0;
    }

    Tensor(std::vector<float> data, std::vector<size_t> shape_)
        : storage(nullptr),
          shape(std::move(shape_)),
          strides{},
          offset(0) {
        size_t expected = numel_from_shape(shape);
        size_t actual = data.size();

        if (actual != expected) {
            throw std::invalid_argument(
                "Tensor data/shape mismatch\n"
                "  ❌ Got " + std::to_string(actual) + " values, but shape requires " +
                std::to_string(expected) + " values\n"
                "  💡 data.size() must equal product(shape)\n"
                "  🔧 Fix: change the shape or provide exactly " +
                std::to_string(expected) + " values"
            );
        }

        storage = std::make_shared<std::vector<float>>(std::move(data));
        strides = compute_strides(shape);

        check_invariants();
    }

    // Intentional non-explicit scalar constructor:
    // later this allows scalar promotion such as tensor + 2.0f.
    Tensor(float scalar)
        : storage(std::make_shared<std::vector<float>>(std::vector<float>{scalar})),
          shape({}),
          strides({}),
          offset(0) {
        check_invariants();
    }

    void check_invariants() const {
        assert(storage != nullptr);
        assert(shape.size() == strides.size());
    }

    std::vector<size_t> compute_strides(const std::vector<size_t>& s) const {
        std::vector<size_t> out(s.size());
        size_t running_product = 1;

        for (int i = static_cast<int>(s.size()) - 1; i >= 0; --i) {
            out[i] = running_product;
            running_product *= s[i];
        }

        return out;
    }

    Tensor shallow_copy() const {
        Tensor out;
        out.storage = storage;
        out.shape = shape;
        out.strides = strides;
        out.offset = offset;
        return out;
    }

    size_t ndim() const {
        return shape.size();
    }

    size_t numel() const {
        return numel_from_shape(shape);
    }

    size_t size() const {
        return numel();
    }

    size_t nbytes() const {
        return numel() * sizeof(float);
    }

    float& at(const std::vector<size_t>& idx) {
        return (*storage)[flat_index(idx)];
    }

    const float& at(const std::vector<size_t>& idx) const {
        return (*storage)[flat_index(idx)];
    }

    bool next_index(std::vector<size_t>& idx,
                    const std::vector<size_t>& s) const {
        for (size_t dim : s) {
            if (dim == 0) return false;
        }

        if (s.empty()) {
            return false;
        }

        for (int d = static_cast<int>(s.size()) - 1; d >= 0; --d) {
            idx[d]++;

            if (idx[d] < s[d]) {
                return true;
            }

            idx[d] = 0;
        }

        return false;
    }

    std::vector<float> to_vector() const {
        std::vector<float> out;
        out.reserve(numel());

        if (numel() == 0) {
            return out;
        }

        if (ndim() == 0) {
            out.push_back(at({}));
            return out;
        }

        std::vector<size_t> idx(shape.size(), 0);

        out.push_back(at(idx));

        while (next_index(idx, shape)) {
            out.push_back(at(idx));
        }

        return out;
    }

    Tensor transpose(int d0 = -2, int d1 = -1) const {
        if (ndim() < 2 && d0 == -2 && d1 == -1) {
            return shallow_copy();
        }

        int n = static_cast<int>(ndim());

        if (d0 < 0) d0 += n;
        if (d1 < 0) d1 += n;

        if (d0 < 0 || d0 >= n || d1 < 0 || d1 >= n) {
            throw std::out_of_range(
                "Transpose dimension out of range\n"
                "  ❌ Invalid dimensions\n"
                "  💡 Dimensions must be in [-ndim(), ndim()-1]"
            );
        }

        Tensor out = shallow_copy();

        std::swap(out.shape[d0], out.shape[d1]);
        std::swap(out.strides[d0], out.strides[d1]);

        return out;
    }

    bool is_contiguous() const {
        if (numel() <= 1) {
            return true;
        }

        // Simple correct check.
        // Note: shapes with size-1 dims can be contiguous with unusual strides.
        // This may give false negatives, causing unnecessary copies, but not wrong results.
        return strides == compute_strides(shape);
    }

    Tensor contiguous() const {
        if (is_contiguous()) {
            return shallow_copy();
        }

        std::vector<float> data;
        data.reserve(numel());

        if (numel() == 0) {
            return Tensor(std::move(data), shape);
        }

        if (ndim() == 0) {
            data.push_back(at({}));
            return Tensor(std::move(data), shape);
        }

        std::vector<size_t> idx(shape.size(), 0);

        data.push_back(at(idx));

        while (next_index(idx, shape)) {
            data.push_back(at(idx));
        }

        return Tensor(std::move(data), shape);
    }

    Tensor reshape(const std::vector<long>& new_shape) const {
        int infer_count = 0;
        long infer_pos = -1;

        size_t known_product = 1;

        for (size_t i = 0; i < new_shape.size(); i++) {
            if (new_shape[i] == -1) {
                infer_count++;
                infer_pos = static_cast<long>(i);
            }
            else if (new_shape[i] <= 0) {
                throw std::invalid_argument(
                    "reshape: dimensions must be positive or -1"
                );
            }
            else {
                known_product *= static_cast<size_t>(new_shape[i]);
            }
        }

        if (infer_count > 1) {
            throw std::invalid_argument(
                "reshape: only one dimension can be inferred"
            );
        }

        std::vector<size_t> final_shape;
        final_shape.reserve(new_shape.size());

        if (infer_count == 1) {
            if (known_product == 0 || numel() % known_product != 0) {
                throw std::invalid_argument(
                    "reshape: cannot infer dimension"
                );
            }

            size_t inferred = numel() / known_product;

            for (size_t i = 0; i < new_shape.size(); i++) {
                if (static_cast<long>(i) == infer_pos) {
                    final_shape.push_back(inferred);
                } else {
                    final_shape.push_back(
                        static_cast<size_t>(new_shape[i])
                    );
                }
            }
        }
        else {
            for (long dim : new_shape) {
                final_shape.push_back(
                    static_cast<size_t>(dim)
                );
            }
        }

        size_t new_numel = numel_from_shape(final_shape);

        if (new_numel != numel()) {
            throw std::invalid_argument(
                "reshape element mismatch\n"
                "  ❌ Current tensor has " + std::to_string(numel()) +
                " elements\n"
                "  ❌ Requested shape requires " +
                std::to_string(new_numel) + " elements"
            );
        }

        // PyTorch reshape semantics:
        // contiguous -> view
        // non-contiguous -> copy then reshape
        if (!is_contiguous()) {
            return contiguous().reshape(new_shape);
        }

        Tensor out = shallow_copy();

        out.shape = final_shape;
        out.strides = compute_strides(final_shape);

        return out;
    }
    Tensor view(const std::vector<long>& new_shape) const {
        if (!is_contiguous()) {
            throw std::invalid_argument(
                "view requires a contiguous tensor\n"
                "  ❌ This tensor is not contiguous\n"
                "  💡 view() never copies data, so it cannot reshape non-contiguous tensors\n"
                "  🔧 Fix: call tensor.contiguous().view(...) or use tensor.reshape(...)"
            );
        }

        return reshape(new_shape);
    }
    Tensor slice(size_t dim,
                size_t start,
                size_t stop) const
    {
        if (dim >= ndim()) {
            throw std::out_of_range("slice: invalid dimension");
        }

        if (!(start <= stop && stop <= shape[dim])) {
            throw std::out_of_range(
                "slice: require start <= stop <= shape[dim]"
            );
        }

        Tensor out = shallow_copy();

        out.offset += start * strides[dim];

        out.shape[dim] = stop - start;

        return out;
    }
    std::string vec_to_string(const std::vector<size_t>& v) const {
        std::ostringstream oss;
        oss << "{";

        for (size_t i = 0; i < v.size(); i++) {
            oss << v[i];
            if (i + 1 < v.size()) oss << ",";
        }

        oss << "}";
        return oss.str();
    }

    std::string flat_data_string() const {
        std::ostringstream oss;
        oss << "[";

        // TODO Phase C: route through at()/to_vector()
        // so non-contiguous views print logical values correctly.
        for (size_t i = 0; i < numel(); i++) {
            oss << (*storage)[offset + i];
            if (i + 1 < numel()) oss << ", ";
        }

        oss << "]";
        return oss.str();
    }

    std::string repr() const {
        std::ostringstream oss;

        oss << "Tensor("
            << "shape=" << vec_to_string(shape)
            << ", strides=" << vec_to_string(strides)
            << ", offset=" << offset
            << ", data=" << flat_data_string()
            << ")";

        return oss.str();
    }

    std::string str() const {
        std::ostringstream oss;

        if (ndim() == 0) {
            oss << at({});
        } else if (ndim() == 1) {
            oss << "[";
            for (size_t i = 0; i < shape[0]; i++) {
                oss << at({i});
                if (i + 1 < shape[0]) oss << ", ";
            }
            oss << "]";
        } else if (ndim() == 2) {
            oss << "[\n";

            for (size_t i = 0; i < shape[0]; i++) {
                oss << "  [";

                for (size_t j = 0; j < shape[1]; j++) {
                    oss << at({i, j});
                    if (j + 1 < shape[1]) oss << ", ";
                }

                oss << "]";
                if (i + 1 < shape[0]) oss << "\n";
            }

            oss << "\n]";
        } else {
            oss << "Tensor(shape=" << vec_to_string(shape)
                << ", data=" << flat_data_string()
                << ")";
        }

        return oss.str();
    }
    Tensor broadcast_to(const std::vector<size_t>& target_shape) const {
    std::vector<size_t> result_shape =
        broadcast_shapes(shape, target_shape);

    if (result_shape != target_shape) {
        throw std::invalid_argument(
            "broadcast_to target shape mismatch\n"
            "  ❌ Tensor cannot broadcast exactly to requested shape\n"
            "  💡 Target shape must be the final broadcast result\n"
            "  🔧 Fix: use a compatible target shape"
        );
    }

    std::vector<size_t> new_strides(target_shape.size(), 0);
        size_t self_rank = shape.size();
        size_t target_rank = target_shape.size();

        for (size_t i = 0; i < target_rank; i++) {
            size_t target_pos = target_rank - 1 - i;
            size_t target_dim = target_shape[target_pos];

            if (i < self_rank) {
                size_t self_pos = self_rank - 1 - i;
                size_t self_dim = shape[self_pos];

                if (self_dim == target_dim) {
                    new_strides[target_pos] = strides[self_pos];
                }
                else if (self_dim == 1) {
                    new_strides[target_pos] = 0;
                }
                else {
                    throw std::invalid_argument(
                        "broadcast_to stride construction failed"
                    );
                }
            }
            else {
                new_strides[target_pos] = 0;
            }
        }

        return Tensor(storage, target_shape, new_strides, offset);
    }
    Tensor masked_fill(const Tensor& mask, float value) const {
        auto out_shape = broadcast_shapes(shape, mask.shape);

        if (out_shape != shape) {
            throw std::invalid_argument(
                "masked_fill shape mismatch\n"
                "  ❌ Mask cannot broadcast to tensor shape\n"
                "  💡 mask must be broadcastable to this tensor's shape"
            );
        }

        Tensor M = mask.broadcast_to(shape);

        std::vector<float> data;
        data.reserve(numel());

        if (numel() == 0) {
            return Tensor(std::move(data), shape);
        }

        if (ndim() == 0) {
            data.push_back(M.at({}) != 0.0f ? value : at({}));
            return Tensor(std::move(data), shape);
        }

        std::vector<size_t> idx(shape.size(), 0);

        data.push_back(M.at(idx) != 0.0f ? value : at(idx));

        while (next_index(idx, shape)) {
            data.push_back(M.at(idx) != 0.0f ? value : at(idx));
        }

        return Tensor(std::move(data), shape);
    }
    Tensor sum(std::optional<int> axis = std::nullopt,
            bool keepdims = false) const {
        // Case 1: reduce all elements -> scalar
        if (!axis.has_value()) {
            double total = 0.0;

            if (numel() == 0) {
                return Tensor(0.0f);
            }

            if (ndim() == 0) {
                return Tensor(at({}));
            }

            std::vector<size_t> idx(shape.size(), 0);

            total += static_cast<double>(at(idx));

            while (next_index(idx, shape)) {
                total += static_cast<double>(at(idx));
            }

            return Tensor(static_cast<float>(total));
        }

        // Case 2: reduce along one axis
        int ax = normalize_axis(axis.value());

        std::vector<size_t> out_shape =
            reduction_shape(ax, keepdims);

        Tensor out = zeros(out_shape);

        if (numel() == 0) {
            return out;
        }

        std::vector<size_t> in_idx(shape.size(), 0);

        {
            auto out_idx =
                reduction_index(in_idx, ax, keepdims);

            out.at(out_idx) += at(in_idx);
        }

        while (next_index(in_idx, shape)) {
            auto out_idx =
                reduction_index(in_idx, ax, keepdims);

            out.at(out_idx) += at(in_idx);
        }

        return out;
    }
    Tensor mean(std::optional<int> axis = std::nullopt,
                bool keepdims = false) const
    {
        if (numel() == 0) {
            return Tensor(
                std::numeric_limits<float>::quiet_NaN()
            );
        }

        Tensor s = sum(axis, keepdims);

        float count;

        if (!axis.has_value()) {
            count = static_cast<float>(numel());
        }
        else {
            int ax = normalize_axis(axis.value());
            count = static_cast<float>(shape[ax]);
        }

        return s / count;
    }
    Tensor max(std::optional<int> axis = std::nullopt,
            bool keepdims = false) const
    {
        float neg_inf =
            -std::numeric_limits<float>::infinity();

        // Case 1: max over all elements
        if (!axis.has_value()) {
            if (numel() == 0) {
                return Tensor(neg_inf);
            }

            float best = neg_inf;

            if (ndim() == 0) {
                return Tensor(at({}));
            }

            std::vector<size_t> idx(shape.size(), 0);

            best = std::max(best, at(idx));

            while (next_index(idx, shape)) {
                best = std::max(best, at(idx));
            }

            return Tensor(best);
        }

        // Case 2: max along axis
        int ax = normalize_axis(axis.value());

        std::vector<size_t> out_shape =
            reduction_shape(ax, keepdims);

        Tensor out =
            full(out_shape, neg_inf);

        if (numel() == 0) {
            return out;
        }

        std::vector<size_t> in_idx(shape.size(), 0);

        auto out_idx =
            reduction_index(in_idx, ax, keepdims);

        out.at(out_idx) =
            std::max(out.at(out_idx), at(in_idx));

        while (next_index(in_idx, shape)) {
            out_idx =
                reduction_index(in_idx, ax, keepdims);

            out.at(out_idx) =
                std::max(out.at(out_idx), at(in_idx));
        }

        return out;
    }
    Tensor argmax(std::optional<int> axis = std::nullopt,
                bool keepdims = false) const
    {
        // Global argmax
        if (!axis.has_value()) {

            if (numel() == 0) {
                throw std::invalid_argument(
                    "argmax of empty tensor"
                );
            }

            float best_val = -std::numeric_limits<float>::infinity();
            float best_idx = 0.0f;

            size_t flat_idx = 0;

            if (ndim() == 0) {
                return Tensor(0.0f);
            }

            std::vector<size_t> idx(shape.size(), 0);

            do {
                float val = at(idx);

                if (val > best_val) {
                    best_val = val;
                    best_idx = static_cast<float>(flat_idx);
                }

                flat_idx++;

            } while (next_index(idx, shape));

            return Tensor(best_idx);
        }

        // Axis argmax
        int ax = normalize_axis(axis.value());

        auto out_shape =
            reduction_shape(ax, keepdims);

        Tensor out =
            full(out_shape, 0.0f);

        Tensor best_vals =
            full(
                out_shape,
                -std::numeric_limits<float>::infinity()
            );

        if (numel() == 0) {
            return out;
        }

        std::vector<size_t> in_idx(
            shape.size(),
            0
        );

        do {

            auto out_idx =
                reduction_index(
                    in_idx,
                    ax,
                    keepdims
                );

            float val =
                at(in_idx);

            if (val > best_vals.at(out_idx)) {

                best_vals.at(out_idx) = val;

                out.at(out_idx) =
                    static_cast<float>(
                        in_idx[ax]
                    );
            }

        } while (
            next_index(
                in_idx,
                shape
            )
        );

        return out;
    }

    void validate_matmul_2d(const Tensor& other) const {
        if (ndim() == 0 || other.ndim() == 0) {
            throw std::invalid_argument(
                "Matrix multiplication does not support scalars\n"
                "  ❌ Got scalar tensor\n"
                "  💡 Use * for element-wise scalar multiplication"
            );
        }

        if (ndim() != 2 || other.ndim() != 2) {
            throw std::invalid_argument(
                "Only 2D matmul is implemented for now\n"
                "  💡 Expected shapes {M,K} @ {K,N}"
            );
        }

        if (shape[1] != other.shape[0]) {
            throw std::invalid_argument(
                "Matrix multiplication shape mismatch\n"
                "  ❌ Got " + vec_to_string(shape) + " @ " + other.vec_to_string(other.shape) + "\n"
                "  💡 For A @ B, A.shape[1] must equal B.shape[0]"
            );
        }
    }

    Tensor matmul_naive(const Tensor& other) const {
        validate_matmul_2d(other);

        Tensor A = contiguous();
        Tensor B = other.contiguous();

        size_t M = A.shape[0];
        size_t K = A.shape[1];
        size_t N = B.shape[1];

        std::vector<float> out(M * N, 0.0f);

        const auto& a = *A.storage;
        const auto& b = *B.storage;

        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                double acc = 0.0;

                for (size_t k = 0; k < K; k++) {
                    acc += static_cast<double>(a[A.offset + i * K + k]) *
                        static_cast<double>(b[B.offset + k * N + j]);
                }

                out[i * N + j] = static_cast<float>(acc);
            }
        }

        return Tensor(std::move(out), {M, N});
    }

    Tensor matmul_transposed(const Tensor& other) const {
        validate_matmul_2d(other);

        Tensor A = contiguous();
        Tensor Bt = other.transpose().contiguous();

        size_t M = A.shape[0];
        size_t K = A.shape[1];
        size_t N = other.shape[1];

        std::vector<float> out(M * N, 0.0f);

        const auto& a = *A.storage;
        const auto& bt = *Bt.storage;

        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                double acc = 0.0;

                for (size_t k = 0; k < K; k++) {
                    acc += static_cast<double>(a[A.offset + i * K + k]) *
                        static_cast<double>(bt[Bt.offset + j * K + k]);
                }

                out[i * N + j] = static_cast<float>(acc);
            }
        }

        return Tensor(std::move(out), {M, N});
    }

    Tensor matmul_blocked(const Tensor& other, size_t BS = 64) const {
        validate_matmul_2d(other);

        Tensor A = contiguous();
        Tensor B = other.contiguous();

        size_t M = A.shape[0];
        size_t K = A.shape[1];
        size_t N = B.shape[1];

        std::vector<float> out(M * N, 0.0f);

        const auto& a = *A.storage;
        const auto& b = *B.storage;

        for (size_t ii = 0; ii < M; ii += BS) {
            for (size_t kk = 0; kk < K; kk += BS) {
                for (size_t jj = 0; jj < N; jj += BS) {

                    size_t i_end = std::min(ii + BS, M);
                    size_t k_end = std::min(kk + BS, K);
                    size_t j_end = std::min(jj + BS, N);

                    for (size_t i = ii; i < i_end; i++) {
                        for (size_t k = kk; k < k_end; k++) {
                            float aval = a[A.offset + i * K + k];

                            for (size_t j = jj; j < j_end; j++) {
                                out[i * N + j] += aval * b[B.offset + k * N + j];
                            }
                        }
                    }
                }
            }
        }

        return Tensor(std::move(out), {M, N});
    }

    Tensor matmul(const Tensor& other) const {
        return matmul_blocked(other);
    }

    Tensor fast_matmul_eigen(const Tensor& other) const {
        validate_matmul_2d(other);

        Tensor A = contiguous();
        Tensor B = other.contiguous();

        size_t M = A.shape[0];
        size_t K = A.shape[1];
        size_t N = B.shape[1];

        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            A_map(A.storage->data() + A.offset, M, K);

        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            B_map(B.storage->data() + B.offset, K, N);

        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> C =
            A_map * B_map;

        std::vector<float> out(C.data(), C.data() + C.size());

        return Tensor(std::move(out), {M, N});
    }
};

inline std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    os << t.str();
    return os;
}

inline Tensor zeros(const std::vector<size_t>& shape) {
    size_t actual = numel_from_shape(shape);
    std::vector<float> data(actual, 0.0f);
    return Tensor(std::move(data), shape);
}

inline Tensor ones(const std::vector<size_t>& shape) {
    size_t actual = numel_from_shape(shape);
    std::vector<float> data(actual, 1.0f);
    return Tensor(std::move(data), shape);
}

inline Tensor full(const std::vector<size_t>& shape, float value) {
    size_t actual = numel_from_shape(shape);
    std::vector<float> data(actual, value);
    return Tensor(std::move(data), shape);
}

inline Tensor arange(float start, float stop, float step = 1.0f) {
    std::vector<float> data;
    if (step == 0.0f) throw std::invalid_argument("step cannot be 0");
    if (step > 0) { for (float i = start; i < stop; i += step) data.push_back(i); }
    else          { for (float i = start; i > stop; i += step) data.push_back(i); }

    size_t n = data.size();                       // read BEFORE the move
    return Tensor(std::move(data), {n});
}

inline Tensor randn(const std::vector<size_t>& shape, unsigned seed = 42) {
    size_t actual = numel_from_shape(shape);

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> data(actual);

    for (size_t i = 0; i < actual; i++) {
        data[i] = dist(rng);
    }

    return Tensor(std::move(data), shape);
}

inline Tensor rand_uniform(const std::vector<size_t>& shape,
                           float lo,
                           float hi,
                           unsigned seed = 42) {
    size_t actual = numel_from_shape(shape);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);

    std::vector<float> data(actual);

    for (size_t i = 0; i < actual; i++) {
        data[i] = dist(rng);
    }

    return Tensor(std::move(data), shape);
}
inline std::vector<size_t> broadcast_shapes(
    const std::vector<size_t>& a,
    const std::vector<size_t>& b
) {
    size_t rank = std::max(a.size(), b.size());
    std::vector<size_t> result(rank);

    for (size_t i = 0; i < rank; i++) {
        size_t a_dim = 1;
        size_t b_dim = 1;

        if (i < a.size()) {
            a_dim = a[a.size() - 1 - i];
        }

        if (i < b.size()) {
            b_dim = b[b.size() - 1 - i];
        }

        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
            throw std::invalid_argument(
                "Broadcast shape mismatch\n"
                "  ❌ Cannot broadcast dimensions " +
                std::to_string(a_dim) + " and " +
                std::to_string(b_dim) + "\n"
                "  💡 Dimensions must be equal, or one dimension must be 1\n"
                "  🔧 Fix: reshape one tensor so dimensions align from the right"
            );
        }

        result[rank - 1 - i] = std::max(a_dim, b_dim);
    }

    return result;
}


template <class F>
inline Tensor binary_op(const Tensor& a, const Tensor& b, F f) {
    auto out_shape = broadcast_shapes(a.shape, b.shape);

    Tensor A = a.broadcast_to(out_shape);
    Tensor B = b.broadcast_to(out_shape);

    size_t total = numel_from_shape(out_shape);
    std::vector<float> out(total);

    // Fast path: same shape + contiguous memory
    if (A.shape == B.shape && A.is_contiguous() && B.is_contiguous()) {
        for (size_t i = 0; i < total; i++) {
            out[i] = f(
                (*A.storage)[A.offset + i],
                (*B.storage)[B.offset + i]
            );
        }

        return Tensor(std::move(out), out_shape);
    }

    // Scalar case: shape {}
    if (out_shape.empty()) {
        out[0] = f(A.at({}), B.at({}));
        return Tensor(std::move(out), out_shape);
    }

    // General path: works for broadcasted/non-contiguous tensors
    std::vector<size_t> idx(out_shape.size(), 0);

    size_t k = 0;
    out[k++] = f(A.at(idx), B.at(idx));

    while (A.next_index(idx, out_shape)) {
        out[k++] = f(A.at(idx), B.at(idx));
    }

    return Tensor(std::move(out), out_shape);
}
inline Tensor operator+(const Tensor& a, const Tensor& b) {
    return binary_op(a, b, [](float x, float y) { return x + y; });
}

inline Tensor operator-(const Tensor& a, const Tensor& b) {
    return binary_op(a, b, [](float x, float y) { return x - y; });
}

inline Tensor operator*(const Tensor& a, const Tensor& b) {
    return binary_op(a, b, [](float x, float y) { return x * y; });
}

// Float32 division intentionally follows NumPy behavior:
// division by zero gives inf/nan.
inline Tensor operator/(const Tensor& a, const Tensor& b) {
    return binary_op(a, b, [](float x, float y) { return x / y; });
}
inline Tensor operator+(const Tensor& a, float b) {
    return a + Tensor(b);
}

inline Tensor operator+(float a, const Tensor& b) {
    return Tensor(a) + b;
}

inline Tensor operator-(const Tensor& a, float b) {
    return a - Tensor(b);
}

inline Tensor operator-(float a, const Tensor& b) {
    return Tensor(a) - b;
}

inline Tensor operator*(const Tensor& a, float b) {
    return a * Tensor(b);
}

inline Tensor operator*(float a, const Tensor& b) {
    return Tensor(a) * b;
}

inline Tensor operator/(const Tensor& a, float b) {
    return a / Tensor(b);
}

inline Tensor operator/(float a, const Tensor& b) {
    return Tensor(a) / b;
}




template <class F>
inline Tensor unary_op(const Tensor& a, F f) {
    std::vector<float> out;
    out.reserve(a.numel());

    if (a.numel() == 0) {
        return Tensor(std::move(out), a.shape);
    }

    if (a.ndim() == 0) {
        out.push_back(f(a.at({})));
        return Tensor(std::move(out), a.shape);
    }

    std::vector<size_t> idx(a.shape.size(), 0);

    out.push_back(f(a.at(idx)));

    while (a.next_index(idx, a.shape)) {
        out.push_back(f(a.at(idx)));
    }

    return Tensor(std::move(out), a.shape);
}

inline Tensor exp(const Tensor& a) {
    return unary_op(a, [](float x) { return std::exp(x); });
}

inline Tensor log(const Tensor& a) {
    return unary_op(a, [](float x) { return std::log(x); });
}

inline Tensor sqrt(const Tensor& a) {
    return unary_op(a, [](float x) { return std::sqrt(x); });
}

inline Tensor abs(const Tensor& a) {
    return unary_op(a, [](float x) { return std::fabs(x); });
}

inline Tensor relu(const Tensor& a) {
    return unary_op(a, [](float x) { return x > 0.0f ? x : 0.0f; });
}

inline Tensor neg(const Tensor& a) {
    return unary_op(a, [](float x) { return -x; });
}

inline Tensor operator-(const Tensor& a) {
    return neg(a);
}
inline Tensor gt(const Tensor& a, const Tensor& b) {
    return binary_op(a, b, [](float x, float y) {
        return x > y ? 1.0f : 0.0f;
    });
}

inline Tensor lt(const Tensor& a, const Tensor& b) {
    return binary_op(a, b, [](float x, float y) {
        return x < y ? 1.0f : 0.0f;
    });
}

inline Tensor eq(const Tensor& a, const Tensor& b) {
    return binary_op(a, b, [](float x, float y) {
        return x == y ? 1.0f : 0.0f;
    });
}

inline Tensor ge(const Tensor& a, const Tensor& b) {
    return binary_op(a, b, [](float x, float y) {
        return x >= y ? 1.0f : 0.0f;
    });
}

inline Tensor le(const Tensor& a, const Tensor& b) {
    return binary_op(a, b, [](float x, float y) {
        return x <= y ? 1.0f : 0.0f;
    });
}
inline Tensor gt(const Tensor& a, float b) { return gt(a, Tensor(b)); }
inline Tensor gt(float a, const Tensor& b) { return gt(Tensor(a), b); }

inline Tensor lt(const Tensor& a, float b) { return lt(a, Tensor(b)); }
inline Tensor lt(float a, const Tensor& b) { return lt(Tensor(a), b); }

inline Tensor eq(const Tensor& a, float b) { return eq(a, Tensor(b)); }
inline Tensor eq(float a, const Tensor& b) { return eq(Tensor(a), b); }

inline Tensor ge(const Tensor& a, float b) { return ge(a, Tensor(b)); }
inline Tensor ge(float a, const Tensor& b) { return ge(Tensor(a), b); }

inline Tensor le(const Tensor& a, float b) { return le(a, Tensor(b)); }
inline Tensor le(float a, const Tensor& b) { return le(Tensor(a), b); }
} // namespace tinytorch