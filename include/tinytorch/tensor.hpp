#include <vector>
#include <cassert>
#include <memory>
#include <random>
#include <cstddef>
#include <sstream>
#include <string>
#include <iostream>
using namespace std;


namespace tinytorch{

    static size_t numel_from_shape(const std::vector<size_t>& shape) {
        size_t total = 1;
        for (size_t dim : shape) {
            total *= dim;
        }
        return total;
    }
    
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
        offset(offset_)
    {
        check_invariants();
    }
    Tensor shallow_copy() const {
        Tensor out;

        out.storage = storage;
        out.shape = shape;
        out.strides = strides;
        out.offset = offset;

        return out;
    }
    public:
        std::shared_ptr<std::vector<float>> storage;
        std::vector<size_t> shape;
        std::vector<size_t> strides;   // in ELEMENTS (NumPy uses bytes; we don't need to)
        size_t offset = 0;
        void check_invariants() const {
            assert(storage != nullptr);

            assert(shape.size() == strides.size());
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

            for (size_t i = 0; i < numel(); i++) {
                oss << (*storage)[offset + i];
                if (i + 1 < numel()) oss << ", ";
            }

            oss << "]";
            return oss.str();
        }
        std::vector<size_t> compute_strides(const std::vector<size_t>& shape) const {
            std::vector<size_t> strides(shape.size());
            size_t running_product = 1;

            for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
                strides[i] = running_product;
                running_product *= shape[i];
            }

            return strides;
        }

        Tensor(){
            storage = std::make_shared<std::vector<float>>();
            shape = {0};
            strides = {1};
            offset = 0;
        } 


        Tensor(std::vector<float> data, std::vector<size_t> shape_)
            : storage(nullptr),
            shape(std::move(shape_)),
            strides{},
            offset(0)
        {
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
        Tensor(float scalar)
            : storage(std::make_shared<std::vector<float>>(
                std::vector<float>{scalar}
            )),
            shape({}),      // 0-D tensor
            strides({}),    // no dimensions, so no strides
            offset(0)
        {
            check_invariants();
        }
        size_t ndim() const {
            return shape.size();
        }

        size_t numel() const {
            return numel_from_shape(shape);
        }
        size_t nbytes() const {
            return numel() * sizeof(float);
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
                oss << (*storage)[offset];
            }
            else if (ndim() == 1) {
                oss << "[";
                for (size_t i = 0; i < shape[0]; i++) {
                    oss << (*storage)[offset + i * strides[0]];
                    if (i + 1 < shape[0]) oss << ", ";
                }
                oss << "]";
            }
            else if (ndim() == 2) {
                oss << "[\n";

                for (size_t i = 0; i < shape[0]; i++) {
                    oss << "  [";

                    for (size_t j = 0; j < shape[1]; j++) {
                        size_t idx = offset + i * strides[0] + j * strides[1];
                        oss << (*storage)[idx];

                        if (j + 1 < shape[1]) oss << ", ";
                    }

                    oss << "]";
                    if (i + 1 < shape[0]) oss << "\n";
                }

                oss << "\n]";
            }
            else {
                oss << "Tensor(shape=" << vec_to_string(shape)
                    << ", data=" << flat_data_string()
                    << ")";
            }

            return oss.str();
        }
        float& at(const std::vector<size_t>& idx) {
            return (*storage)[flat_index(idx)];
        }

        const float& at(const std::vector<size_t>& idx) const {
            return (*storage)[flat_index(idx)];
        }
        // returns false when iteration is finished
        bool next_index(std::vector<size_t>& idx,
                        const std::vector<size_t>& shape) const
        {
            // Any zero dimension means tensor has no elements.
            for (size_t dim : shape) {
                if (dim == 0) return false;
            }

            // Rank-0 scalar: only one logical index {}, no next index.
            if (shape.empty()) {
                return false;
            }

            for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
                idx[d]++;

                if (idx[d] < shape[d]) {
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
    // Python behavior: transpose() on 0D/1D returns a view.
            if (ndim() < 2 && d0 == -2 && d1 == -1) {
                return shallow_copy();
            }

            int n = static_cast<int>(ndim());

            // Normalize negative dimensions.
            if (d0 < 0) d0 += n;
            if (d1 < 0) d1 += n;

            // Bounds check.
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

            return strides == compute_strides(shape);
        }
    };
    std::ostream& operator<<(std::ostream& os, const Tensor& t) {
        os << t.str();
        return os;
    }
    Tensor zeros(const vector<size_t>& shape){
        size_t actual = numel_from_shape(shape);
        vector<float> data(actual, 0.0f);
        return Tensor(data, shape);
    }
    Tensor ones(const vector<size_t>& shape){
        size_t actual = numel_from_shape(shape);
        vector<float> data(actual, 1.0f);
        return Tensor(data, shape);
    }
    Tensor full(const vector<size_t>& shape, const float value ){
        size_t actual = numel_from_shape(shape);
        vector<float> data(actual, value);
        return Tensor(data, shape);
    }
    Tensor arange(float start, float stop , float step){
        std::vector<float> data;

        if (step == 0)
            throw std::invalid_argument("step cannot be 0");

        if (step > 0) {
            for (float i = start; i < stop; i += step) {
                data.push_back(i);
            }
        }
        else {
            for (float i = start; i > stop; i += step) {
                data.push_back(i);
            }
        }
        return Tensor(data, {data.size()});
    }
    Tensor randn(const std::vector<size_t>& shape, unsigned seed = 42){
        size_t actual = numel_from_shape(shape);
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0, 1);
        vector<float> data(actual);
        for(int i = 0;i<actual;i++){
            data[i] = dist(rng);
        }
        return Tensor(data, shape);
    }
    Tensor rand_uniform(const std::vector<size_t>& shape,
                    float lo,
                    float hi,
                    unsigned seed = 42)
    {
        size_t actual = numel_from_shape(shape);

        std::mt19937 rng(seed);

        std::uniform_real_distribution<float> dist(lo, hi);

        std::vector<float> data(actual);

        for (size_t i = 0; i < actual; i++)
        {
            data[i] = dist(rng);
        }

        return Tensor(std::move(data), shape);
    }



}