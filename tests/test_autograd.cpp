#include <cassert>
#include <iostream>

#include "tinytorch/tensor.hpp"

using namespace tinytorch;

struct DummyBackward : Function {
    std::vector<Tensor> backward(
        const Tensor& grad_output
    ) override {
        return {grad_output};
    }
};

void test_requires_grad_flag() {
    Tensor x({1, 2, 3}, {3});

    assert(!x.requires_grad());

    x.set_requires_grad(true);

    assert(x.requires_grad());
}

void test_autograd_meta_shared_on_copy() {
    Tensor x({1, 2, 3}, {3});

    x.set_requires_grad(true);

    Tensor y = x;

    assert(y.requires_grad());
}

void test_zero_grad_no_crash() {
    Tensor x({1, 2, 3}, {3});

    x.set_requires_grad(true);

    x.zero_grad();
}

void test_grad_before_backward_throws() {
    Tensor x({1, 2, 3}, {3});

    x.set_requires_grad(true);

    bool thrown = false;

    try {
        x.grad();
    }
    catch (const std::runtime_error&) {
        thrown = true;
    }

    assert(thrown);
}

void test_make_tracked() {
    Tensor x({1, 2, 3}, {3});

    x.set_requires_grad(true);

    Tensor raw({4, 5, 6}, {3});

    auto fn = std::make_shared<DummyBackward>();

    Tensor y =
        make_tracked(raw, fn, {x});

    assert(y.requires_grad());
}

void test_reduce_grad_to_shape_leading_axis() {
    Tensor g(
        {1, 1, 1,
         1, 1, 1},
        {2, 3}
    );

    Tensor r =
        reduce_grad_to_shape(
            g,
            {3}
        );

    assert(
        r.shape ==
        std::vector<size_t>({3})
    );

    assert(
        r.to_vector() ==
        std::vector<float>({2, 2, 2})
    );
}

void test_reduce_grad_to_shape_keepdim_axis() {
    Tensor g(
        {1, 2, 3,
         4, 5, 6},
        {2, 3}
    );

    Tensor r =
        reduce_grad_to_shape(
            g,
            {2, 1}
        );

    assert(
        r.shape ==
        std::vector<size_t>({2, 1})
    );

    assert(
        r.to_vector() ==
        std::vector<float>({6, 15})
    );
}

void test_reduce_grad_to_shape_same_shape() {
    Tensor g(
        {1, 2, 3,
         4, 5, 6},
        {2, 3}
    );

    Tensor r =
        reduce_grad_to_shape(
            g,
            {2, 3}
        );

    assert(
        r.shape ==
        std::vector<size_t>({2, 3})
    );

    assert(
        r.to_vector() ==
        std::vector<float>({
            1, 2, 3,
            4, 5, 6
        })
    );
}

void test_backward_non_scalar_requires_grad_output() {
    Tensor x(
        {1, 2, 3},
        {3}
    );

    x.set_requires_grad(true);

    bool thrown = false;

    try {
        x.backward();
    }
    catch (const std::runtime_error&) {
        thrown = true;
    }

    assert(thrown);
}

void test_add_backward_simple() {
    Tensor x(
        {2.0f},
        {1}
    );

    x.set_requires_grad(true);

    Tensor y =
        x + 3.0f;

    y.backward();

    assert(
        x.grad().item() == 1.0f
    );
}

void test_mul_backward_simple() {
    Tensor x(
        {2.0f},
        {1}
    );

    x.set_requires_grad(true);

    Tensor y =
        x * 3.0f;

    y.backward();

    assert(
        x.grad().item() == 3.0f
    );
}

void test_chain_mul_add_backward() {
    Tensor x(
        {2.0f},
        {1}
    );

    x.set_requires_grad(true);

    Tensor y =
        x * 3.0f + 1.0f;

    y.backward();

    assert(
        x.grad().item() == 3.0f
    );
}
void test_div_backward_simple() {
    Tensor x({6.0f}, {1});
    x.set_requires_grad(true);

    Tensor y = x / 2.0f;

    y.backward();

    assert(x.grad().item() == 0.5f);
}
void test_div_backward_denominator() {
    Tensor x({2.0f}, {1});
    x.set_requires_grad(true);

    Tensor y({4.0f}, {1});
    y.set_requires_grad(true);

    Tensor z = x / y;

    z.backward();

    assert(std::abs(x.grad().item() - 0.25f) < 1e-5);
    assert(std::abs(y.grad().item() + 0.125f) < 1e-5);
}
void test_matmul_backward_simple() {
    Tensor x({2.0f}, {1, 1});
    x.set_requires_grad(true);

    Tensor w({3.0f}, {1, 1});
    w.set_requires_grad(true);

    Tensor y = x.matmul(w);

    y.backward();

    assert(std::abs(x.grad().item() - 3.0f) < 1e-5);
    assert(std::abs(w.grad().item() - 2.0f) < 1e-5);
}
void test_sum_backward_simple() {
    Tensor x({1.0f, 2.0f, 3.0f}, {3});
    x.set_requires_grad(true);

    Tensor y = x.sum();

    y.backward();

    assert(std::abs(x.grad().at({0}) - 1.0f) < 1e-5);
    assert(std::abs(x.grad().at({1}) - 1.0f) < 1e-5);
    assert(std::abs(x.grad().at({2}) - 1.0f) < 1e-5);
}
void test_mean_backward_simple() {
    Tensor x({2.0f, 4.0f, 6.0f}, {3});
    x.set_requires_grad(true);

    Tensor y = x.mean();

    y.backward();

    assert(std::abs(x.grad().at({0}) - 1.0f / 3.0f) < 1e-5);
    assert(std::abs(x.grad().at({1}) - 1.0f / 3.0f) < 1e-5);
    assert(std::abs(x.grad().at({2}) - 1.0f / 3.0f) < 1e-5);
}
void test_relu_backward() {
    Tensor x({-2.0f, 0.0f, 3.0f}, {3});
    x.set_requires_grad(true);

    Tensor y =
        relu(x).sum();

    y.backward();

    assert(x.grad().at({0}) == 0.0f);
    assert(x.grad().at({1}) == 0.0f);
    assert(x.grad().at({2}) == 1.0f);
}
void test_neg_backward() {
    Tensor x({2.0f, -3.0f}, {2});
    x.set_requires_grad(true);

    Tensor y = (-x).sum();

    y.backward();

    assert(x.grad().at({0}) == -1.0f);
    assert(x.grad().at({1}) == -1.0f);
}
void test_exp_backward() {
    Tensor x({0.0f}, {1});
    x.set_requires_grad(true);

    Tensor y = exp(x).sum();

    y.backward();

    assert(std::abs(x.grad().at({0}) - 1.0f) < 1e-5);
}
void test_log_backward() {
    Tensor x({2.0f}, {1});
    x.set_requires_grad(true);

    Tensor y = log(x).sum();

    y.backward();

    assert(
        std::abs(
            x.grad().at({0}) - 0.5f
        ) < 1e-5
    );
}
void test_sqrt_backward() {
    Tensor x({4.0f}, {1});
    x.set_requires_grad(true);

    Tensor y = sqrt(x).sum();

    y.backward();

    assert(
        std::abs(
            x.grad().at({0}) - 0.25f
        ) < 1e-5
    );
}
void test_sigmoid_backward() {
    Tensor x({0.0f}, {1});
    x.set_requires_grad(true);

    Tensor y = sigmoid(x).sum();

    y.backward();

    assert(
        std::abs(
            x.grad().at({0}) - 0.25f
        ) < 1e-5
    );
}
void test_tanh_backward() {
    Tensor x({0.0f}, {1});
    x.set_requires_grad(true);

    Tensor y = tanh(x).sum();

    y.backward();

    assert(
        std::abs(
            x.grad().at({0}) - 1.0f
        ) < 1e-5
    );
}
void test_pow_backward() {
    Tensor x({3.0f}, {1});
    x.set_requires_grad(true);

    Tensor y =
        pow(x, 2.0f).sum();

    y.backward();

    assert(
        std::abs(
            x.grad().at({0}) - 6.0f
        ) < 1e-5
    );
}

void test_reshape_backward() {
    Tensor x({1,2,3,4,5,6}, {6});
    x.set_requires_grad(true);

    Tensor y = x.reshape({2,3});
    Tensor loss = y.sum();

    loss.backward();

    for (size_t i = 0; i < 6; ++i) {
        assert(
            std::abs(
                x.grad().at({i}) - 1.0f
            ) < 1e-5
        );
    }
}
void test_transpose_backward() {

    Tensor x(
        {1,2,3,4},
        {2,2}
    );

    x.set_requires_grad(true);

    Tensor y =
        x.transpose();

    Tensor loss =
        y.sum();

    loss.backward();

    assert(
        x.grad().at({0,0}) == 1.0f
    );

    assert(
        x.grad().at({0,1}) == 1.0f
    );

    assert(
        x.grad().at({1,0}) == 1.0f
    );

    assert(
        x.grad().at({1,1}) == 1.0f
    );
}
int main() {
    test_requires_grad_flag();
    test_autograd_meta_shared_on_copy();
    test_zero_grad_no_crash();
    test_grad_before_backward_throws();

    test_make_tracked();

    test_reduce_grad_to_shape_leading_axis();
    test_reduce_grad_to_shape_keepdim_axis();
    test_reduce_grad_to_shape_same_shape();

    test_backward_non_scalar_requires_grad_output();

    test_add_backward_simple();
    test_mul_backward_simple();
    test_chain_mul_add_backward();

    test_div_backward_simple();
    test_div_backward_denominator();

    test_matmul_backward_simple();
    test_sum_backward_simple();
    test_mean_backward_simple();
    test_relu_backward();
    test_neg_backward();
    test_exp_backward();
    test_log_backward();
    test_sqrt_backward();
    test_sigmoid_backward();
    test_tanh_backward();
    test_pow_backward();
    test_reshape_backward();
    test_transpose_backward();
    std::cout
        << "✅ All autograd tests passed\n";

    return 0;
}