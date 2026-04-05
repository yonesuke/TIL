// Tridiagonal solver: C++17, pybind11
//
// Public API (float32 and float64 overloads):
//
//   solve_thomas(lo, di, up, b)             – pure Thomas (sequential loop)
//   solve_bwd_scan(lo, di, up, b)           – loop fwd + affine scan bwd
//   solve_const_thomas(l, d, u, b)          – scalar coefficients, Thomas
//   solve_const_bwd_scan(l, d, u, b)        – scalar coefficients, bwd scan
//
// Backward affine scan
// ─────────────────────
//   After forward elimination: x[i] = dp[i] − c[i]·x[i+1]
//   Rewrite as affine map: x[i] = a_i + b_i·x[i+1]  (a_i=dp[i], b_i=−c[i])
//   Composition: (a,b)∘(a',b') = (a + b·a',  b·b')
//   → Build reversed state vector, run inclusive_scan, reverse back.
//     std::inclusive_scan gets unrolled / vectorised by Clang/GCC with -O3.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

namespace py = pybind11;

template<typename T>
using arr_t = py::array_t<T, py::array::c_style | py::array::forcecast>;

template<typename T>
using P = std::pair<T, T>;

template<typename T>
inline P<T> p_compose(P<T> acc, P<T> cur) noexcept {
    return { cur.first + cur.second * acc.first,  cur.second * acc.second };
}

// ── pure Thomas ───────────────────────────────────────────────────────────────
template<typename T>
py::array_t<T> solve_thomas(arr_t<T> lower, arr_t<T> diag,
                             arr_t<T> upper, arr_t<T> rhs) {
    auto lo = lower.template unchecked<1>();
    auto di = diag .template unchecked<1>();
    auto up = upper.template unchecked<1>();
    auto b  = rhs  .template unchecked<1>();
    const int n = static_cast<int>(di.shape(0));

    std::vector<T> c(n - 1), dp(n);
    c[0]  = up(0) / di(0);
    dp[0] = b(0)  / di(0);
    for (int i = 1; i < n; ++i) {
        const T denom = di(i) - lo(i-1) * c[i-1];
        if (i < n-1) c[i] = up(i) / denom;
        dp[i] = (b(i) - lo(i-1) * dp[i-1]) / denom;
    }

    auto result = py::array_t<T>(n);
    auto x = result.template mutable_unchecked<1>();
    x(n-1) = dp[n-1];
    for (int i = n-2; i >= 0; --i) x(i) = dp[i] - c[i] * x(i+1);
    return result;
}

// ── loop fwd + backward affine scan ──────────────────────────────────────────
template<typename T>
py::array_t<T> solve_bwd_scan(arr_t<T> lower, arr_t<T> diag,
                               arr_t<T> upper, arr_t<T> rhs) {
    auto lo = lower.template unchecked<1>();
    auto di = diag .template unchecked<1>();
    auto up = upper.template unchecked<1>();
    auto b  = rhs  .template unchecked<1>();
    const int n = static_cast<int>(di.shape(0));

    std::vector<T> c(n - 1), dp(n);
    c[0]  = up(0) / di(0);
    dp[0] = b(0)  / di(0);
    for (int i = 1; i < n; ++i) {
        const T denom = di(i) - lo(i-1) * c[i-1];
        if (i < n-1) c[i] = up(i) / denom;
        dp[i] = (b(i) - lo(i-1) * dp[i-1]) / denom;
    }

    std::vector<P<T>> states(n);
    states[n-1] = { dp[n-1], T(0) };
    for (int i = 0; i < n-1; ++i) states[i] = { dp[i], -c[i] };

    std::reverse(states.begin(), states.end());
    std::inclusive_scan(states.begin(), states.end(), states.begin(),
                        [](P<T> r, P<T> l) { return p_compose<T>(r, l); });
    std::reverse(states.begin(), states.end());

    auto result = py::array_t<T>(n);
    auto x = result.template mutable_unchecked<1>();
    for (int i = 0; i < n; ++i) x(i) = states[i].first;
    return result;
}

// ── const-coeff Thomas ────────────────────────────────────────────────────────
template<typename T>
py::array_t<T> solve_const_thomas(T lo, T di, T up, arr_t<T> rhs) {
    auto b  = rhs.template unchecked<1>();
    const int n = static_cast<int>(b.shape(0));

    std::vector<T> c(n - 1), dp(n);
    c[0]  = up / di;
    dp[0] = b(0) / di;
    for (int i = 1; i < n; ++i) {
        const T denom = di - lo * c[i-1];
        if (i < n-1) c[i] = up / denom;
        dp[i] = (b(i) - lo * dp[i-1]) / denom;
    }

    auto result = py::array_t<T>(n);
    auto x = result.template mutable_unchecked<1>();
    x(n-1) = dp[n-1];
    for (int i = n-2; i >= 0; --i) x(i) = dp[i] - c[i] * x(i+1);
    return result;
}

// ── const-coeff bwd scan ──────────────────────────────────────────────────────
template<typename T>
py::array_t<T> solve_const_bwd_scan(T lo, T di, T up, arr_t<T> rhs) {
    auto b  = rhs.template unchecked<1>();
    const int n = static_cast<int>(b.shape(0));

    std::vector<T> c(n - 1), dp(n);
    c[0]  = up / di;
    dp[0] = b(0) / di;
    for (int i = 1; i < n; ++i) {
        const T denom = di - lo * c[i-1];
        if (i < n-1) c[i] = up / denom;
        dp[i] = (b(i) - lo * dp[i-1]) / denom;
    }

    std::vector<P<T>> states(n);
    states[n-1] = { dp[n-1], T(0) };
    for (int i = 0; i < n-1; ++i) states[i] = { dp[i], -c[i] };

    std::reverse(states.begin(), states.end());
    std::inclusive_scan(states.begin(), states.end(), states.begin(),
                        [](P<T> r, P<T> l) { return p_compose<T>(r, l); });
    std::reverse(states.begin(), states.end());

    auto result = py::array_t<T>(n);
    auto x = result.template mutable_unchecked<1>();
    for (int i = 0; i < n; ++i) x(i) = states[i].first;
    return result;
}

// ── bindings ──────────────────────────────────────────────────────────────────
PYBIND11_MODULE(tridiagonal_cpp, m) {
    m.doc() = "Tridiagonal solvers: Thomas variants, float32 and float64 overloads";

    m.def("solve_thomas",        &solve_thomas<double>);
    m.def("solve_thomas",        &solve_thomas<float>);
    m.def("solve_bwd_scan",      &solve_bwd_scan<double>);
    m.def("solve_bwd_scan",      &solve_bwd_scan<float>);
    m.def("solve_const_thomas",  &solve_const_thomas<double>);
    m.def("solve_const_thomas",  &solve_const_thomas<float>);
    m.def("solve_const_bwd_scan",&solve_const_bwd_scan<double>);
    m.def("solve_const_bwd_scan",&solve_const_bwd_scan<float>);
}
