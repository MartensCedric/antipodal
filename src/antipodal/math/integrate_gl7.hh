#pragma once

#include <antipodal/math/integrate.hh>

#include <cmath>
#include <concepts>
#include <type_traits>

namespace antipodal
{
namespace detail::gl7
{
// Gauss-Legendre 7-point nodes on [-1, 1] (positive side only).
inline constexpr double x1 = 0.4058451513773971669;
inline constexpr double x2 = 0.7415311855993944399;
inline constexpr double x3 = 0.9491079123427585245;

// 7-point Gauss weights (center + |x1|, |x2|, |x3| pairs).
inline constexpr double w0 = 0.4179591836734693878;
inline constexpr double w1 = 0.3818300505051189449;
inline constexpr double w2 = 0.2797053914892766679;
inline constexpr double w3 = 0.1294849661688696933;
} // namespace detail::gl7

/// Adaptive Gauss-Legendre 7-point integration. The error estimate is
/// `|I2 - I1|` where `I1` is the rule applied to `[a, b]` and
/// `I2 = eval(left half) + eval(right half)`.
///
/// `fun` is invoked as `fun(idx, t)`. The index is hierarchical and stable: a
/// segment with hierarchy index `h` evaluates 7 nodes with indices
/// `7*h + k` for `k` in `[0, 7)`; children are `2*h + 1` and `2*h + 2`. With
/// this scheme a parent's `I2` evaluations land on the children's own `I1`
/// indices, so a cache populated in `preinvoke` mode is hit by `evaluate`
/// without redundancy.
///
/// Behaviour by `Mode`:
///   * `evaluate`     — full adaptive integration; returns the integral.
///                      Optionally writes accumulated `|I2 - I1|` of accepted
///                      leaves into `*out_err` and the segment count into
///                      `*out_intervals`.
///   * `preinvoke`    — returns void. Walks the tree calling `fun` at every
///                      node `evaluate` could conceivably query. `cfg.tol` and
///                      `cfg.min_depth` are ignored.
///   * `single_level` — returns the GL7 estimate from one 7-node evaluation
///                      on `[a, b]`. No recursion, no error estimate.
template <integrate_mode Mode = integrate_mode::evaluate, class T, class F>
    requires std::invocable<F&, int, T>
auto integrate_gl7(F&& fun,
                   T a,
                   T b,
                   integrate_config const& cfg,
                   std::type_identity_t<T>* out_err = nullptr,
                   int* out_intervals = nullptr)
    -> std::conditional_t<Mode == integrate_mode::preinvoke, void, T>
{
    using std::abs;
    namespace gl = detail::gl7;

    // 7-node rule on a segment, calling `fun` at indices 7*idx + k. Returns the
    // segment integral. Used by all three modes.
    auto eval_segment = [&](int idx, T aa, T bb) -> T
    {
        int const ib = idx * 7;
        T const c = (aa + bb) * T(0.5);
        T const h = (bb - aa) * T(0.5);

        T const f0 = fun(ib + 0, c);

        T const dx1 = h * T(gl::x1);
        T const f1m = fun(ib + 1, c - dx1);
        T const f1p = fun(ib + 2, c + dx1);

        T const dx2 = h * T(gl::x2);
        T const f2m = fun(ib + 3, c - dx2);
        T const f2p = fun(ib + 4, c + dx2);

        T const dx3 = h * T(gl::x3);
        T const f3m = fun(ib + 5, c - dx3);
        T const f3p = fun(ib + 6, c + dx3);

        T I = T(gl::w0) * f0;
        I = I + T(gl::w1) * (f1m + f1p);
        I = I + T(gl::w2) * (f2m + f2p);
        I = I + T(gl::w3) * (f3m + f3p);
        return I * h;
    };

    if constexpr (Mode == integrate_mode::single_level)
    {
        (void)cfg;
        (void)out_err;
        (void)out_intervals;

        if (a == b)
            return T(0);

        return eval_segment(0, a, b);
    }
    else
    {
        if constexpr (Mode == integrate_mode::evaluate)
        {
            if (out_err)
                *out_err = T(0);
            if (out_intervals)
                *out_intervals = 0;
        }

        if (a == b)
        {
            if constexpr (Mode == integrate_mode::evaluate)
                return T(0);
            else
                return;
        }

        T total_err = T(0);
        int total_intervals = 0;

        auto recurse = [&](auto&& self, int idx, T aa, T bb, T local_tol, int depth) -> auto
        {
            T const mid = (aa + bb) * T(0.5);

            if constexpr (Mode == integrate_mode::evaluate)
            {
                T const I1 = eval_segment(idx, aa, bb);
                T const I2 = eval_segment(idx * 2 + 1, aa, mid)  //
                             + eval_segment(idx * 2 + 2, mid, bb);

                T const err = abs(I2 - I1);
                bool const at_max = depth >= cfg.max_depth;
                bool const can_accept = depth >= cfg.min_depth && err <= local_tol;

                if (at_max || can_accept)
                {
                    total_err = total_err + err;
                    total_intervals += 1;
                    return I2;
                }

                T const half_tol = local_tol * T(0.5);
                T const left = self(self, idx * 2 + 1, aa, mid, half_tol, depth + 1);
                T const right = self(self, idx * 2 + 2, mid, bb, half_tol, depth + 1);
                return left + right;
            }
            else // preinvoke — mirror evaluate's call pattern but skip the math
            {
                (void)local_tol;
                eval_segment(idx, aa, bb);
                eval_segment(idx * 2 + 1, aa, mid);
                eval_segment(idx * 2 + 2, mid, bb);

                if (depth >= cfg.max_depth)
                    return;

                self(self, idx * 2 + 1, aa, mid, local_tol, depth + 1);
                self(self, idx * 2 + 2, mid, bb, local_tol, depth + 1);
            }
        };

        if constexpr (Mode == integrate_mode::evaluate)
        {
            T const result = recurse(recurse, 0, a, b, T(cfg.tol), 0);
            if (out_err)
                *out_err = total_err;
            if (out_intervals)
                *out_intervals = total_intervals;
            return result;
        }
        else
        {
            recurse(recurse, 0, a, b, T(cfg.tol), 0);
        }
    }
}

/// Callable wrapper around `integrate_gl7` matching the duck-typed
/// `Integrator` concept used by the parametric GWN kernels:
/// `op()(F&& fun, T a, T b, integrate_config const&) -> T`, with `fun(T t)`.
/// The hierarchical index is hidden — pass `integrate_gl7` directly when
/// callers need it for caching.
struct IntegratorGL7
{
    template <class F, class T>
    [[nodiscard]] T operator()(F&& fun, T a, T b, integrate_config const& cfg) const
    {
        return integrate_gl7([&](int /*idx*/, T t) { return fun(t); }, a, b, cfg);
    }
};
} // namespace antipodal
