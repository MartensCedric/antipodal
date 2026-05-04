#pragma once

#include <antipodal/math/integrate.hh>

#include <cmath>
#include <concepts>
#include <type_traits>

namespace antipodal
{
namespace detail::gk15
{
// G7-K15 nodes on [-1, 1] (positive Kronrod abscissae).
inline constexpr double x1 = 0.9914553711208126;
inline constexpr double x2 = 0.9491079123427585;
inline constexpr double x3 = 0.8648644233597691;
inline constexpr double x4 = 0.7415311855993945;
inline constexpr double x5 = 0.5860872354676911;
inline constexpr double x6 = 0.4058451513773972;
inline constexpr double x7 = 0.2077849550078985;

// 15-point Kronrod weights (center, then |x1|..|x7| pairs).
inline constexpr double wK0 = 0.2094821410847278;
inline constexpr double wK1 = 0.0229353220105292;
inline constexpr double wK2 = 0.0630920926299786;
inline constexpr double wK3 = 0.1047900103222502;
inline constexpr double wK4 = 0.1406532597155259;
inline constexpr double wK5 = 0.1690047266392679;
inline constexpr double wK6 = 0.1903505780647854;
inline constexpr double wK7 = 0.2044329400752989;

// 7-point Gauss weights (center + pairs at |x2|, |x4|, |x6|).
inline constexpr double wG_center = 0.4179591836734694;
inline constexpr double wG_x2 = 0.1294849661688697;
inline constexpr double wG_x4 = 0.2797053914892767;
inline constexpr double wG_x6 = 0.3818300505051189;
} // namespace detail::gk15

/// Adaptive integration of `fun` over `[a, b]` using the 15-point Gauss-Kronrod
/// rule (with the embedded 7-point Gauss subset providing the error estimate).
///
/// `fun` is invoked as `fun(idx, t)`. The index is hierarchical and stable:
///   * The 15 nodes of the segment with hierarchy index `h` use indices
///     `15*h + k` for `k` in `[0, 15)`. Hierarchy starts at `h = 0` (root); a
///     segment at `h` has children `2*h + 1` (left) and `2*h + 2` (right).
///   * For the same `(a, b, max_depth)`, every node has a stable index, so the
///     index can be used as a cache key.
///
/// Behaviour by `Mode`:
///   * `evaluate`     — full adaptive integration; returns the integral.
///                      Optionally writes the accumulated `|I_K - I_G|` of the
///                      accepted leaves into `*out_err` and the number of
///                      visited segments into `*out_intervals`.
///   * `preinvoke`    — returns void. Walks the recursion tree to
///                      `cfg.max_depth` regardless of tolerance and just calls
///                      `fun` at every node. Useful for pre-populating an
///                      external cache. `cfg.min_depth` and `cfg.tol` are
///                      ignored; `out_err` / `out_intervals` are unused.
///   * `single_level` — returns the Kronrod estimate from one 15-node
///                      evaluation on `[a, b]`. No recursion, no error
///                      estimate. Everything in `cfg` apart from the (unused)
///                      type is ignored.
template <integrate_mode Mode = integrate_mode::evaluate, class T, class F>
    requires std::invocable<F&, int, T>
auto integrate_gk15(F&& fun,
                    T a,
                    T b,
                    integrate_config const& cfg,
                    std::type_identity_t<T>* out_err = nullptr,
                    int* out_intervals = nullptr)
    -> std::conditional_t<Mode == integrate_mode::preinvoke, void, T>
{
    using std::abs;
    namespace gk = detail::gk15;

    if constexpr (Mode == integrate_mode::single_level)
    {
        (void)cfg;
        (void)out_err;
        (void)out_intervals;

        if (a == b)
            return T(0);

        T const c = (a + b) * T(0.5);
        T const h = (b - a) * T(0.5);

        T const f0 = fun(0, c);

        T const dx1 = h * T(gk::x1);
        T const f1m = fun(1, c - dx1);
        T const f1p = fun(2, c + dx1);

        T const dx2 = h * T(gk::x2);
        T const f2m = fun(3, c - dx2);
        T const f2p = fun(4, c + dx2);

        T const dx3 = h * T(gk::x3);
        T const f3m = fun(5, c - dx3);
        T const f3p = fun(6, c + dx3);

        T const dx4 = h * T(gk::x4);
        T const f4m = fun(7, c - dx4);
        T const f4p = fun(8, c + dx4);

        T const dx5 = h * T(gk::x5);
        T const f5m = fun(9, c - dx5);
        T const f5p = fun(10, c + dx5);

        T const dx6 = h * T(gk::x6);
        T const f6m = fun(11, c - dx6);
        T const f6p = fun(12, c + dx6);

        T const dx7 = h * T(gk::x7);
        T const f7m = fun(13, c - dx7);
        T const f7p = fun(14, c + dx7);

        T Ik = T(gk::wK0) * f0;
        Ik = Ik + T(gk::wK1) * (f1m + f1p);
        Ik = Ik + T(gk::wK2) * (f2m + f2p);
        Ik = Ik + T(gk::wK3) * (f3m + f3p);
        Ik = Ik + T(gk::wK4) * (f4m + f4p);
        Ik = Ik + T(gk::wK5) * (f5m + f5p);
        Ik = Ik + T(gk::wK6) * (f6m + f6p);
        Ik = Ik + T(gk::wK7) * (f7m + f7p);

        return Ik * h;
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
            T const c = (aa + bb) * T(0.5);
            T const h = (bb - aa) * T(0.5);
            int const ib = idx * 15;

            T const f0 = fun(ib + 0, c);

            T const dx1 = h * T(gk::x1);
            T const f1m = fun(ib + 1, c - dx1);
            T const f1p = fun(ib + 2, c + dx1);

            T const dx2 = h * T(gk::x2);
            T const f2m = fun(ib + 3, c - dx2);
            T const f2p = fun(ib + 4, c + dx2);

            T const dx3 = h * T(gk::x3);
            T const f3m = fun(ib + 5, c - dx3);
            T const f3p = fun(ib + 6, c + dx3);

            T const dx4 = h * T(gk::x4);
            T const f4m = fun(ib + 7, c - dx4);
            T const f4p = fun(ib + 8, c + dx4);

            T const dx5 = h * T(gk::x5);
            T const f5m = fun(ib + 9, c - dx5);
            T const f5p = fun(ib + 10, c + dx5);

            T const dx6 = h * T(gk::x6);
            T const f6m = fun(ib + 11, c - dx6);
            T const f6p = fun(ib + 12, c + dx6);

            T const dx7 = h * T(gk::x7);
            T const f7m = fun(ib + 13, c - dx7);
            T const f7p = fun(ib + 14, c + dx7);

            if constexpr (Mode == integrate_mode::evaluate)
            {
                T Ig = T(gk::wG_center) * f0;
                T Ik = T(gk::wK0) * f0;

                Ik = Ik + T(gk::wK1) * (f1m + f1p);

                {
                    T const s = f2m + f2p;
                    Ik = Ik + T(gk::wK2) * s;
                    Ig = Ig + T(gk::wG_x2) * s;
                }

                Ik = Ik + T(gk::wK3) * (f3m + f3p);

                {
                    T const s = f4m + f4p;
                    Ik = Ik + T(gk::wK4) * s;
                    Ig = Ig + T(gk::wG_x4) * s;
                }

                Ik = Ik + T(gk::wK5) * (f5m + f5p);

                {
                    T const s = f6m + f6p;
                    Ik = Ik + T(gk::wK6) * s;
                    Ig = Ig + T(gk::wG_x6) * s;
                }

                Ik = Ik + T(gk::wK7) * (f7m + f7p);

                Ig = Ig * h;
                Ik = Ik * h;

                T const err = abs(Ik - Ig);
                bool const at_max = depth >= cfg.max_depth;
                bool const can_accept = depth >= cfg.min_depth && err <= local_tol;

                if (at_max || can_accept)
                {
                    total_err = total_err + err;
                    total_intervals += 1;
                    return Ik;
                }

                T const half_tol = local_tol * T(0.5);
                T const left = self(self, idx * 2 + 1, aa, c, half_tol, depth + 1);
                T const right = self(self, idx * 2 + 2, c, bb, half_tol, depth + 1);
                return left + right;
            }
            else // preinvoke
            {
                (void)f0;
                (void)f1m; (void)f1p; (void)f2m; (void)f2p;
                (void)f3m; (void)f3p; (void)f4m; (void)f4p;
                (void)f5m; (void)f5p; (void)f6m; (void)f6p;
                (void)f7m; (void)f7p;
                (void)local_tol;

                if (depth >= cfg.max_depth)
                    return;

                self(self, idx * 2 + 1, aa, c, local_tol, depth + 1);
                self(self, idx * 2 + 2, c, bb, local_tol, depth + 1);
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
} // namespace antipodal
