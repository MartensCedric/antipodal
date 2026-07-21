#pragma once

#include <antipodal/math/common.hh>

#include <cmath>
#include <format>
#include <string>
#include <type_traits>

namespace antipodal
{
template <class T>
struct fwd_diff_diff;

using fdd32 = fwd_diff_diff<float>;
using fdd64 = fwd_diff_diff<double>;

/// Second-order forward-mode automatic differentiation scalar type.
///
/// This type implements a "hyper-dual number" that tracks a value along with its
/// first and second derivatives with respect to a single input variable.
/// It enables computing f(x), f'(x), and f''(x) simultaneously in a single forward pass.
///
/// Mathematical model:
///   Each fwd_diff_diff represents: v + v'·ε + v''·ε²/2
///   where ε is an infinitesimal with ε³ = 0.
///   Arithmetic and math functions propagate derivatives via the chain rule.
///
/// Usage:
///   auto x = fwd_diff_diff<double>::input(2.0);  // x = 2, dx/dx = 1, d²x/dx² = 0
///   auto y = x * x + sin(x);                     // computes y, dy/dx, d²y/dx² at x=2
///   // y.value = result, y.d_value = first derivative, y.dd_value = second derivative
///
/// Supported operations:
///   - Arithmetic: +, -, *, / (including mixed fwd_diff_diff<T> and T)
///   - Comparisons: ==, !=, <, <=, >, >= (compare values only, ignore derivatives)
///   - Math functions: abs, min, max, sqrt, pow, log, log2, log10, exp,
///                     sin, cos, tan, asin, acos, atan, atan2
///
/// Trig functions take and return plain fwd_diff_diff<T>; angle values are
/// interpreted as radians (matching std::sin / std::atan2).
///
/// See also: dual numbers, hyper-dual numbers, Taylor arithmetic
template <class T>
struct fwd_diff_diff
{
    T value = {};    ///< f(x) - the function value
    T d_value = {};  ///< f'(x) - first derivative with respect to input
    T dd_value = {}; ///< f''(x) - second derivative with respect to input

    // -- static factory functions --

    /// creates a constant (derivatives are zero)
    [[nodiscard]] static constexpr fwd_diff_diff constant(T v) { return {v, T(0), T(0)}; }

    /// creates an input variable (first derivative is 1, second is 0)
    [[nodiscard]] static constexpr fwd_diff_diff input(T v) { return {v, T(1), T(0)}; }

    // -- constructors --

    constexpr fwd_diff_diff() = default;
    explicit constexpr fwd_diff_diff(T v) : value(v), d_value(T(0)), dd_value(T(0)) {}
    constexpr fwd_diff_diff(T v, T dv, T ddv) : value(v), d_value(dv), dd_value(ddv) {}

    // -- comparisons (only compare value, not derivatives) --

    [[nodiscard]] constexpr bool operator==(fwd_diff_diff const& rhs) const { return value == rhs.value; }
    [[nodiscard]] constexpr bool operator!=(fwd_diff_diff const& rhs) const { return value != rhs.value; }
    [[nodiscard]] constexpr bool operator<(fwd_diff_diff const& rhs) const { return value < rhs.value; }
    [[nodiscard]] constexpr bool operator<=(fwd_diff_diff const& rhs) const { return value <= rhs.value; }
    [[nodiscard]] constexpr bool operator>(fwd_diff_diff const& rhs) const { return value > rhs.value; }
    [[nodiscard]] constexpr bool operator>=(fwd_diff_diff const& rhs) const { return value >= rhs.value; }

    // comparisons with T
    [[nodiscard]] constexpr bool operator==(T const& rhs) const { return value == rhs; }
    [[nodiscard]] constexpr bool operator!=(T const& rhs) const { return value != rhs; }
    [[nodiscard]] constexpr bool operator<(T const& rhs) const { return value < rhs; }
    [[nodiscard]] constexpr bool operator<=(T const& rhs) const { return value <= rhs; }
    [[nodiscard]] constexpr bool operator>(T const& rhs) const { return value > rhs; }
    [[nodiscard]] constexpr bool operator>=(T const& rhs) const { return value >= rhs; }

    // -- type requirements --
    static_assert(std::is_same_v<decltype(std::declval<T>() + std::declval<T>()), T>, "T + T must return T");
    static_assert(std::is_same_v<decltype(std::declval<T>() - std::declval<T>()), T>, "T - T must return T");
    static_assert(std::is_same_v<decltype(std::declval<T>() * std::declval<T>()), T>, "T * T must return T");
    static_assert(std::is_same_v<decltype(std::declval<T>() / std::declval<T>()), T>, "T / T must return T");
    static_assert(std::is_convertible_v<decltype(std::declval<T>() == std::declval<T>()), bool>,
                  "T == T must be convertible to bool");
    static_assert(std::is_convertible_v<decltype(std::declval<T>() < std::declval<T>()), bool>,
                  "T < T must be convertible to bool");
};

// -- reverse comparisons (T vs fwd_diff_diff) --

template <class T>
[[nodiscard]] constexpr bool operator==(T const& lhs, fwd_diff_diff<T> const& rhs)
{
    return lhs == rhs.value;
}
template <class T>
[[nodiscard]] constexpr bool operator!=(T const& lhs, fwd_diff_diff<T> const& rhs)
{
    return lhs != rhs.value;
}
template <class T>
[[nodiscard]] constexpr bool operator<(T const& lhs, fwd_diff_diff<T> const& rhs)
{
    return lhs < rhs.value;
}
template <class T>
[[nodiscard]] constexpr bool operator<=(T const& lhs, fwd_diff_diff<T> const& rhs)
{
    return lhs <= rhs.value;
}
template <class T>
[[nodiscard]] constexpr bool operator>(T const& lhs, fwd_diff_diff<T> const& rhs)
{
    return lhs > rhs.value;
}
template <class T>
[[nodiscard]] constexpr bool operator>=(T const& lhs, fwd_diff_diff<T> const& rhs)
{
    return lhs >= rhs.value;
}

// -- tests (operate on value only) --

template <class T>
[[nodiscard]] bool is_finite(fwd_diff_diff<T> const& v)
{
    return std::isfinite(v.value);
}
template <class T>
[[nodiscard]] bool is_nan(fwd_diff_diff<T> const& v)
{
    return std::isnan(v.value);
}
template <class T>
[[nodiscard]] bool is_inf(fwd_diff_diff<T> const& v)
{
    return std::isinf(v.value);
}

// -- projections --

template <class T>
[[nodiscard]] constexpr vec3<T> value_of(vec3<fwd_diff_diff<T>> const& v)
{
    return {v.x.value, v.y.value, v.z.value};
}
template <class T>
[[nodiscard]] constexpr vec3<T> d_value_of(vec3<fwd_diff_diff<T>> const& v)
{
    return {v.x.d_value, v.y.d_value, v.z.d_value};
}
template <class T>
[[nodiscard]] constexpr vec3<T> dd_value_of(vec3<fwd_diff_diff<T>> const& v)
{
    return {v.x.dd_value, v.y.dd_value, v.z.dd_value};
}

template <class T>
[[nodiscard]] constexpr vec2<T> value_of(vec2<fwd_diff_diff<T>> const& v)
{
    return {v.x.value, v.y.value};
}
template <class T>
[[nodiscard]] constexpr vec2<T> d_value_of(vec2<fwd_diff_diff<T>> const& v)
{
    return {v.x.d_value, v.y.d_value};
}
template <class T>
[[nodiscard]] constexpr vec2<T> dd_value_of(vec2<fwd_diff_diff<T>> const& v)
{
    return {v.x.dd_value, v.y.dd_value};
}

// -- unary operators --

template <class T>
[[nodiscard]] constexpr fwd_diff_diff<T> operator+(fwd_diff_diff<T> const& v)
{
    return v;
}
template <class T>
[[nodiscard]] constexpr fwd_diff_diff<T> operator-(fwd_diff_diff<T> const& v)
{
    return {-v.value, -v.d_value, -v.dd_value};
}

// -- binary arithmetic: fwd_diff_diff vs fwd_diff_diff --

// (a + b)' = a' + b'
// (a + b)'' = a'' + b''
template <class T>
[[nodiscard]] constexpr fwd_diff_diff<T> operator+(fwd_diff_diff<T> const& a, fwd_diff_diff<T> const& b)
{
    return {a.value + b.value, a.d_value + b.d_value, a.dd_value + b.dd_value};
}

// (a - b)' = a' - b'
// (a - b)'' = a'' - b''
template <class T>
[[nodiscard]] constexpr fwd_diff_diff<T> operator-(fwd_diff_diff<T> const& a, fwd_diff_diff<T> const& b)
{
    return {a.value - b.value, a.d_value - b.d_value, a.dd_value - b.dd_value};
}

// (a * b)' = a' * b + a * b'
// (a * b)'' = a'' * b + 2 * a' * b' + a * b''
template <class T>
[[nodiscard]] constexpr fwd_diff_diff<T> operator*(fwd_diff_diff<T> const& a, fwd_diff_diff<T> const& b)
{
    return {a.value * b.value, a.d_value * b.value + a.value * b.d_value,
            a.dd_value * b.value + T(2) * a.d_value * b.d_value + a.value * b.dd_value};
}

// (a / b)' = (a' * b - a * b') / b^2
// (a / b)'' = (a'' - 2 * (a/b)' * b' - (a/b) * b'') / b
template <class T>
[[nodiscard]] constexpr fwd_diff_diff<T> operator/(fwd_diff_diff<T> const& a, fwd_diff_diff<T> const& b)
{
    auto const inv_b = T(1) / b.value;
    auto const q = a.value * inv_b; // a / b
    auto const dq = (a.d_value - q * b.d_value) * inv_b;
    auto const ddq = (a.dd_value - T(2) * dq * b.d_value - q * b.dd_value) * inv_b;
    return {q, dq, ddq};
}

// -- binary arithmetic: fwd_diff_diff vs T --

template <class T>
[[nodiscard]] constexpr fwd_diff_diff<T> operator+(fwd_diff_diff<T> const& a, T const& b)
{
    return {a.value + b, a.d_value, a.dd_value};
}
template <class T>
[[nodiscard]] constexpr fwd_diff_diff<T> operator-(fwd_diff_diff<T> const& a, T const& b)
{
    return {a.value - b, a.d_value, a.dd_value};
}
template <class T>
[[nodiscard]] constexpr fwd_diff_diff<T> operator*(fwd_diff_diff<T> const& a, T const& b)
{
    return {a.value * b, a.d_value * b, a.dd_value * b};
}
template <class T>
[[nodiscard]] constexpr fwd_diff_diff<T> operator/(fwd_diff_diff<T> const& a, T const& b)
{
    auto const inv_b = T(1) / b;
    return {a.value * inv_b, a.d_value * inv_b, a.dd_value * inv_b};
}

// -- binary arithmetic: T vs fwd_diff_diff --

template <class T>
[[nodiscard]] constexpr fwd_diff_diff<T> operator+(T const& a, fwd_diff_diff<T> const& b)
{
    return {a + b.value, b.d_value, b.dd_value};
}
template <class T>
[[nodiscard]] constexpr fwd_diff_diff<T> operator-(T const& a, fwd_diff_diff<T> const& b)
{
    return {a - b.value, -b.d_value, -b.dd_value};
}
template <class T>
[[nodiscard]] constexpr fwd_diff_diff<T> operator*(T const& a, fwd_diff_diff<T> const& b)
{
    return {a * b.value, a * b.d_value, a * b.dd_value};
}
// a / b where a is constant
template <class T>
[[nodiscard]] constexpr fwd_diff_diff<T> operator/(T const& a, fwd_diff_diff<T> const& b)
{
    auto const inv_b = T(1) / b.value;
    auto const q = a * inv_b;
    auto const dq = -q * b.d_value * inv_b;
    auto const ddq = (-q * b.dd_value - T(2) * dq * b.d_value) * inv_b;
    return {q, dq, ddq};
}

// -- mathematical functions --

// abs(x)' = sign(x) * x'
// abs(x)'' = sign(x) * x''  (discontinuous at x=0, but derivative is 0 there anyway for smooth functions)
template <class T>
[[nodiscard]] constexpr fwd_diff_diff<T> abs(fwd_diff_diff<T> const& x)
{
    if (x.value < T(0))
        return {-x.value, -x.d_value, -x.dd_value};
    return x;
}

// min(a, b) - returns the one with smaller value, preserving its derivatives
template <class T>
[[nodiscard]] constexpr fwd_diff_diff<T> min(fwd_diff_diff<T> const& a, fwd_diff_diff<T> const& b)
{
    return a.value <= b.value ? a : b;
}
template <class T>
[[nodiscard]] constexpr fwd_diff_diff<T> min(fwd_diff_diff<T> const& a, T const& b)
{
    return a.value <= b ? a : fwd_diff_diff<T>::constant(b);
}
template <class T>
[[nodiscard]] constexpr fwd_diff_diff<T> min(T const& a, fwd_diff_diff<T> const& b)
{
    return a <= b.value ? fwd_diff_diff<T>::constant(a) : b;
}

// max(a, b) - returns the one with larger value, preserving its derivatives
template <class T>
[[nodiscard]] constexpr fwd_diff_diff<T> max(fwd_diff_diff<T> const& a, fwd_diff_diff<T> const& b)
{
    return a.value >= b.value ? a : b;
}
template <class T>
[[nodiscard]] constexpr fwd_diff_diff<T> max(fwd_diff_diff<T> const& a, T const& b)
{
    return a.value >= b ? a : fwd_diff_diff<T>::constant(b);
}
template <class T>
[[nodiscard]] constexpr fwd_diff_diff<T> max(T const& a, fwd_diff_diff<T> const& b)
{
    return a >= b.value ? fwd_diff_diff<T>::constant(a) : b;
}

// sqrt(x)' = x' / (2 * sqrt(x))
// sqrt(x)'' = x'' / (2 * sqrt(x)) - x'^2 / (4 * x * sqrt(x))
template <class T>
[[nodiscard]] fwd_diff_diff<T> sqrt(fwd_diff_diff<T> const& x)
{
    auto const s = std::sqrt(x.value);
    auto const inv_2s = T(0.5) / s;
    auto const ds = x.d_value * inv_2s;
    auto const dds = x.dd_value * inv_2s - ds * ds * (inv_2s + inv_2s);
    return {s, ds, dds};
}

// pow(x, n) where n is constant
// f = x^n, f' = n * x^(n-1) * x', f'' = n * (n-1) * x^(n-2) * x'^2 + n * x^(n-1) * x''
template <class T>
[[nodiscard]] fwd_diff_diff<T> pow(fwd_diff_diff<T> const& x, T const& n)
{
    auto const xn_1 = std::pow(x.value, n - T(1));
    auto const xn = xn_1 * x.value;
    auto const df_dx = n * xn_1;
    auto const d2f_dx2 = n * (n - T(1)) * std::pow(x.value, n - T(2));
    return {xn, df_dx * x.d_value, d2f_dx2 * x.d_value * x.d_value + df_dx * x.dd_value};
}

// pow(x, y) general case: x^y = exp(y * log(x))
// Let u = y * log(x), then f = exp(u)
// f' = exp(u) * u'
// f'' = exp(u) * (u'^2 + u'')
template <class T>
[[nodiscard]] fwd_diff_diff<T> pow(fwd_diff_diff<T> const& x, fwd_diff_diff<T> const& y)
{
    auto const log_x = std::log(x.value);
    auto const u = y.value * log_x;
    auto const inv_x = T(1) / x.value;
    // u' = y' * log(x) + y * x' / x
    auto const du = y.d_value * log_x + y.value * x.d_value * inv_x;
    // u'' = y'' * log(x) + 2 * y' * x' / x + y * (x'' / x - x'^2 / x^2)
    auto const ddu = y.dd_value * log_x + T(2) * y.d_value * x.d_value * inv_x
                   + y.value * (x.dd_value * inv_x - x.d_value * x.d_value * inv_x * inv_x);
    auto const exp_u = std::exp(u);
    return {exp_u, exp_u * du, exp_u * (du * du + ddu)};
}

// log(x)' = x' / x
// log(x)'' = x'' / x - (x' / x)^2
template <class T>
[[nodiscard]] fwd_diff_diff<T> log(fwd_diff_diff<T> const& x)
{
    auto const inv_x = T(1) / x.value;
    auto const d_log = x.d_value * inv_x;
    auto const dd_log = x.dd_value * inv_x - d_log * d_log;
    return {std::log(x.value), d_log, dd_log};
}

// log2(x) = log(x) / log(2)
template <class T>
[[nodiscard]] fwd_diff_diff<T> log2(fwd_diff_diff<T> const& x)
{
    auto const inv_log2 = T(1) / std::log(T(2));
    auto const inv_x = T(1) / x.value;
    auto const d_log = x.d_value * inv_x * inv_log2;
    auto const dd_log = (x.dd_value * inv_x - x.d_value * x.d_value * inv_x * inv_x) * inv_log2;
    return {std::log2(x.value), d_log, dd_log};
}

// log10(x) = log(x) / log(10)
template <class T>
[[nodiscard]] fwd_diff_diff<T> log10(fwd_diff_diff<T> const& x)
{
    auto const inv_log10 = T(1) / std::log(T(10));
    auto const inv_x = T(1) / x.value;
    auto const d_log = x.d_value * inv_x * inv_log10;
    auto const dd_log = (x.dd_value * inv_x - x.d_value * x.d_value * inv_x * inv_x) * inv_log10;
    return {std::log10(x.value), d_log, dd_log};
}

// exp(x)' = exp(x) * x'
// exp(x)'' = exp(x) * (x'^2 + x'')
template <class T>
[[nodiscard]] fwd_diff_diff<T> exp(fwd_diff_diff<T> const& x)
{
    auto const e = std::exp(x.value);
    return {e, e * x.d_value, e * (x.d_value * x.d_value + x.dd_value)};
}

// sin(x)' = cos(x) * x'
// sin(x)'' = -sin(x) * x'^2 + cos(x) * x''
// x is interpreted as radians.
template <class T>
[[nodiscard]] fwd_diff_diff<T> sin(fwd_diff_diff<T> const& x)
{
    auto const s = std::sin(x.value);
    auto const c = std::cos(x.value);
    return {s, c * x.d_value, -s * x.d_value * x.d_value + c * x.dd_value};
}

// cos(x)' = -sin(x) * x'
// cos(x)'' = -cos(x) * x'^2 - sin(x) * x''
// x is interpreted as radians.
template <class T>
[[nodiscard]] fwd_diff_diff<T> cos(fwd_diff_diff<T> const& x)
{
    auto const s = std::sin(x.value);
    auto const c = std::cos(x.value);
    return {c, -s * x.d_value, -c * x.d_value * x.d_value - s * x.dd_value};
}

// tan(x)' = sec^2(x) * x' = (1 + tan^2(x)) * x'
// tan(x)'' = 2 * tan(x) * sec^2(x) * x'^2 + sec^2(x) * x''
// x is interpreted as radians.
template <class T>
[[nodiscard]] fwd_diff_diff<T> tan(fwd_diff_diff<T> const& x)
{
    auto const t = std::tan(x.value);
    auto const sec2 = T(1) + t * t;
    return {t, sec2 * x.d_value, T(2) * t * sec2 * x.d_value * x.d_value + sec2 * x.dd_value};
}

// asin(x)' = x' / sqrt(1 - x^2)
// asin(x)'' = x'' / sqrt(1-x^2) + x * x'^2 / (1-x^2)^(3/2)
// Returns radians.
template <class T>
[[nodiscard]] fwd_diff_diff<T> asin(fwd_diff_diff<T> const& x)
{
    auto const one_minus_x2 = T(1) - x.value * x.value;
    auto const sqrt_1mx2 = std::sqrt(one_minus_x2);
    auto const inv_sqrt = T(1) / sqrt_1mx2;
    auto const d_asin = x.d_value * inv_sqrt;
    auto const dd_asin = x.dd_value * inv_sqrt + x.value * x.d_value * x.d_value * inv_sqrt / one_minus_x2;
    return {std::asin(x.value), d_asin, dd_asin};
}

// acos(x)' = -x' / sqrt(1 - x^2)
// acos(x)'' = -x'' / sqrt(1-x^2) - x * x'^2 / (1-x^2)^(3/2)
// Returns radians.
template <class T>
[[nodiscard]] fwd_diff_diff<T> acos(fwd_diff_diff<T> const& x)
{
    auto const one_minus_x2 = T(1) - x.value * x.value;
    auto const sqrt_1mx2 = std::sqrt(one_minus_x2);
    auto const inv_sqrt = T(1) / sqrt_1mx2;
    auto const d_acos = -x.d_value * inv_sqrt;
    auto const dd_acos = -x.dd_value * inv_sqrt - x.value * x.d_value * x.d_value * inv_sqrt / one_minus_x2;
    return {std::acos(x.value), d_acos, dd_acos};
}

// atan(x)' = x' / (1 + x^2)
// atan(x)'' = x'' / (1 + x^2) - 2 * x * x'^2 / (1 + x^2)^2
// Returns radians.
template <class T>
[[nodiscard]] fwd_diff_diff<T> atan(fwd_diff_diff<T> const& x)
{
    auto const one_plus_x2 = T(1) + x.value * x.value;
    auto const inv_1px2 = T(1) / one_plus_x2;
    auto const d_atan = x.d_value * inv_1px2;
    auto const dd_atan = x.dd_value * inv_1px2 - T(2) * x.value * x.d_value * x.d_value * inv_1px2 * inv_1px2;
    return {std::atan(x.value), d_atan, dd_atan};
}

// atan2(y, x) = atan(y/x) with quadrant correction. Returns radians.
// d(atan2)/dx = -y / r^2,  d(atan2)/dy = x / r^2,  r^2 = x^2 + y^2
// atan2' = (x * y' - y * x') / r^2
template <class T>
[[nodiscard]] fwd_diff_diff<T> atan2(fwd_diff_diff<T> const& y, fwd_diff_diff<T> const& x)
{
    auto const r2 = x.value * x.value + y.value * y.value;
    auto const inv_r2 = T(1) / r2;
    auto const num = x.value * y.d_value - y.value * x.d_value;
    auto const df = num * inv_r2;
    auto const dnum = x.value * y.dd_value - y.value * x.dd_value;
    auto const dr2 = T(2) * (x.value * x.d_value + y.value * y.d_value);
    auto const ddf = dnum * inv_r2 - num * dr2 * inv_r2 * inv_r2;
    return {std::atan2(y.value, x.value), df, ddf};
}

template <class T>
[[nodiscard]] fwd_diff_diff<T> atan2(fwd_diff_diff<T> const& y, T const& x)
{
    auto const r2 = x * x + y.value * y.value;
    auto const inv_r2 = T(1) / r2;
    auto const df = x * y.d_value * inv_r2;
    auto const dr2 = T(2) * y.value * y.d_value;
    auto const ddf = x * y.dd_value * inv_r2 - x * y.d_value * dr2 * inv_r2 * inv_r2;
    return {std::atan2(y.value, x), df, ddf};
}

template <class T>
[[nodiscard]] fwd_diff_diff<T> atan2(T const& y, fwd_diff_diff<T> const& x)
{
    auto const r2 = x.value * x.value + y * y;
    auto const inv_r2 = T(1) / r2;
    auto const df = -y * x.d_value * inv_r2;
    auto const dr2 = T(2) * x.value * x.d_value;
    auto const ddf = -y * x.dd_value * inv_r2 + y * x.d_value * dr2 * inv_r2 * inv_r2;
    return {std::atan2(y, x.value), df, ddf};
}

// -- mixed vec3 ops (only those needed for now) --

template <class T>
[[nodiscard]] fwd_diff_diff<T> dot(vec3<fwd_diff_diff<T>> const& l, vec3<T> const& r)
{
    return l.x * r.x + l.y * r.y + l.z * r.z;
}

template <class T>
[[nodiscard]] vec3<fwd_diff_diff<T>> operator*(vec3<T> const& l, fwd_diff_diff<T> const& r)
{
    return {l.x * r, l.y * r, l.z * r};
}

template <class T>
[[nodiscard]] vec3<fwd_diff_diff<T>> operator+(vec3<T> const& l, vec3<fwd_diff_diff<T>> const& r)
{
    return {l.x + r.x, l.y + r.y, l.z + r.z};
}

template <class T>
[[nodiscard]] vec3<fwd_diff_diff<T>> operator-(vec3<fwd_diff_diff<T>> const& l, vec3<T> const& r)
{
    return {l.x - r.x, l.y - r.y, l.z - r.z};
}

// Specialized normalize for 3D vectors with fwd_diff_diff scalars.
// This avoids propagating sqrt/div through autodiff and instead
// computes value, first and second derivatives analytically in T.
template <class T>
[[nodiscard]] vec3<fwd_diff_diff<T>> normalize_fdd(vec3<fwd_diff_diff<T>> const& v)
{
    // Extract values and derivatives of v(t)
    T const vx0 = v.x.value;
    T const vx1 = v.y.value;
    T const vx2 = v.z.value;

    T const dvx0 = v.x.d_value;
    T const dvx1 = v.y.d_value;
    T const dvx2 = v.z.d_value;

    T const ddvx0 = v.x.dd_value;
    T const ddvx1 = v.y.dd_value;
    T const ddvx2 = v.z.dd_value;

    // q = ||v||^2, and its first and second derivatives
    // q  = v·v
    // q' = 2 * v·v'
    // q''= 2 * (||v'||^2 + v·v'')
    T const q = vx0 * vx0 + vx1 * vx1 + vx2 * vx2;
    T const q1 = T(2) * (vx0 * dvx0 + vx1 * dvx1 + vx2 * dvx2);
    T const q2 = T(2) * (dvx0 * dvx0 + dvx1 * dvx1 + dvx2 * dvx2 + vx0 * ddvx0 + vx1 * ddvx1 + vx2 * ddvx2);

    // s = sqrt(q), s' and s'' via scalar chain rule
    // s'  = q' / (2 * s)
    // s'' = q'' / (2 * s) - (q'^2) / (4 * s^3)
    T const s = std::sqrt(q); // q > 0 is assumed (same as generic normalize)
    T const inv_s = T(1) / s;
    T const inv_s3 = inv_s * inv_s * inv_s;

    T const s1 = q1 * (T(0.5) * inv_s);
    T const s2 = q2 * (T(0.5) * inv_s) - q1 * q1 * (T(0.25) * inv_s3);

    // For each component: x_i = v_i / s
    // Use the same division rule as in fwd_diff_diff:
    //   q  = a / b
    //   q' = (a' - q * b') / b
    //   q''= (a'' - 2 * q' * b' - q * b'') / b
    auto make_component = [&](T vi, T dvi, T ddvi) -> fwd_diff_diff<T>
    {
        T const xi = vi * inv_s;
        T const dxi = (dvi - xi * s1) * inv_s;
        T const ddxi = (ddvi - T(2) * dxi * s1 - xi * s2) * inv_s;
        return {xi, dxi, ddxi};
    };

    return {make_component(vx0, dvx0, ddvx0), make_component(vx1, dvx1, ddvx1), make_component(vx2, dvx2, ddvx2)};
}

template <class T>
[[nodiscard]] inline std::string to_string(fwd_diff_diff<T> const& x)
{
    return std::format("fdd({}, {}, {})", x.value, x.d_value, x.dd_value);
}

namespace fdd_detail
{
inline double value_scalar(double x)
{
    return x;
}

template <class S>
inline double value_scalar(fwd_diff_diff<S> const& x)
{
    return static_cast<double>(x.value);
}
} // namespace fdd_detail

} // namespace antipodal
