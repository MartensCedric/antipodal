#pragma once

#include <antipodal/math/common.hh>

#include <cmath>
#include <format>
#include <string>
#include <type_traits>

namespace antipodal
{
template <class T>
struct fwd_diff;

using fd32 = fwd_diff<float>;
using fd64 = fwd_diff<double>;

/// First-order forward-mode automatic differentiation scalar type.
///
/// This type implements a "dual number" that tracks a value along with its
/// first derivative with respect to a single input variable.
/// It enables computing f(x) and f'(x) simultaneously in a single forward pass.
///
/// Mathematical model:
///   Each fwd_diff represents: v + v'·ε
///   where ε is an infinitesimal with ε² = 0.
///   Arithmetic and math functions propagate derivatives via the chain rule.
///
/// Usage:
///   auto x = fwd_diff<double>::input(2.0);  // x = 2, dx/dx = 1
///   auto y = x * x + sin(x);                // computes y, dy/dx at x=2
///   // y.value = result, y.d_value = first derivative
///
/// Supported operations:
///   - Arithmetic: +, -, *, / (including mixed fwd_diff<T> and T)
///   - Comparisons: ==, !=, <, <=, >, >= (compare values only, ignore derivatives)
///   - Math functions: abs, min, max, sqrt, pow, log, log2, log10, exp,
///                     sin, cos, tan, asin, acos, atan, atan2
///
/// Trig functions take and return plain fwd_diff<T>; angle values are
/// interpreted as radians (matching std::sin / std::atan2).
///
/// See also: dual numbers, fwd_diff_diff (second-order variant)
template <class T>
struct fwd_diff
{
    T value = {};   ///< f(x) - the function value
    T d_value = {}; ///< f'(x) - first derivative with respect to input

    // -- static factory functions --

    /// creates a constant (derivative is zero)
    [[nodiscard]] static constexpr fwd_diff constant(T v) { return {v, T(0)}; }

    /// creates an input variable (first derivative is 1)
    [[nodiscard]] static constexpr fwd_diff input(T v) { return {v, T(1)}; }

    // -- constructors --

    constexpr fwd_diff() = default;
    explicit constexpr fwd_diff(T v) : value(v), d_value(T(0)) {}
    constexpr fwd_diff(T v, T dv) : value(v), d_value(dv) {}

    // -- comparisons (only compare value, not derivatives) --

    [[nodiscard]] constexpr bool operator==(fwd_diff const& rhs) const { return value == rhs.value; }
    [[nodiscard]] constexpr bool operator!=(fwd_diff const& rhs) const { return value != rhs.value; }
    [[nodiscard]] constexpr bool operator<(fwd_diff const& rhs) const { return value < rhs.value; }
    [[nodiscard]] constexpr bool operator<=(fwd_diff const& rhs) const { return value <= rhs.value; }
    [[nodiscard]] constexpr bool operator>(fwd_diff const& rhs) const { return value > rhs.value; }
    [[nodiscard]] constexpr bool operator>=(fwd_diff const& rhs) const { return value >= rhs.value; }

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

// -- reverse comparisons (T vs fwd_diff) --

template <class T>
[[nodiscard]] constexpr bool operator==(T const& lhs, fwd_diff<T> const& rhs)
{
    return lhs == rhs.value;
}
template <class T>
[[nodiscard]] constexpr bool operator!=(T const& lhs, fwd_diff<T> const& rhs)
{
    return lhs != rhs.value;
}
template <class T>
[[nodiscard]] constexpr bool operator<(T const& lhs, fwd_diff<T> const& rhs)
{
    return lhs < rhs.value;
}
template <class T>
[[nodiscard]] constexpr bool operator<=(T const& lhs, fwd_diff<T> const& rhs)
{
    return lhs <= rhs.value;
}
template <class T>
[[nodiscard]] constexpr bool operator>(T const& lhs, fwd_diff<T> const& rhs)
{
    return lhs > rhs.value;
}
template <class T>
[[nodiscard]] constexpr bool operator>=(T const& lhs, fwd_diff<T> const& rhs)
{
    return lhs >= rhs.value;
}

// -- tests (operate on value only) --

template <class T>
[[nodiscard]] bool is_finite(fwd_diff<T> const& v)
{
    return std::isfinite(v.value);
}
template <class T>
[[nodiscard]] bool is_nan(fwd_diff<T> const& v)
{
    return std::isnan(v.value);
}
template <class T>
[[nodiscard]] bool is_inf(fwd_diff<T> const& v)
{
    return std::isinf(v.value);
}

// -- projections --

template <class T>
[[nodiscard]] constexpr vec3<T> value_of(vec3<fwd_diff<T>> const& v)
{
    return {v.x.value, v.y.value, v.z.value};
}
template <class T>
[[nodiscard]] constexpr vec3<T> d_value_of(vec3<fwd_diff<T>> const& v)
{
    return {v.x.d_value, v.y.d_value, v.z.d_value};
}

template <class T>
[[nodiscard]] constexpr vec2<T> value_of(vec2<fwd_diff<T>> const& v)
{
    return {v.x.value, v.y.value};
}
template <class T>
[[nodiscard]] constexpr vec2<T> d_value_of(vec2<fwd_diff<T>> const& v)
{
    return {v.x.d_value, v.y.d_value};
}

// -- unary operators --

template <class T>
[[nodiscard]] constexpr fwd_diff<T> operator+(fwd_diff<T> const& v)
{
    return v;
}
template <class T>
[[nodiscard]] constexpr fwd_diff<T> operator-(fwd_diff<T> const& v)
{
    return {-v.value, -v.d_value};
}

// -- binary arithmetic: fwd_diff vs fwd_diff --

// (a + b)' = a' + b'
template <class T>
[[nodiscard]] constexpr fwd_diff<T> operator+(fwd_diff<T> const& a, fwd_diff<T> const& b)
{
    return {a.value + b.value, a.d_value + b.d_value};
}

// (a - b)' = a' - b'
template <class T>
[[nodiscard]] constexpr fwd_diff<T> operator-(fwd_diff<T> const& a, fwd_diff<T> const& b)
{
    return {a.value - b.value, a.d_value - b.d_value};
}

// (a * b)' = a' * b + a * b'
template <class T>
[[nodiscard]] constexpr fwd_diff<T> operator*(fwd_diff<T> const& a, fwd_diff<T> const& b)
{
    return {a.value * b.value, a.d_value * b.value + a.value * b.d_value};
}

// (a / b)' = (a' - (a/b) * b') / b
template <class T>
[[nodiscard]] constexpr fwd_diff<T> operator/(fwd_diff<T> const& a, fwd_diff<T> const& b)
{
    auto const inv_b = T(1) / b.value;
    auto const q = a.value * inv_b; // a / b
    auto const dq = (a.d_value - q * b.d_value) * inv_b;
    return {q, dq};
}

// -- binary arithmetic: fwd_diff vs T --

template <class T>
[[nodiscard]] constexpr fwd_diff<T> operator+(fwd_diff<T> const& a, T const& b)
{
    return {a.value + b, a.d_value};
}
template <class T>
[[nodiscard]] constexpr fwd_diff<T> operator-(fwd_diff<T> const& a, T const& b)
{
    return {a.value - b, a.d_value};
}
template <class T>
[[nodiscard]] constexpr fwd_diff<T> operator*(fwd_diff<T> const& a, T const& b)
{
    return {a.value * b, a.d_value * b};
}
template <class T>
[[nodiscard]] constexpr fwd_diff<T> operator/(fwd_diff<T> const& a, T const& b)
{
    auto const inv_b = T(1) / b;
    return {a.value * inv_b, a.d_value * inv_b};
}

// -- binary arithmetic: T vs fwd_diff --

template <class T>
[[nodiscard]] constexpr fwd_diff<T> operator+(T const& a, fwd_diff<T> const& b)
{
    return {a + b.value, b.d_value};
}
template <class T>
[[nodiscard]] constexpr fwd_diff<T> operator-(T const& a, fwd_diff<T> const& b)
{
    return {a - b.value, -b.d_value};
}
template <class T>
[[nodiscard]] constexpr fwd_diff<T> operator*(T const& a, fwd_diff<T> const& b)
{
    return {a * b.value, a * b.d_value};
}
// a / b where a is constant
template <class T>
[[nodiscard]] constexpr fwd_diff<T> operator/(T const& a, fwd_diff<T> const& b)
{
    auto const inv_b = T(1) / b.value;
    auto const q = a * inv_b;
    auto const dq = -q * b.d_value * inv_b;
    return {q, dq};
}

// -- mathematical functions --

// abs(x)' = sign(x) * x'
template <class T>
[[nodiscard]] constexpr fwd_diff<T> abs(fwd_diff<T> const& x)
{
    if (x.value < T(0))
        return {-x.value, -x.d_value};
    return x;
}

// min(a, b) - returns the one with smaller value, preserving its derivative
template <class T>
[[nodiscard]] constexpr fwd_diff<T> min(fwd_diff<T> const& a, fwd_diff<T> const& b)
{
    return a.value <= b.value ? a : b;
}
template <class T>
[[nodiscard]] constexpr fwd_diff<T> min(fwd_diff<T> const& a, T const& b)
{
    return a.value <= b ? a : fwd_diff<T>::constant(b);
}
template <class T>
[[nodiscard]] constexpr fwd_diff<T> min(T const& a, fwd_diff<T> const& b)
{
    return a <= b.value ? fwd_diff<T>::constant(a) : b;
}

// max(a, b) - returns the one with larger value, preserving its derivative
template <class T>
[[nodiscard]] constexpr fwd_diff<T> max(fwd_diff<T> const& a, fwd_diff<T> const& b)
{
    return a.value >= b.value ? a : b;
}
template <class T>
[[nodiscard]] constexpr fwd_diff<T> max(fwd_diff<T> const& a, T const& b)
{
    return a.value >= b ? a : fwd_diff<T>::constant(b);
}
template <class T>
[[nodiscard]] constexpr fwd_diff<T> max(T const& a, fwd_diff<T> const& b)
{
    return a >= b.value ? fwd_diff<T>::constant(a) : b;
}

// sqrt(x)' = x' / (2 * sqrt(x))
template <class T>
[[nodiscard]] fwd_diff<T> sqrt(fwd_diff<T> const& x)
{
    auto const s = std::sqrt(x.value);
    auto const inv_2s = T(0.5) / s;
    return {s, x.d_value * inv_2s};
}

// pow(x, n) where n is constant
// f = x^n, f' = n * x^(n-1) * x'
template <class T>
[[nodiscard]] fwd_diff<T> pow(fwd_diff<T> const& x, T const& n)
{
    auto const xn_1 = std::pow(x.value, n - T(1));
    auto const xn = xn_1 * x.value;
    auto const df_dx = n * xn_1;
    return {xn, df_dx * x.d_value};
}

// pow(x, y) general case: x^y = exp(y * log(x))
// Let u = y * log(x), then f = exp(u), f' = exp(u) * u'
template <class T>
[[nodiscard]] fwd_diff<T> pow(fwd_diff<T> const& x, fwd_diff<T> const& y)
{
    auto const log_x = std::log(x.value);
    auto const u = y.value * log_x;
    auto const inv_x = T(1) / x.value;
    // u' = y' * log(x) + y * x' / x
    auto const du = y.d_value * log_x + y.value * x.d_value * inv_x;
    auto const exp_u = std::exp(u);
    return {exp_u, exp_u * du};
}

// log(x)' = x' / x
template <class T>
[[nodiscard]] fwd_diff<T> log(fwd_diff<T> const& x)
{
    auto const inv_x = T(1) / x.value;
    return {std::log(x.value), x.d_value * inv_x};
}

// log2(x) = log(x) / log(2)
template <class T>
[[nodiscard]] fwd_diff<T> log2(fwd_diff<T> const& x)
{
    auto const inv_log2 = T(1) / std::log(T(2));
    auto const inv_x = T(1) / x.value;
    return {std::log2(x.value), x.d_value * inv_x * inv_log2};
}

// log10(x) = log(x) / log(10)
template <class T>
[[nodiscard]] fwd_diff<T> log10(fwd_diff<T> const& x)
{
    auto const inv_log10 = T(1) / std::log(T(10));
    auto const inv_x = T(1) / x.value;
    return {std::log10(x.value), x.d_value * inv_x * inv_log10};
}

// exp(x)' = exp(x) * x'
template <class T>
[[nodiscard]] fwd_diff<T> exp(fwd_diff<T> const& x)
{
    auto const e = std::exp(x.value);
    return {e, e * x.d_value};
}

// sin(x)' = cos(x) * x'
// x is interpreted as radians.
template <class T>
[[nodiscard]] fwd_diff<T> sin(fwd_diff<T> const& x)
{
    auto const s = std::sin(x.value);
    auto const c = std::cos(x.value);
    return {s, c * x.d_value};
}

// cos(x)' = -sin(x) * x'
// x is interpreted as radians.
template <class T>
[[nodiscard]] fwd_diff<T> cos(fwd_diff<T> const& x)
{
    auto const s = std::sin(x.value);
    auto const c = std::cos(x.value);
    return {c, -s * x.d_value};
}

// tan(x)' = sec^2(x) * x' = (1 + tan^2(x)) * x'
// x is interpreted as radians.
template <class T>
[[nodiscard]] fwd_diff<T> tan(fwd_diff<T> const& x)
{
    auto const t = std::tan(x.value);
    auto const sec2 = T(1) + t * t;
    return {t, sec2 * x.d_value};
}

// asin(x)' = x' / sqrt(1 - x^2)
// Returns radians.
template <class T>
[[nodiscard]] fwd_diff<T> asin(fwd_diff<T> const& x)
{
    auto const one_minus_x2 = T(1) - x.value * x.value;
    auto const sqrt_1mx2 = std::sqrt(one_minus_x2);
    auto const inv_sqrt = T(1) / sqrt_1mx2;
    return {std::asin(x.value), x.d_value * inv_sqrt};
}

// acos(x)' = -x' / sqrt(1 - x^2)
// Returns radians.
template <class T>
[[nodiscard]] fwd_diff<T> acos(fwd_diff<T> const& x)
{
    auto const one_minus_x2 = T(1) - x.value * x.value;
    auto const sqrt_1mx2 = std::sqrt(one_minus_x2);
    auto const inv_sqrt = T(1) / sqrt_1mx2;
    return {std::acos(x.value), -x.d_value * inv_sqrt};
}

// atan(x)' = x' / (1 + x^2)
// Returns radians.
template <class T>
[[nodiscard]] fwd_diff<T> atan(fwd_diff<T> const& x)
{
    auto const one_plus_x2 = T(1) + x.value * x.value;
    auto const inv_1px2 = T(1) / one_plus_x2;
    return {std::atan(x.value), x.d_value * inv_1px2};
}

// atan2(y, x) = atan(y/x) with quadrant correction. Returns radians.
// d(atan2)/dx = -y / r^2,  d(atan2)/dy = x / r^2,  r^2 = x^2 + y^2
// atan2' = (x * y' - y * x') / r^2
template <class T>
[[nodiscard]] fwd_diff<T> atan2(fwd_diff<T> const& y, fwd_diff<T> const& x)
{
    auto const r2 = x.value * x.value + y.value * y.value;
    auto const inv_r2 = T(1) / r2;
    auto const num = x.value * y.d_value - y.value * x.d_value;
    return {std::atan2(y.value, x.value), num * inv_r2};
}

template <class T>
[[nodiscard]] fwd_diff<T> atan2(fwd_diff<T> const& y, T const& x)
{
    auto const r2 = x * x + y.value * y.value;
    auto const inv_r2 = T(1) / r2;
    return {std::atan2(y.value, x), x * y.d_value * inv_r2};
}

template <class T>
[[nodiscard]] fwd_diff<T> atan2(T const& y, fwd_diff<T> const& x)
{
    auto const r2 = x.value * x.value + y * y;
    auto const inv_r2 = T(1) / r2;
    return {std::atan2(y, x.value), -y * x.d_value * inv_r2};
}

// -- mixed vec3 ops (only those needed for now) --

template <class T>
[[nodiscard]] fwd_diff<T> dot(vec3<fwd_diff<T>> const& l, vec3<T> const& r)
{
    return l.x * r.x + l.y * r.y + l.z * r.z;
}

template <class T>
[[nodiscard]] vec3<fwd_diff<T>> operator*(vec3<T> const& l, fwd_diff<T> const& r)
{
    return {l.x * r, l.y * r, l.z * r};
}

template <class T>
[[nodiscard]] vec3<fwd_diff<T>> operator+(vec3<T> const& l, vec3<fwd_diff<T>> const& r)
{
    return {l.x + r.x, l.y + r.y, l.z + r.z};
}

template <class T>
[[nodiscard]] vec3<fwd_diff<T>> operator-(vec3<fwd_diff<T>> const& l, vec3<T> const& r)
{
    return {l.x - r.x, l.y - r.y, l.z - r.z};
}

// Specialized normalize for 3D vectors with fwd_diff scalars.
// This avoids propagating sqrt/div through autodiff and instead
// computes value and first derivative analytically in T.
template <class T>
[[nodiscard]] vec3<fwd_diff<T>> normalize_fd(vec3<fwd_diff<T>> const& v)
{
    // Extract values and derivatives of v(t)
    T const vx0 = v.x.value;
    T const vx1 = v.y.value;
    T const vx2 = v.z.value;

    T const dvx0 = v.x.d_value;
    T const dvx1 = v.y.d_value;
    T const dvx2 = v.z.d_value;

    // q = ||v||^2, and its first derivative
    // q  = v·v
    // q' = 2 * v·v'
    T const q = vx0 * vx0 + vx1 * vx1 + vx2 * vx2;
    T const q1 = T(2) * (vx0 * dvx0 + vx1 * dvx1 + vx2 * dvx2);

    // s = sqrt(q), s' via scalar chain rule
    // s' = q' / (2 * s)
    T const s = std::sqrt(q); // q > 0 is assumed (same as generic normalize)
    T const inv_s = T(1) / s;
    T const s1 = q1 * (T(0.5) * inv_s);

    // For each component: x_i = v_i / s
    // Use the same division rule as in fwd_diff:
    //   q  = a / b
    //   q' = (a' - q * b') / b
    auto make_component = [&](T vi, T dvi) -> fwd_diff<T>
    {
        T const xi = vi * inv_s;
        T const dxi = (dvi - xi * s1) * inv_s;
        return {xi, dxi};
    };

    return {make_component(vx0, dvx0), make_component(vx1, dvx1), make_component(vx2, dvx2)};
}

template <class T>
[[nodiscard]] inline std::string to_string(fwd_diff<T> const& x)
{
    return std::format("fd({}, {})", x.value, x.d_value);
}

namespace fd_detail
{
inline double value_scalar(double x)
{
    return x;
}

template <class S>
inline double value_scalar(fwd_diff<S> const& x)
{
    return static_cast<double>(x.value);
}
} // namespace fd_detail

} // namespace antipodal
