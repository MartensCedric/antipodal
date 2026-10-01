#pragma once

// Exact predicates for the unconditionally-robust mesh GWN (see gwn_mesh_robust.hh).
//
// We specialize on a ray pointing along a fixed coordinate axis (+x).
// What matters is that one 2x2 determinant has an exact sign; the precision policies below differ in how.
// int64_grid and int128_grid quantize the query and all geometry to integer coordinates (see the quantizer in
// gwn_mesh_robust.hh), coarse enough that the determinant is exact in 64- or 128-bit integers.
// exact_float keeps the doubles and computes the determinant's sign exactly, with an adaptive fallback.
//
// Convention (vec3 fields):
//   .x    = depth, along the +x ray direction
//   .y, .z = the 2D projection plane the ray is cast onto
//
// Symbolic perturbation of the query q_eps = q + (eps1 on x, eps2 on y, eps3 on z) with 0 < eps3 << eps2 << eps1.
// eps1 (depth) only matters for the front/back test, where a hit exactly at the query's depth counts as in front.
// The in-plane / atan2-numerator discontinuity is governed exactly by the 2x2 determinant of the projected edge.
// It is evaluated with the lexicographic (eps2, eps3) tie-break below.
//
// Key identity that makes the integer part and the fractional part couple exactly:
//
//   t_real(q; q0, q1) = (q.y - q0.y)*(q0.z - q1.z) + (q.z - q0.z)*(q1.y - q0.y)
//   num   (q; q0, q1) = (q0.z - q.z)*(q1.y - q.y) - (q0.y - q.y)*(q1.z - q.z)
//                     = -t_real
//
// So the edge-classification predicate and the atan2 numerator are the same determinant up to sign.
// (The edge predicate is "which side of the projected edge the query is on", used by the integer test.)
// Both share the discontinuity.
// As long as the sign is evaluated by the same exact code on both sides, the integer jump and the atan2 jump cancel.

#include <antipodal/math/common.hh>

#include <algorithm>
#include <cmath>
#include <compare>
#include <cstdint>
#include <limits>
#include <type_traits>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
#include <intrin.h>
#endif

namespace antipodal::robust
{
using i64 = std::int64_t;

// Minimal signed 128-bit integer for compilers without __int128 (MSVC).
// Only supports what the predicates need.
struct int128
{
    std::uint64_t lo = 0;
    std::int64_t hi = 0;

    constexpr int128() = default;
    constexpr int128(std::int64_t v) : lo(std::uint64_t(v)), hi(v < 0 ? -1 : 0) {}

    // Exact 64x64 -> 128-bit product.
    // On MSVC this is the hardware multiply, which the speed of the 128-bit mode rests on.
    [[nodiscard]] static constexpr int128 mul(std::int64_t a, std::int64_t b)
    {
#if defined(_MSC_VER) && defined(_M_X64)
        if (!std::is_constant_evaluated())
        {
            std::int64_t hi;
            std::uint64_t const lo = std::uint64_t(_mul128(a, b, &hi));
            return from_parts(lo, hi);
        }
#elif defined(_MSC_VER) && defined(_M_ARM64)
        if (!std::is_constant_evaluated())
            return from_parts(std::uint64_t(a) * std::uint64_t(b), __mulh(a, b));
#endif
        return mul_portable(a, b);
    }

    // Schoolbook multiply on 32-bit halves; public so tests can check mul() against it on every compiler.
    [[nodiscard]] static constexpr int128 mul_portable(std::int64_t a, std::int64_t b)
    {
        bool const neg = (a < 0) != (b < 0);
        // works for INT64_MIN too
        auto const mag = [](std::int64_t v) { return v < 0 ? std::uint64_t(0) - std::uint64_t(v) : std::uint64_t(v); };
        std::uint64_t const ua = mag(a);
        std::uint64_t const ub = mag(b);

        // schoolbook multiply on 32-bit halves
        std::uint64_t const a0 = ua & 0xFFFFFFFFu, a1 = ua >> 32;
        std::uint64_t const b0 = ub & 0xFFFFFFFFu, b1 = ub >> 32;
        std::uint64_t const p00 = a0 * b0;
        std::uint64_t const p01 = a0 * b1;
        std::uint64_t const p10 = a1 * b0;
        std::uint64_t const p11 = a1 * b1;
        std::uint64_t const mid = (p00 >> 32) + (p01 & 0xFFFFFFFFu) + (p10 & 0xFFFFFFFFu);

        int128 r;
        r.lo = (mid << 32) | (p00 & 0xFFFFFFFFu);
        r.hi = std::int64_t(p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32));
        return neg ? -r : r;
    }

    [[nodiscard]] static constexpr int128 from_parts(std::uint64_t lo, std::int64_t hi)
    {
        int128 r;
        r.lo = lo;
        r.hi = hi;
        return r;
    }

    [[nodiscard]] friend constexpr int128 operator+(int128 a, int128 b)
    {
        int128 r;
        r.lo = a.lo + b.lo;
        std::uint64_t const carry = r.lo < a.lo ? 1 : 0;
        r.hi = std::int64_t(std::uint64_t(a.hi) + std::uint64_t(b.hi) + carry);
        return r;
    }

    [[nodiscard]] friend constexpr int128 operator-(int128 a)
    {
        int128 r;
        r.lo = ~a.lo + 1;
        r.hi = std::int64_t(~std::uint64_t(a.hi) + (r.lo == 0 ? 1 : 0));
        return r;
    }

    [[nodiscard]] friend constexpr int128 operator-(int128 a, int128 b) { return a + (-b); }

    [[nodiscard]] friend constexpr bool operator==(int128 a, int128 b) = default;

    [[nodiscard]] friend constexpr std::strong_ordering operator<=>(int128 a, int128 b)
    {
        if (a.hi != b.hi)
            return a.hi <=> b.hi;
        return a.lo <=> b.lo;
    }

    // Convert the magnitude so small negative values keep their sign.
    [[nodiscard]] explicit constexpr operator double() const
    {
        bool const neg = hi < 0;
        int128 const m = neg ? -*this : *this;
        double const d = double(std::uint64_t(m.hi)) * 18446744073709551616.0 + double(m.lo);
        return neg ? -d : d;
    }
};

// Define ANTIPODAL_ROBUST_PORTABLE_INT128 to use the portable type even when __int128 exists.
// clang-cl has __int128 but may not link the runtime helpers it needs, so it gets the portable type too.
#if defined(__SIZEOF_INT128__) && !defined(_MSC_VER) && !defined(ANTIPODAL_ROBUST_PORTABLE_INT128)
__extension__ typedef __int128 i128; // __extension__ silences -Wpedantic
#else
using i128 = int128;
#endif

// Quantized integer point of int64_grid.
// Mesh coordinates are kept to 20 bits and queries are clamped to 30 bits (see RobustMeshGwn::quantize),
// so a determinant term stays below 2^61 and never overflows i64.
using ivec3 = vec3<std::int32_t>;

// Same for int128_grid: 50-bit mesh coordinates, queries clamped to 60 bits, terms below 2^121 in i128.
using lvec3 = vec3<std::int64_t>;

// Type of edge_det for C coordinates: an integer wide enough to be exact, or a double whose sign is exact.
template <class C>
using wide_t = std::conditional_t<std::is_floating_point_v<C>, double, std::conditional_t<sizeof(C) == 4, i64, i128>>;

template <class C>
[[nodiscard]] constexpr wide_t<C> mul_wide(C a, C b)
{
    static_assert(std::is_same_v<C, std::int32_t> || std::is_same_v<C, std::int64_t>, "robust predicates support int32 "
                                                                                      "or int64 coordinates");
    if constexpr (std::is_same_v<wide_t<C>, int128>)
        return int128::mul(a, b);
    else
        return wide_t<C>(a) * wide_t<C>(b);
}

// --- precision policies ---------------------------------------------------------------------------------------------
//
// The second template argument of RobustMeshGwn: how the shared determinant is made exact.
// Each cost is per query relative to int64_grid, measured on MSVC x64 (16k-triangle closed sphere, 8k-triangle open
// hemisphere, 20k random queries).

// 20-bit grid, 64-bit integer determinants.
// Accurate to about 1e-6 of the mesh size: a query within a few grid cells of the surface can snap across it.
// Queries further than about 500 mesh sizes are clamped to that distance, which costs up to ~1.5e-7 absolute.
// The fastest; the default for float.
struct int64_grid
{
    using coord = std::int32_t;
    static constexpr int grid_bits = 20;
};

// 50-bit grid, 128-bit integer determinants.
// About double precision within ~500 mesh sizes; beyond, queries are clamped as for int64_grid.
// 50 bits already matches double precision, so a finer grid would not help.
// Costs about 1.8-1.9x int64_grid.
struct int128_grid
{
    using coord = std::int64_t;
    static constexpr int grid_bits = 50;
};

// No grid: coordinates stay doubles, and each determinant's sign is computed exactly.
// A double evaluation with a forward error bound decides almost every sign; only near-degenerate configurations fall
// back to an exact expansion (Shewchuk's orient2d scheme).
// Full double precision at any distance.
// Coordinates must stay well inside the double range (magnitudes below ~1e150 and not subnormal-small), so products
// neither overflow nor underflow; Embree's traversal threads run with flush-to-zero on.
// Costs about 1.4-2x int64_grid, about the same as int128_grid, with no far-field limit; the default for double.
struct exact_float
{
    using coord = double;
};

template <class P>
inline constexpr bool is_grid_policy = !std::is_floating_point_v<typename P::coord>;

template <class T>
using default_precision = std::conditional_t<std::is_same_v<T, float>, int64_grid, exact_float>;

// --- exact sign of a double determinant -----------------------------------------------------------------------------
namespace detail
{
// x + y == a + b exactly (Knuth's two-sum; needs IEEE round-to-nearest and no fast-math reassociation).
inline void two_sum(double a, double b, double& x, double& y)
{
    x = a + b;
    double const bv = x - a;
    double const av = x - bv;
    y = (a - av) + (b - bv);
}

// Exact sign of the sum of the six products a[i] * b[i].
// Each product splits exactly into p + e with fma, and the twelve parts are summed as a nonoverlapping expansion.
// The sign of such an expansion is the sign of its largest component, which is its last nonzero one.
inline int exact_sign_of_products(double const (&a)[6], double const (&b)[6])
{
    double e[12];
    int n = 0;
    auto const grow = [&](double v)
    {
        for (int i = 0; i < n; ++i)
        {
            double sum;
            double err;
            two_sum(v, e[i], sum, err);
            e[i] = err;
            v = sum;
        }
        e[n++] = v;
    };
    for (int i = 0; i < 6; ++i)
    {
        double const p = a[i] * b[i];
        grow(std::fma(a[i], b[i], -p));
        grow(p);
    }
    for (int i = n - 1; i >= 0; --i)
        if (e[i] != 0)
            return e[i] > 0 ? 1 : -1;
    return 0;
}
} // namespace detail

// The 2x2 determinant of (q - q0) and (q1 - q0) in the (y,z) plane.
// Its sign says which side of the directed projected edge (q0 -> q1) the query lies on; 0 means on the edge's line.
// Integer coordinates: the exact value.
// Double coordinates: the sign is exact, and the magnitude is the double estimate, kept at least DBL_MIN when the exact
// value is not 0, so that `!= 0` and `> 0` read the exact sign.
template <class C>
[[nodiscard]] inline wide_t<C> edge_det(vec3<C> q, vec3<C> q0, vec3<C> q1)
{
    if constexpr (std::is_floating_point_v<C>)
    {
        static_assert(std::is_same_v<C, double>, "exact_float works on double coordinates");
        double const p1 = (q.y - q0.y) * (q0.z - q1.z);
        double const p2 = (q.z - q0.z) * (q1.y - q0.y);
        double const det = p1 + p2;

        // Shewchuk's orient2d error bound for this form
        constexpr double eps = std::numeric_limits<double>::epsilon() / 2;
        constexpr double err_bound = (3.0 + 16.0 * eps) * eps;
        double const magnitude = std::abs(p1) + std::abs(p2);
        if (std::abs(det) > err_bound * magnitude)
            return det;

        // Both products are exactly 0 only if a difference is: a rounded difference is 0 only for equal inputs,
        // and a product of nonzero doubles only underflows outside the precondition.
        // A degenerate edge lands here on every call, so it must not reach the expansion.
        if (magnitude == 0)
            return 0.0;

        // expanded into raw coordinates, so no rounded difference is involved:
        // det = qy*q0z - qy*q1z + q0y*q1z + qz*q1y - qz*q0y - q0z*q1y
        double const a[6] = {q.y, -q.y, q0.y, q.z, -q.z, -q0.z};
        double const b[6] = {q0.z, q1.z, q1.z, q1.y, q0.y, q1.y};
        int const sign = detail::exact_sign_of_products(a, b);
        if (sign == 0)
            return 0.0;
        return std::copysign(std::max(std::abs(det), std::numeric_limits<double>::min()), double(sign));
    }
    else
    {
        return mul_wide<C>(q.y - q0.y, q0.z - q1.z) + mul_wide<C>(q.z - q0.z, q1.y - q0.y);
    }
}

// Signed side of the directed projected edge (q0 -> q1) that the query q lies on.
// Returns +1 / -1 via the exact real determinant.
// On the exact-zero (grazing) case the lexicographic perturbation (eps2 on y dominating eps3 on z) decides.
// Returns 0 only for a degenerate edge that projects to a single point (cannot be hit / contributes nothing).
template <class C>
[[nodiscard]] inline int edge_sign(vec3<C> q, vec3<C> q0, vec3<C> q1)
{
    wide_t<C> const t_real = edge_det(q, q0, q1);
    if (t_real != 0)
        return t_real > 0 ? 1 : -1;

    // eps2 coefficient (perturb q.y): (q0.z - q1.z)
    if (q0.z != q1.z)
        return q0.z > q1.z ? 1 : -1;

    // eps3 coefficient (perturb q.z): (q1.y - q0.y)
    if (q1.y != q0.y)
        return q1.y > q0.y ? 1 : -1;

    // degenerate edge (projects to a point)
    return 0;
}

// Numerator of the spherical-area atan2 for the edge (q0 -> q1) seen from query q.
// It equals -t_real, and its sign is exact: exactly 0 iff t_real is.
// Use sign_num_perturbed() to resolve the num==0 grazing case consistently.
// For double coordinates the magnitude is computed from the query-relative vectors, not from edge_det:
// as the query approaches the ray through a vertex, that vertex's (y,z) offset gets small, and only differences taken
// from the query keep it to full relative precision; the neighbouring edges' atan2 terms then still cancel.
template <class C>
[[nodiscard]] inline wide_t<C> atan2_num(vec3<C> q, vec3<C> q0, vec3<C> q1)
{
    auto const v0 = q0 - q;
    auto const v1 = q1 - q;
    if constexpr (std::is_floating_point_v<C>)
    {
        double const t_real = edge_det(q, q0, q1);
        if (t_real == 0)
            return 0.0;
        double const num = v0.z * v1.y - v0.y * v1.z;
        return std::copysign(std::max(std::abs(num), std::numeric_limits<double>::min()), -t_real);
    }
    else
    {
        return mul_wide<C>(v0.z, v1.y) - mul_wide<C>(v0.y, v1.z);
    }
}

// The perturbed sign of atan2_num when it is exactly zero.
// Because num == -t_real, the perturbed sign of num is exactly the negation of edge_sign().
// This is the single fact that ties the atan2 quadrant choice to the ray-triangle predicate.
template <class C>
[[nodiscard]] inline int sign_num_perturbed(vec3<C> q, vec3<C> q0, vec3<C> q1)
{
    return -edge_sign(q, q0, q1);
}

// a.y * b.y + a.z * b.z; exact for integer coordinates up to the final conversion to double.
template <class C>
[[nodiscard]] inline double dot_yz(vec3<C> a, vec3<C> b)
{
    if constexpr (std::is_floating_point_v<C>)
        return a.y * b.y + a.z * b.z;
    else
        return double(mul_wide<C>(a.y, b.y) + mul_wide<C>(a.z, b.z));
}

// Where the +x ray from q meets the projected triangle (q0,q1,q2), if it does.
// sign is the common edge sign (+1/-1 = orientation of the projected triangle) when q projects strictly or
// symbolically inside, and 0 otherwise (outside, or a degenerate projection).
// w0, w1, w2 are then the unnormalized barycentric weights of q0, q1, q2: each has the same sign as `sign` or is 0,
// and their sum is twice the projected area, which is not 0.
struct tri_projection
{
    int sign = 0;
    double w0 = 0.0;
    double w1 = 0.0;
    double w2 = 0.0;
};

template <class C>
[[nodiscard]] inline tri_projection project_into_tri(vec3<C> q, vec3<C> q0, vec3<C> q1, vec3<C> q2)
{
    int const e01 = edge_sign(q, q0, q1);
    int const e12 = edge_sign(q, q1, q2);
    int const e20 = edge_sign(q, q2, q0);

    if (e01 == 0 || e01 != e12 || e01 != e20)
        return {};

    // the weight of a vertex is the determinant of the edge opposite it
    return {e01, double(edge_det(q, q1, q2)), double(edge_det(q, q2, q0)), double(edge_det(q, q0, q1))};
}

// In-projected-triangle test for the +x ray from q against quantized triangle (q0,q1,q2).
// Returns project_into_tri(...).sign.
// This is the signed contribution the integer part adds if the hit is in front (positive depth).
template <class C>
[[nodiscard]] inline int inside_tri_sign(vec3<C> q, vec3<C> q0, vec3<C> q1, vec3<C> q2)
{
    return project_into_tri(q, q0, q1, q2).sign;
}
} // namespace antipodal::robust
