#pragma once

// Exact integer predicates for the unconditionally-robust mesh GWN (see gwn_mesh_robust.hh).
//
// We specialize on a ray pointing along a fixed coordinate axis (+x).
// The query and all geometry are quantized to integer coordinates (see the quantizer in gwn_mesh_robust.hh).
// That keeps the relevant 2x2 determinant exact in 64-bit integers, or in 128-bit integers for the 128-bit option.
//
// Convention (ivec3 fields):
//   .x    = depth, along the +x ray direction
//   .y, .z = the 2D projection plane the ray is cast onto
//
// Symbolic perturbation of the query q_eps = q + (eps1 on x, eps2 on y, eps3 on z) with 0 < eps3 << eps2 << eps1.
// eps1 (depth) only matters for the front/back (supporting-plane) test, and is handled in double there.
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
// As long as the sign is evaluated by the same integer code on both sides, the integer jump and the atan2 jump cancel.

#include <antipodal/math/common.hh>

#include <compare>
#include <cstdint>
#include <type_traits>

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

    [[nodiscard]] static constexpr int128 mul(std::int64_t a, std::int64_t b)
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

// Quantized integer point.
// Coordinates are kept to ~20 bits by the quantizer, so the 2x2 determinants below never overflow i64.
using ivec3 = vec3<std::int32_t>;

// Same for the 128-bit option, with 50-bit coordinates and i128 determinants.
using lvec3 = vec3<std::int64_t>;

// Integer wide enough for an exact 2x2 determinant of C coordinates.
template <class C>
using wide_t = std::conditional_t<sizeof(C) == 4, i64, i128>;

template <class C>
[[nodiscard]] constexpr wide_t<C> mul_wide(C a, C b)
{
    static_assert(std::is_same_v<C, std::int32_t> || std::is_same_v<C, std::int64_t>,
                  "robust predicates support int32 or int64 coordinates");
    if constexpr (std::is_same_v<wide_t<C>, int128>)
        return int128::mul(a, b);
    else
        return wide_t<C>(a) * wide_t<C>(b);
}

// Quantization grid used by RobustMeshGwn<T, Bits>.
// Queries further than about 500 mesh sizes away are clamped to that distance before rounding
// (see RobustMeshGwn::quantize), so the determinants cannot overflow.
// 50 bits already matches double precision, so a finer grid would not help.
template <int Bits>
struct precision
{
    static_assert(Bits == 64 || Bits == 128, "robust precision must be 64 or 128 bits");
};

template <>
struct precision<64>
{
    using coord = std::int32_t;
    static constexpr int grid_bits = 20;
};

template <>
struct precision<128>
{
    using coord = std::int64_t;
    static constexpr int grid_bits = 50;
};

// Signed side of the directed projected edge (q0 -> q1) that the query q lies on.
// Returns +1 / -1 via the exact real determinant.
// On the exact-zero (grazing) case the lexicographic perturbation (eps2 on y dominating eps3 on z) decides.
// Returns 0 only for a degenerate edge that projects to a single point (cannot be hit / contributes nothing).
template <class C>
[[nodiscard]] inline int edge_sign(vec3<C> q, vec3<C> q0, vec3<C> q1)
{
    // real part: 2x2 determinant of (q - q0) and (q1 - q0) in the (y,z) plane
    wide_t<C> const t_real = mul_wide<C>(q.y - q0.y, q0.z - q1.z) //
                           + mul_wide<C>(q.z - q0.z, q1.y - q0.y);
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

// Exact (perturbed) numerator value of the spherical-area atan2 for the edge (q0 -> q1) seen from query q.
// Equals -t_real, so it can be large; returned as wide_t<C>.
// Use sign_num_perturbed() to resolve the num==0 grazing case consistently.
template <class C>
[[nodiscard]] inline wide_t<C> atan2_num(vec3<C> q, vec3<C> q0, vec3<C> q1)
{
    auto const v0 = q0 - q;
    auto const v1 = q1 - q;
    return mul_wide<C>(v0.z, v1.y) - mul_wide<C>(v0.y, v1.z);
}

// The perturbed sign of atan2_num when it is exactly zero.
// Because num == -t_real, the perturbed sign of num is exactly the negation of edge_sign().
// This is the single fact that ties the atan2 quadrant choice to the ray-triangle predicate.
template <class C>
[[nodiscard]] inline int sign_num_perturbed(vec3<C> q, vec3<C> q0, vec3<C> q1)
{
    return -edge_sign(q, q0, q1);
}

// In-projected-triangle test for the +x ray from q against quantized triangle (q0,q1,q2).
// Returns the common edge sign (+1/-1 = sign of the projected signed area = orientation) when q projects inside.
// "inside" here means strictly or symbolically inside; otherwise returns 0 (outside or degenerate).
// This is the signed contribution the integer part adds if the hit is in front (positive depth).
template <class C>
[[nodiscard]] inline int inside_tri_sign(vec3<C> q, vec3<C> q0, vec3<C> q1, vec3<C> q2)
{
    int const e01 = edge_sign(q, q0, q1);
    int const e12 = edge_sign(q, q1, q2);
    int const e20 = edge_sign(q, q2, q0);

    if (e01 != 0 && e01 == e12 && e01 == e20)
        return e01;

    return 0;
}
} // namespace antipodal::robust
