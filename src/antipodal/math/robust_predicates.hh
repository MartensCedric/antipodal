#pragma once

// Exact integer predicates for the unconditionally-robust mesh GWN (see gwn_mesh_robust.hh).
//
// We specialize on a ray pointing along a fixed coordinate axis (+x).
// The query and all geometry are quantized to integer coordinates (see the quantizer in gwn_mesh_robust.hh).
// That keeps the relevant 2x2 determinant inside 64-bit integer arithmetic.
// (The paper's 128-bit option is deliberately out of scope here.)
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

#include <cstdint>

#include <antipodal/math/common.hh>

namespace antipodal::robust
{
using i64 = std::int64_t;

// Quantized integer point.
// Coordinates are kept to ~20 bits by the quantizer, so the 2x2 determinants below never overflow i64.
using ivec3 = vec3<std::int32_t>;

// Signed side of the directed projected edge (q0 -> q1) that the query q lies on.
// Returns +1 / -1 via the exact real determinant.
// On the exact-zero (grazing) case the lexicographic perturbation (eps2 on y dominating eps3 on z) decides.
// Returns 0 only for a degenerate edge that projects to a single point (cannot be hit / contributes nothing).
[[nodiscard]] inline int edge_sign(ivec3 q, ivec3 q0, ivec3 q1)
{
    // real part: 2x2 determinant of (q - q0) and (q1 - q0) in the (y,z) plane
    i64 const t_real = i64(q.y - q0.y) * i64(q0.z - q1.z) //
                       + i64(q.z - q0.z) * i64(q1.y - q0.y);
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
// Equals -t_real, so it can be large; returned as i64.
// Use sign_num_perturbed() to resolve the num==0 grazing case consistently.
[[nodiscard]] inline i64 atan2_num(ivec3 q, ivec3 q0, ivec3 q1)
{
    auto const v0 = q0 - q;
    auto const v1 = q1 - q;
    return i64(v0.z) * i64(v1.y) - i64(v0.y) * i64(v1.z);
}

// The perturbed sign of atan2_num when it is exactly zero.
// Because num == -t_real, the perturbed sign of num is exactly the negation of edge_sign().
// This is the single fact that ties the atan2 quadrant choice to the ray-triangle predicate.
[[nodiscard]] inline int sign_num_perturbed(ivec3 q, ivec3 q0, ivec3 q1)
{
    return -edge_sign(q, q0, q1);
}

// In-projected-triangle test for the +x ray from q against quantized triangle (q0,q1,q2).
// Returns the common edge sign (+1/-1 = sign of the projected signed area = orientation) when q projects inside.
// "inside" here means strictly or symbolically inside; otherwise returns 0 (outside or degenerate).
// This is the signed contribution the integer part adds if the hit is in front (positive depth).
[[nodiscard]] inline int inside_tri_sign(ivec3 q, ivec3 q0, ivec3 q1, ivec3 q2)
{
    int const e01 = edge_sign(q, q0, q1);
    int const e12 = edge_sign(q, q1, q2);
    int const e20 = edge_sign(q, q2, q0);

    if (e01 != 0 && e01 == e12 && e01 == e20)
        return e01;

    return 0;
}
} // namespace antipodal::robust
