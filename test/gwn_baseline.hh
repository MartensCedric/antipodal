#pragma once

#include <antipodal/common.hh>

#include <cmath>
#include <cstddef>
#include <span>

namespace antipodal_test
{
// Naive baseline GWN for an indexed triangle mesh at a single query point:
// no acceleration structure, just the raw per-triangle solid angle.
//
// For each triangle, project the three vertices onto the unit sphere
// centered at `p`, sum the signed solid angles via the Van Oosterom-Strackee
// formula, and divide by 4*pi.
//
// `indices.size()` must be a multiple of 3. Behavior is undefined if `p`
// coincides with a triangle vertex (the projection is singular there).
template <class T>
[[nodiscard]] T eval_gwn_baseline(antipodal::vec3<T> p,
                                  std::span<antipodal::vec3<T> const> vertices,
                                  std::span<int const> indices)
{
    using namespace antipodal;
    auto const tri_count = indices.size() / 3;
    T solid_angle = T(0);
    for (std::size_t i = 0; i < tri_count; ++i)
    {
        auto const v0 = vertices[indices[3 * i + 0]];
        auto const v1 = vertices[indices[3 * i + 1]];
        auto const v2 = vertices[indices[3 * i + 2]];

        auto const a = normalize(v0 - p);
        auto const b = normalize(v1 - p);
        auto const c = normalize(v2 - p);

        auto const num = dot(a, cross(b, c));
        auto const denom = T(1) + dot(a, b) + dot(b, c) + dot(c, a);
        solid_angle += T(2) * std::atan2(num, denom);
    }
    return solid_angle / (T(4) * antipodal::pi<T>);
}
} // namespace antipodal_test
