#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <span>
#include <unordered_map>
#include <vector>

namespace antipodal
{
template <class T>
struct vec2
{
    T x{};
    T y{};

    [[nodiscard]] friend constexpr vec2 operator+(vec2 a, vec2 b) { return {a.x + b.x, a.y + b.y}; }
    [[nodiscard]] friend constexpr vec2 operator-(vec2 a, vec2 b) { return {a.x - b.x, a.y - b.y}; }
    [[nodiscard]] friend constexpr vec2 operator-(vec2 a) { return {-a.x, -a.y}; }
    [[nodiscard]] friend constexpr vec2 operator*(vec2 a, T s) { return {a.x * s, a.y * s}; }
    [[nodiscard]] friend constexpr vec2 operator*(T s, vec2 a) { return a * s; }
    [[nodiscard]] friend constexpr vec2 operator/(vec2 a, T s) { return {a.x / s, a.y / s}; }
};

using fvec2 = vec2<float>;
using dvec2 = vec2<double>;

template <class T>
struct vec3
{
    T x{};
    T y{};
    T z{};

    [[nodiscard]] friend constexpr vec3 operator+(vec3 a, vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
    [[nodiscard]] friend constexpr vec3 operator-(vec3 a, vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
    [[nodiscard]] friend constexpr vec3 operator-(vec3 a) { return {-a.x, -a.y, -a.z}; }
    [[nodiscard]] friend constexpr vec3 operator*(vec3 a, T s) { return {a.x * s, a.y * s, a.z * s}; }
    [[nodiscard]] friend constexpr vec3 operator*(T s, vec3 a) { return a * s; }
    [[nodiscard]] friend constexpr vec3 operator/(vec3 a, T s) { return {a.x / s, a.y / s, a.z / s}; }
};

using fvec3 = vec3<float>;
using dvec3 = vec3<double>;

template <class T>
struct segment3
{
    vec3<T> pos0;
    vec3<T> pos1;
};

using fsegment3 = segment3<float>;
using dsegment3 = segment3<double>;

template <class T>
struct weighted_segment3
{
    segment3<T> segment;
    T weight;
};

using weighted_fsegment3 = weighted_segment3<float>;
using weighted_dsegment3 = weighted_segment3<double>;

template <class T>
inline constexpr T pi = std::numbers::pi_v<T>;

template <class T>
[[nodiscard]] constexpr T dot(vec3<T> a, vec3<T> b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
template <class T>
[[nodiscard]] constexpr vec3<T> cross(vec3<T> a, vec3<T> b)
{
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}
template <class T>
[[nodiscard]] constexpr T length_sqr(vec3<T> a)
{
    return dot(a, a);
}
template <class T>
[[nodiscard]] T length(vec3<T> a)
{
    return std::sqrt(length_sqr(a));
}
template <class T>
[[nodiscard]] vec3<T> normalize(vec3<T> a)
{
    return a / length(a);
}
// Returns the zero vector if `a` is too short to safely normalize.
template <class T>
[[nodiscard]] vec3<T> normalize_safe(vec3<T> a)
{
    auto const l2 = length_sqr(a);
    if (l2 <= T{})
        return {};
    return a / std::sqrt(l2);
}

// Convenience: extract the open boundary of a triangle mesh as a list of
// weighted, oriented segments suitable for `eval_gwnr_mesh_single` & friends.
//
// Each undirected edge {min, max} accumulates +1 for every triangle that
// traverses it as min→max and -1 for every traversal as max→min. Interior
// edges of an oriented manifold cancel to zero and are dropped. Surviving
// edges are emitted with positive weight; the segment direction is flipped
// when the net count is negative so the weight is always > 0.
//
// Non-manifold meshes (≥3 incident triangles per edge) are handled the same
// way: the surviving net count becomes the weight.
template <class T>
[[nodiscard]] std::vector<weighted_segment3<T>> build_boundary_segments(std::span<vec3<T> const> vertices,
                                                                        std::span<int const> indices)
{
    assert(indices.size() % 3 == 0);

    struct edge_key
    {
        int lo;
        int hi;
        [[nodiscard]] bool operator==(edge_key const&) const = default;
    };
    struct edge_hash
    {
        [[nodiscard]] std::size_t operator()(edge_key const& k) const noexcept
        {
            auto const packed = (std::uint64_t(std::uint32_t(k.lo)) << 32) | std::uint32_t(k.hi);
            return std::hash<std::uint64_t>{}(packed);
        }
    };

    std::unordered_map<edge_key, int, edge_hash> counts;
    counts.reserve(indices.size());

    auto const n_tris = indices.size() / 3;
    for (std::size_t t = 0; t < n_tris; ++t)
    {
        int const tri[3] = {indices[3 * t + 0], indices[3 * t + 1], indices[3 * t + 2]};
        for (int e = 0; e < 3; ++e)
        {
            int const u = tri[e];
            int const v = tri[(e + 1) % 3];
            edge_key const k{std::min(u, v), std::max(u, v)};
            counts[k] += (u < v) ? +1 : -1;
        }
    }

    std::vector<weighted_segment3<T>> result;
    result.reserve(counts.size());
    for (auto const& [k, sum] : counts)
    {
        if (sum == 0)
            continue;
        auto const& a = vertices[k.lo];
        auto const& b = vertices[k.hi];
        if (sum > 0)
            result.push_back({{a, b}, T(sum)});
        else
            result.push_back({{b, a}, T(-sum)});
    }
    return result;
}

// Half the signed solid-angle subtended by the spherical triangle (x1, v0, v1)
// at the origin, using the unnormalized Van Oosterom-Strackee formulation.
// `v0` and `v1` need NOT be unit vectors; `x1` is expected to be a unit vector
// (the antipodal "south pole" direction, i.e. -x0 in the paper notation).
// Callers accumulate per-edge contributions and divide by 2*pi.
template <class T>
[[nodiscard]] T signed_spherical_tri_area_half_unorm(vec3<T> x1, vec3<T> v0, vec3<T> v1)
{
    auto const l0 = length(v0);
    auto const l1 = length(v1);
    auto const num = dot(x1, cross(v0, v1));
    auto const denom = l0 * l1 + l1 * dot(x1, v0) + l0 * dot(x1, v1) + dot(v0, v1);
    return std::atan2(num, denom);
}
} // namespace antipodal
