#pragma once

/**
 * @file intersector.hh
 * @brief Ray–triangle-mesh "signed intersection count" abstraction used by the
 *        integer and full GWN kernels.
 *
 * An @em Intersector is any type that satisfies the following duck-typed
 * concept (there is no base class — pass any conforming type by `const&`):
 *
 * @code
 * struct MyIntersector
 * {
 *     // For some scalar T (typically float or double):
 *     [[nodiscard]] int signed_intersection_count(vec3<T> p, vec3<T> dir) const;
 * };
 * @endcode
 *
 * Contract — `signed_intersection_count` returns
 * @f$ \sum_i \mathrm{sign}(\vec{N_i} \cdot \vec{d}) @f$
 * over every triangle @f$i@f$ of the underlying mesh whose ray
 * @f$(\vec{p}, \vec{d})@f$ at parameter @f$t \ge 0@f$ hits, where
 * @f$\vec{N_i}@f$ is the (unnormalized) geometric face normal. This is the
 * `SignedIntersectionNumber` term from the antipodal paper's Algorithm
 * `wnr-unorm`.
 *
 * Additional requirements:
 * - The query is `const` and must be safe to call concurrently from multiple
 *   threads (the batch kernels parallelize over query points, sharing one
 *   intersector across threads).
 * - `dir` need not be unit-length; only its sign relative to face normals
 *   matters.
 * - The intersector does not own the input vertex/index data; callers are
 *   responsible for keeping that storage alive for the intersector's lifetime.
 *   Implementations are free to build internal acceleration structures.
 *
 * Two implementations ship with the library:
 * - `NaiveIntersector<T>` (this header) — always available; linear scan with
 *   Möller–Trumbore.
 * - `EmbreeIntersector` (`intersector_embree.hh`) — gated on
 *   `ANTIPODAL_HAS_EMBREE`; float-only; delegates traversal to Embree 4 with an
 *   occluded-ray filter callback.
 */

#include <cassert>
#include <cstddef>
#include <span>

#include "../common.hh"

namespace antipodal
{
// Linear scan over a triangle list. Holds non-owning spans into caller memory;
// the spans must remain valid for the lifetime of any query call.
template <class T>
struct NaiveIntersector
{
    std::span<vec3<T> const> vertices;
    std::span<int const> indices; // triangle list, length must be a multiple of 3

    [[nodiscard]] int signed_intersection_count(vec3<T> p, vec3<T> dir) const
    {
        assert(indices.size() % 3 == 0);
        int sum = 0;
        auto const n_tris = indices.size() / 3;
        for (std::size_t t = 0; t < n_tris; ++t)
        {
            auto const& a = vertices[indices[3 * t + 0]];
            auto const& b = vertices[indices[3 * t + 1]];
            auto const& c = vertices[indices[3 * t + 2]];

            // Möller-Trumbore against (p, dir).
            auto const e1 = b - a;
            auto const e2 = c - a;
            auto const h = cross(dir, e2);
            auto const det = dot(e1, h);
            if (det == T(0))
                continue; // ray parallel to triangle plane

            auto const inv_det = T(1) / det;
            auto const s = p - a;
            auto const u = inv_det * dot(s, h);
            if (u < T(0) || u > T(1))
                continue;

            auto const q = cross(s, e1);
            auto const v = inv_det * dot(dir, q);
            if (v < T(0) || u + v > T(1))
                continue;

            auto const t_hit = inv_det * dot(e2, q);
            if (t_hit < T(0))
                continue;

            // Geometric (unnormalized) normal — only its sign vs. dir matters.
            auto const ng = cross(e1, e2);
            sum += dot(ng, dir) > T(0) ? +1 : -1;
        }
        return sum;
    }
};
} // namespace antipodal
