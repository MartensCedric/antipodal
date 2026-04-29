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
 * - `NaiveIntersector<T>` — always available; linear scan with Möller–Trumbore.
 * - `EmbreeIntersector` — gated on `ANTIPODAL_HAS_EMBREE`; float-only;
 *   delegates traversal to Embree 4 with an occluded-ray filter callback.
 */

#include <cassert>
#include <cstddef>
#include <limits>
#include <span>

#include "common.hh"

#if defined(ANTIPODAL_HAS_EMBREE) && ANTIPODAL_HAS_EMBREE
#include <pmmintrin.h>
#include <xmmintrin.h>

#include <embree4/rtcore.h>
#endif

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

#if defined(ANTIPODAL_HAS_EMBREE) && ANTIPODAL_HAS_EMBREE
// Embree-accelerated intersector (float-only — Embree's geometry API is float).
//
// Constructed from non-owning vertex/index spans via rtcSetSharedGeometryBuffer:
// the caller's vertex and index data MUST remain valid for the full lifetime
// of the EmbreeIntersector (Embree reads from those buffers during traversal).
struct EmbreeIntersector
{
    EmbreeIntersector(std::span<fvec3 const> vertices, std::span<int const> indices)
    {
        assert(indices.size() % 3 == 0);

        m_device = rtcNewDevice(nullptr);
        m_scene = rtcNewScene(m_device);

        auto geom = rtcNewGeometry(m_device, RTC_GEOMETRY_TYPE_TRIANGLE);

        // Shared buffers: zero-copy, caller keeps the data alive.
        rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
                                   vertices.data(), 0, sizeof(fvec3), vertices.size());
        rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
                                   indices.data(), 0, 3 * sizeof(int), indices.size() / 3);

        rtcCommitGeometry(geom);
        rtcAttachGeometry(m_scene, geom);
        rtcReleaseGeometry(geom);
        rtcCommitScene(m_scene);
    }
    ~EmbreeIntersector()
    {
        if (m_scene)
            rtcReleaseScene(m_scene);
        if (m_device)
            rtcReleaseDevice(m_device);
    }
    EmbreeIntersector(EmbreeIntersector const&) = delete;
    EmbreeIntersector& operator=(EmbreeIntersector const&) = delete;
    EmbreeIntersector(EmbreeIntersector&&) = delete;
    EmbreeIntersector& operator=(EmbreeIntersector&&) = delete;

    [[nodiscard]] int signed_intersection_count(fvec3 p, fvec3 dir) const
    {
        // Embree recommends FTZ/DAZ on every traversal thread.
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

        struct query_context
        {
            RTCRayQueryContext base; // MUST be first for the reinterpret_cast in the filter
            int sum;
        };

        query_context q;
        rtcInitRayQueryContext(&q.base);
        q.sum = 0;

        RTCRay ray{};
        ray.org_x = p.x;
        ray.org_y = p.y;
        ray.org_z = p.z;
        ray.dir_x = dir.x;
        ray.dir_y = dir.y;
        ray.dir_z = dir.z;
        ray.tnear = 0.0f;
        ray.tfar = std::numeric_limits<float>::infinity();
        ray.mask = 0xFFFFFFFFu;
        ray.flags = 0;

        RTCOccludedArguments rargs;
        rtcInitOccludedArguments(&rargs);
        rargs.flags = RTCRayQueryFlags(RTC_RAY_QUERY_FLAG_COHERENT | RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER);
        rargs.feature_mask = RTC_FEATURE_FLAG_ALL;
        rargs.context = &q.base;
        rargs.filter = +[](RTCFilterFunctionNArguments const* fargs)
        {
            assert(fargs->N == 1 && fargs->valid[0]);

            auto const& fray = reinterpret_cast<RTCRay&>(*fargs->ray);
            auto const& fhit = reinterpret_cast<RTCHit&>(*fargs->hit);

            auto const d = fray.dir_x * fhit.Ng_x + fray.dir_y * fhit.Ng_y + fray.dir_z * fhit.Ng_z;
            reinterpret_cast<query_context*>(fargs->context)->sum += d > 0.0f ? +1 : -1;

            // Reject the hit so traversal continues and we visit every crossing.
            fargs->valid[0] = 0;
        };
        rargs.occluded = nullptr;

        rtcOccluded1(m_scene, &ray, &rargs);
        return q.sum;
    }

private:
    RTCDevice m_device{};
    RTCScene m_scene{};
};
#endif
} // namespace antipodal
