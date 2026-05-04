#pragma once

/**
 * @file intersector_embree.hh
 * @brief Embree-backed Intersector (`EmbreeIntersector`).
 *
 * Implements the Intersector concept (see @ref intersector.hh) on top of
 * Embree 4. The type is gated on `ANTIPODAL_HAS_EMBREE`; including this header
 * without Embree enabled is always safe — it simply does not introduce
 * `EmbreeIntersector`.
 *
 * Float-only (Embree's geometry API is float). Constructed from non-owning
 * vertex/index spans via `rtcSetSharedGeometryBuffer`, so the caller's vertex
 * and index data MUST remain valid for the full lifetime of the
 * `EmbreeIntersector`. Signed crossings are accumulated in an occluded-ray
 * filter callback that rejects every hit so traversal visits all of them.
 */

#include <antipodal/math/common.hh>

#if defined(ANTIPODAL_HAS_EMBREE) && ANTIPODAL_HAS_EMBREE
#include <cassert>
#include <limits>
#include <span>

#include <pmmintrin.h>
#include <xmmintrin.h>

#include <embree4/rtcore.h>

namespace antipodal
{
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
} // namespace antipodal
#endif
