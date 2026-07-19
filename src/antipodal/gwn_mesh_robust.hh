#pragma once

// Unconditionally-robust generalized winding number for a triangle mesh.
//
// RobustMeshGwn<T> evaluates the full mesh GWN (fractional boundary + integer ray-crossing terms).
// It uses exact integer predicates so the two terms always agree.
// The classical floating-point evaluation can jump by ~±1 near the surface.
// That happens when the integer count and the atan2 boundary term disagree on a grazing edge.
// Here both terms are decided by the same quantized integer predicate.
// See math/robust_predicates.hh for the coupling identity atan2_num == -t_real.
// Their discontinuities therefore cancel exactly.
//
// The floating-point kernels in gwn_mesh.hh accept an arbitrary reference direction x0.
// The robust path instead bakes in a single fixed, axis-aligned ray (-x).
// The symbolic perturbation makes that axis-aligned ray unable to fail.
// So no caller x0, frame, or projection is needed.
// World coordinates are quantized directly and both terms use the same fixed axis.
// The integer query takes no direction at all (the ray is baked in).
// Use eval / eval_gwnr_mesh_batch_robust for the full, consistent GWN.
//
// The integer term is accelerated with an Embree user geometry, so this header requires Embree.
// Including it without ANTIPODAL_HAS_EMBREE is safe but does not introduce RobustMeshGwn.

#include <antipodal/math/common.hh>
#include <antipodal/math/robust_predicates.hh>

#if defined(ANTIPODAL_HAS_EMBREE) && ANTIPODAL_HAS_EMBREE
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include <pmmintrin.h>
#include <xmmintrin.h>

#include <embree4/rtcore.h>

namespace antipodal
{
namespace detail
{
// Quantized (integer) and double copies of a triangle, 1:1 by primID.
// The integer copy drives the exact in-triangle predicate.
// The double copy drives the (non-critical) front/back depth test.
struct robust_itriangle
{
    robust::ivec3 pos0;
    robust::ivec3 pos1;
    robust::ivec3 pos2;
};
struct robust_dtriangle
{
    dvec3 pos0;
    dvec3 pos1;
    dvec3 pos2;
};

// Stable backing referenced by the Embree callbacks (owned by RobustMeshGwn).
struct robust_geom_data
{
    robust_itriangle const* faces_iquant = nullptr;
    robust_dtriangle const* faces_double = nullptr;
    std::size_t face_count = 0;
    double pad = 0.0; // world-space AABB slack; see robust_bounds_fn
};

// Ray-query context carrying the exact (quantized) query and the double query,
// plus the accumulated signed intersection count.
struct robust_query_context
{
    RTCRayQueryContext base; // MUST be first for the reinterpret_cast below
    int sum;
    robust::ivec3 qi; // quantized query (.x depth, (.y,.z) projection)
    dvec3 qd;         // double query (for the front/back depth test)
};

// User-geometry bounds: the float AABB of the triangle, grown for two reasons.
// Embree only invokes the callback for primitives whose reported (float) box the ray hits.
// So the box must contain every query the predicate accepts, or a grazing hit is culled.
// Slack source 1: the predicate tests the quantized query.
// A query up to ~half a quantization cell outside the true AABB is still counted.
// data->pad (a couple of cells, derived from the quantizer) covers that.
// Slack source 2: converting the double box to float, and Embree testing the float ray origin.
// std::nextafter rounds each bound one ULP outward to cover that.
// False positives cost nothing — the exact predicate rejects them — so we err generous.
inline void robust_bounds_fn(RTCBoundsFunctionArguments const* args)
{
    auto const* data = static_cast<robust_geom_data const*>(args->geometryUserPtr);
    auto const& t = data->faces_double[args->primID];

    double const pad = data->pad;
    auto const mnx = std::min({t.pos0.x, t.pos1.x, t.pos2.x}) - pad;
    auto const mny = std::min({t.pos0.y, t.pos1.y, t.pos2.y}) - pad;
    auto const mnz = std::min({t.pos0.z, t.pos1.z, t.pos2.z}) - pad;
    auto const mxx = std::max({t.pos0.x, t.pos1.x, t.pos2.x}) + pad;
    auto const mxy = std::max({t.pos0.y, t.pos1.y, t.pos2.y}) + pad;
    auto const mxz = std::max({t.pos0.z, t.pos1.z, t.pos2.z}) + pad;

    constexpr float ninf = -std::numeric_limits<float>::infinity();
    constexpr float pinf = std::numeric_limits<float>::infinity();

    auto* bb = args->bounds_o;
    bb->lower_x = std::nextafter(float(mnx), ninf);
    bb->lower_y = std::nextafter(float(mny), ninf);
    bb->lower_z = std::nextafter(float(mnz), ninf);
    bb->upper_x = std::nextafter(float(mxx), pinf);
    bb->upper_y = std::nextafter(float(mxy), pinf);
    bb->upper_z = std::nextafter(float(mxz), pinf);
}

// Custom robust intersection for the -x ray.
// First the exact in-triangle predicate in the (y,z) projection, then a double front/back test.
// The ray direction is -x and the fractional part uses north pole N = +x = -dir (antipodal method).
// The integer sign is sign(dir . n) = -sign(n.x).
// Never reports occlusion, so traversal visits every candidate.
inline void robust_occluded_fn(RTCOccludedFunctionNArguments const* args)
{
    if (!args->valid[0])
        return;

    auto const* data = static_cast<robust_geom_data const*>(args->geometryUserPtr);
    auto* ctx = reinterpret_cast<robust_query_context*>(args->context);
    auto const primID = args->primID;

    auto const& ti = data->faces_iquant[primID];
    int const s = robust::inside_tri_sign(ctx->qi, ti.pos0, ti.pos1, ti.pos2); // == sign(n.x)
    if (s == 0)
        return; // query projects outside this triangle (or degenerate projection)

    // depth test in double: ray q + t*(-x) hits the supporting plane at t > 0 iff the tri is in front.
    // s != 0 => n.x != 0, so the division is safe.
    auto const& td = data->faces_double[primID];
    auto const n = cross(td.pos1 - td.pos0, td.pos2 - td.pos0);
    double const t = dot(n, td.pos0 - ctx->qd) / (-n.x); // ray dir = -x
    if (t > 0)
        ctx->sum += -s; // sign(dir . n), dir = -x
}
} // namespace detail

/**
 * @brief Prepared robust GWN evaluator for a triangle mesh.
 *
 * Built once from the mesh's vertices/indices and its weighted boundary
 * segments (e.g. from `build_boundary_segments`). Holds non-owning spans into
 * the caller's vertex/index/boundary data only during construction; afterwards
 * it owns all quantized geometry and the Embree acceleration structure, so the
 * caller's arrays need not outlive the evaluator.
 *
 * Thread-safe for concurrent const queries (the batch helper shares one
 * evaluator across threads).
 *
 * @tparam T Scalar type of the query points (`float` or `double`). Internal
 *           predicate and quantization math is always done in `double`/`int64`.
 */
template <class T>
struct RobustMeshGwn
{
    RobustMeshGwn(std::span<vec3<T> const> vertices,
                  std::span<int const> indices,
                  std::span<weighted_segment3<T> const> boundary)
    {
        assert(indices.size() % 3 == 0);

        build_quantization(vertices);

        // --- quantized + double geometry (1:1 by primID) ---
        auto const n_tris = indices.size() / 3;
        m_faces_i.reserve(n_tris);
        m_faces_d.reserve(n_tris);
        for (std::size_t t = 0; t < n_tris; ++t)
        {
            auto const a = to_d(vertices[indices[3 * t + 0]]);
            auto const b = to_d(vertices[indices[3 * t + 1]]);
            auto const c = to_d(vertices[indices[3 * t + 2]]);
            m_faces_i.push_back({quantize(a), quantize(b), quantize(c)});
            m_faces_d.push_back({a, b, c});
        }

        // --- quantized boundary (same quantizer => shared vertices match) ---
        // Endpoints are stored swapped (pos1, pos0).
        // The exact atan2_num used by fractional() expects the opposite boundary orientation.
        // That is opposite to build_boundary_segments / signed_spherical_tri_area_half_unorm(-x0).
        // Swapping negates the fractional term to match the float kernels.
        // It keeps the atan2/grazing-sign pair internally consistent.
        m_boundary.reserve(boundary.size());
        for (auto const& ws : boundary)
            m_boundary.push_back({quantize(to_d(ws.segment.pos1)), quantize(to_d(ws.segment.pos0)), double(ws.weight)});

        build_embree();
    }

    ~RobustMeshGwn()
    {
        if (m_scene)
            rtcReleaseScene(m_scene);
        if (m_device)
            rtcReleaseDevice(m_device);
    }

    RobustMeshGwn(RobustMeshGwn const&) = delete;
    RobustMeshGwn& operator=(RobustMeshGwn const&) = delete;
    RobustMeshGwn(RobustMeshGwn&&) = delete;
    RobustMeshGwn& operator=(RobustMeshGwn&&) = delete;

    // The fixed ray direction used by the robust integer term.
    // Informational only: the evaluator always casts along this axis.
    [[nodiscard]] static constexpr vec3<T> axis() { return {T(-1), T(0), T(0)}; }

    // Integer term — signed ray-crossing count along the fixed -x ray.
    // Takes no direction: the robust ray is baked into the quantization.
    // So, unlike the Intersector concept, it cannot honor an arbitrary per-query direction.
    [[nodiscard]] int signed_intersection_count(vec3<T> p) const
    {
        // Embree recommends FTZ/DAZ on every traversal thread.
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

        detail::robust_query_context q;
        rtcInitRayQueryContext(&q.base);
        q.sum = 0;
        q.qd = to_d(p);
        q.qi = quantize(q.qd);

        RTCRay ray{};
        ray.org_x = float(p.x);
        ray.org_y = float(p.y);
        ray.org_z = float(p.z);
        ray.dir_x = -1.0f; // -x ray (antipodal to fractional north pole +x)
        ray.dir_y = 0.0f;
        ray.dir_z = 0.0f;
        ray.tnear = 0.0f;
        ray.tfar = std::numeric_limits<float>::infinity();
        ray.mask = 0xFFFFFFFFu;
        ray.flags = 0;

        RTCOccludedArguments rargs;
        rtcInitOccludedArguments(&rargs);
        rargs.context = &q.base;

        rtcOccluded1(m_scene, &ray, &rargs);
        return q.sum;
    }

    // Fractional term — the robust Van Oosterom-Strackee boundary integral.
    // The atan2 numerator is the exact integer atan2_num.
    // Only the grazing (num == 0) branch consults the perturbed sign.
    // That sign is locked to the ray-triangle predicate above, so the two terms agree.
    [[nodiscard]] T fractional(vec3<T> p) const
    {
        auto const qi = quantize(to_d(p));

        // f_norm folds in the factor 2 ("half area") and the 1/(4*pi) solid-angle normalization.
        constexpr double f_norm = 2.0 / (4.0 * pi<double>);

        double half_area = 0.0;
        for (auto const& seg : m_boundary)
        {
            auto const v0 = seg.pos0 - qi; // ivec3
            auto const v1 = seg.pos1 - qi;

            dvec3 const d0{double(v0.x), double(v0.y), double(v0.z)};
            dvec3 const d1{double(v1.x), double(v1.y), double(v1.z)};
            double const l0 = length(d0);
            double const l1 = length(d1);

            // exact integer atan2 numerator (== -t_real of the edge predicate)
            robust::i64 const num = robust::atan2_num(qi, seg.pos0, seg.pos1);
            double const denom = l0 * l1 + d0.x * l1 + d1.x * l0 + dot(d0, d1);

            double contrib;
            if (num != 0)
            {
                contrib = std::atan2(double(num), denom);
            }
            else
            {
                // grazing: atan2 -> 0 (denom>0) or +-pi (denom<0).
                // The perturbed sign is locked to the ray-triangle predicate.
                int const sgn = robust::sign_num_perturbed(qi, seg.pos0, seg.pos1);
                contrib = (denom < 0.0) ? double(sgn) * pi<double> : 0.0;
            }

            half_area += seg.weight * contrib;
        }

        return T(half_area * f_norm);
    }

    // Full robust GWN: fractional boundary term + integer ray-crossing term.
    [[nodiscard]] T eval(vec3<T> p) const
    {
        return fractional(p) + static_cast<T>(signed_intersection_count(p));
    }

private:
    [[nodiscard]] static dvec3 to_d(vec3<T> p) { return {double(p.x), double(p.y), double(p.z)}; }

    [[nodiscard]] robust::ivec3 quantize(dvec3 p) const
    {
        auto const q = (p - m_center) * m_scale;
        return {
            int(std::lround(q.x)),
            int(std::lround(q.y)),
            int(std::lround(q.z)),
        };
    }

    void build_quantization(std::span<vec3<T> const> vertices)
    {
        if (vertices.empty())
        {
            m_center = {};
            m_scale = 1.0;
            return;
        }

        auto mn = to_d(vertices[0]);
        auto mx = mn;
        for (auto const& v : vertices)
        {
            auto const d = to_d(v);
            mn = {std::min(mn.x, d.x), std::min(mn.y, d.y), std::min(mn.z, d.z)};
            mx = {std::max(mx.x, d.x), std::max(mx.y, d.y), std::max(mx.z, d.z)};
        }

        auto const ext = mx - mn;
        m_center = mn + ext * 0.5;

        double const half = 0.5 * std::max({ext.x, ext.y, ext.z});
        // target ~20-bit magnitude so the 2x2 i64 predicate cannot overflow
        constexpr double max_coord = double(1 << 20) - 1;
        m_scale = half > 0.0 ? max_coord / half : 1.0;
    }

    void build_embree()
    {
        m_geom_data.faces_iquant = m_faces_i.data();
        m_geom_data.faces_double = m_faces_d.data();
        m_geom_data.face_count = m_faces_i.size();
        // Two quantization cells of AABB slack.
        // One cell covers the ~half-cell a counted query can sit outside the true triangle box.
        // (The predicate uses the quantized query, hence that slack.)
        // The second cell is headroom.
        // 1/m_scale is one cell in world units, so this auto-scales with the mesh.
        m_geom_data.pad = m_scale > 0.0 ? 2.0 / m_scale : 0.0;

        m_device = rtcNewDevice(nullptr);
        m_scene = rtcNewScene(m_device);
        rtcSetSceneBuildQuality(m_scene, RTC_BUILD_QUALITY_HIGH);
        rtcSetSceneFlags(m_scene, RTC_SCENE_FLAG_ROBUST);

        auto geom = rtcNewGeometry(m_device, RTC_GEOMETRY_TYPE_USER);
        rtcSetGeometryUserPrimitiveCount(geom, unsigned(m_faces_i.size()));
        rtcSetGeometryUserData(geom, &m_geom_data);
        rtcSetGeometryBoundsFunction(geom, detail::robust_bounds_fn, nullptr);
        rtcSetGeometryOccludedFunction(geom, detail::robust_occluded_fn);
        rtcCommitGeometry(geom);
        rtcAttachGeometry(m_scene, geom);
        rtcReleaseGeometry(geom);
        rtcCommitScene(m_scene);
    }

    struct qsegment
    {
        robust::ivec3 pos0;
        robust::ivec3 pos1;
        double weight;
    };

    dvec3 m_center{};
    double m_scale = 1.0;
    std::vector<detail::robust_itriangle> m_faces_i;
    std::vector<detail::robust_dtriangle> m_faces_d;
    std::vector<qsegment> m_boundary;
    detail::robust_geom_data m_geom_data;
    RTCDevice m_device{};
    RTCScene m_scene{};
};

/**
 * @brief Batch robust GWN over many query points, parallelized through a
 *        Dispatcher.
 *
 * Equivalent to calling `robust.eval` once per element of `positions`, writing
 * results into the matching slot of `out_wnrs`.
 *
 * @tparam Dispatcher Dispatcher concept (see @ref dispatcher.hh).
 * @tparam T          Scalar type.
 * @param  dispatcher Parallel-for backend.
 * @param  robust     Shared robust evaluator; safe to query concurrently.
 * @param  positions  Input query points.
 * @param  out_wnrs   Output buffer; must have the same size as `positions`.
 */
template <class Dispatcher, class T>
void eval_gwnr_mesh_batch_robust(Dispatcher& dispatcher,
                                 RobustMeshGwn<T> const& robust,
                                 std::span<vec3<T> const> positions,
                                 std::span<T> out_wnrs)
{
    assert(positions.size() == out_wnrs.size());
    auto const cnt = static_cast<int>(positions.size());
    dispatcher.parallel_for( //
        0, cnt,
        [&](int i)
        {
            //
            out_wnrs[i] = robust.eval(positions[i]);
        });
}
} // namespace antipodal
#endif
