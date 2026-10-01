#pragma once

// Unconditionally-robust generalized winding number for a triangle mesh.
//
// RobustMeshGwn<T, Precision> evaluates the full mesh GWN (fractional boundary + integer ray-crossing terms).
// It uses exact predicates so the two terms always agree.
// The classical floating-point evaluation can jump by ~±1 near the surface.
// That happens when the integer count and the atan2 boundary term disagree on a grazing edge.
// Here both terms are decided by the same exact predicate.
// See math/robust_predicates.hh for the coupling identity atan2_num == -t_real.
// Their discontinuities therefore cancel exactly.
//
// The floating-point kernels in gwn_mesh.hh accept an arbitrary reference direction x0.
// The robust path instead bakes in a single fixed, axis-aligned ray (-x).
// The symbolic perturbation makes that axis-aligned ray unable to fail.
// So no caller x0, frame, or projection is needed.
// Both terms use the same fixed axis.
// Precision (robust::int64_grid, int128_grid or exact_float) picks how the predicate is made exact;
// see robust_predicates.hh for what each costs and how accurate it is.
// The integer query takes no direction at all (the ray is baked in).
// Use eval / eval_gwnr_mesh_batch_robust for the full, consistent GWN.
//
// The integer term is accelerated with an Embree user geometry, so this header requires Embree.
// Including it without ANTIPODAL_HAS_EMBREE is safe but does not introduce RobustMeshGwn.

#include <antipodal/math/common.hh>
#include <antipodal/math/robust_predicates.hh>

#if defined(ANTIPODAL_HAS_EMBREE) && ANTIPODAL_HAS_EMBREE
#include <embree4/rtcore.h>
#include <pmmintrin.h>
#include <xmmintrin.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace antipodal
{
namespace detail
{
// The predicate's copy (quantized, or the doubles for exact_float) and a double copy of a triangle, 1:1 by primID.
// The predicate's copy decides everything the count depends on: the in-triangle test and the front/back test.
// The double copy only feeds Embree's bounding boxes.
template <class C>
struct robust_ptriangle
{
    vec3<C> pos0;
    vec3<C> pos1;
    vec3<C> pos2;
};
struct robust_dtriangle
{
    dvec3 pos0;
    dvec3 pos1;
    dvec3 pos2;
};

// Stable backing referenced by the Embree callbacks (owned by RobustMeshGwn).
template <class C>
struct robust_geom_data
{
    robust_ptriangle<C> const* faces_pred = nullptr;
    robust_dtriangle const* faces_double = nullptr;
    std::size_t face_count = 0;
    double pad = 0.0; // world-space AABB slack; see robust_bounds_fn
};

// Ray-query context carrying the predicate's query, plus the accumulated signed intersection count.
template <class C>
struct robust_query_context
{
    RTCRayQueryContext base; // MUST be first for the reinterpret_cast below
    int sum;
    vec3<C> qi; // query as the predicate sees it (.x depth, (.y,.z) projection)
};

// User-geometry bounds: the float AABB of the triangle, grown for two reasons.
// Embree only invokes the callback for primitives whose reported (float) box the ray hits.
// So the box must contain every query the predicate accepts, or a grazing hit is culled.
// Slack source 1: a grid policy tests the quantized query.
// A query up to ~half a quantization cell outside the true AABB is still counted.
// data->pad (a couple of cells, derived from the quantizer; 0 for exact_float) covers that.
// Slack source 2: converting the double box to float, and Embree testing the float ray origin.
// std::nextafter rounds each bound one ULP outward to cover that.
// False positives cost nothing — the exact predicate rejects them — so we err generous.
template <class C>
void robust_bounds_fn(RTCBoundsFunctionArguments const* args)
{
    auto const* data = static_cast<robust_geom_data<C> const*>(args->geometryUserPtr);
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
// First the exact in-triangle predicate in the (y,z) projection, then the front/back test on the same triangle.
// The ray direction is -x and the fractional part uses north pole N = +x = -dir (antipodal method).
// The integer sign is sign(dir . n) = -sign(n.x).
// Never reports occlusion, so traversal visits every candidate.
template <class C>
void robust_occluded_fn(RTCOccludedFunctionNArguments const* args)
{
    if (!args->valid[0])
        return;

    auto const* data = static_cast<robust_geom_data<C> const*>(args->geometryUserPtr);
    auto* ctx = reinterpret_cast<robust_query_context<C>*>(args->context);
    auto const primID = args->primID;

    auto const& ti = data->faces_pred[primID];
    auto const hit = robust::project_into_tri(ctx->qi, ti.pos0, ti.pos1, ti.pos2); // .sign == sign(n.x)
    if (hit.sign == 0)
        return; // query projects outside this triangle (or degenerate projection)

    // The ray meets the triangle at the barycentric mix of its vertices' depths.
    // That always lies between them, so the front/back answer cannot contradict the in-triangle test,
    // however close to parallel to the ray the triangle is.
    // Rounding in double only matters for a query within rounding of the triangle, where the GWN jumps anyway.
    double const hit_x = (hit.w0 * double(ti.pos0.x) + hit.w1 * double(ti.pos1.x) + hit.w2 * double(ti.pos2.x))
                       / (hit.w0 + hit.w1 + hit.w2);
    if (double(ctx->qi.x) >= hit_x) // ray dir = -x; a tie is in front under the eps1 perturbation
        ctx->sum += -hit.sign;      // sign(dir . n), dir = -x
}
} // namespace detail

/**
 * @brief Prepared robust GWN evaluator for a triangle mesh.
 *
 * Built once from the mesh's vertices/indices and its weighted boundary
 * segments (e.g. from `build_boundary_segments`). Holds non-owning spans into
 * the caller's vertex/index/boundary data only during construction; afterwards
 * it owns the predicates' geometry and the Embree acceleration structure, so the
 * caller's arrays need not outlive the evaluator.
 *
 * Mesh coordinates must fit in float under every policy: Embree culls with float
 * boxes, and a triangle beyond float range is never reported. Queries may lie
 * farther out; each policy states its own query range.
 *
 * Thread-safe for concurrent const queries (the batch helper shares one
 * evaluator across threads).
 *
 * @tparam T         Scalar type of the query points (`float` or `double`).
 * @tparam Precision How the exact predicates are evaluated:
 *                   robust::int64_grid, robust::int128_grid or robust::exact_float.
 *                   The default is exact_float for double and int64_grid for float;
 *                   robust_predicates.hh documents the accuracy and cost of each.
 */
template <class T, class Precision = robust::default_precision<T>>
struct RobustMeshGwn
{
    using precision = Precision;
    using coord = typename precision::coord;
    using qvec3 = vec3<coord>;

    RobustMeshGwn(std::span<vec3<T> const> vertices,
                  std::span<int const> indices,
                  std::span<weighted_segment3<T> const> boundary)
    {
        assert(indices.size() % 3 == 0);

        build_quantization(vertices);

        // --- quantized + double geometry (1:1 by primID) ---
        auto const n_tris = indices.size() / 3;
        m_faces_pred.reserve(n_tris);
        m_faces_d.reserve(n_tris);
        for (std::size_t t = 0; t < n_tris; ++t)
        {
            auto const a = to_d(vertices[indices[3 * t + 0]]);
            auto const b = to_d(vertices[indices[3 * t + 1]]);
            auto const c = to_d(vertices[indices[3 * t + 2]]);
            m_faces_pred.push_back({quantize(a), quantize(b), quantize(c)});
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
    // Takes no direction: the robust ray is baked into the predicates.
    // So, unlike the Intersector concept, it cannot honor an arbitrary per-query direction.
    [[nodiscard]] int signed_intersection_count(vec3<T> p) const
    {
        // Embree recommends FTZ/DAZ on every traversal thread.
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

        detail::robust_query_context<coord> q;
        rtcInitRayQueryContext(&q.base);
        q.sum = 0;
        q.qi = quantize(to_d(p));

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
    // The atan2 numerator is atan2_num, whose sign is exact under every policy.
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
            auto const v0 = seg.pos0 - qi;
            auto const v1 = seg.pos1 - qi;

            dvec3 const d0{double(v0.x), double(v0.y), double(v0.z)};
            dvec3 const d1{double(v1.x), double(v1.y), double(v1.z)};
            double const l0 = length(d0);
            double const l1 = length(d1);

            // atan2 numerator with exact sign (== -t_real of the edge predicate)
            auto const num = robust::atan2_num(qi, seg.pos0, seg.pos1);

            // denom = l0*l1 + d0.x*l1 + d1.x*l0 + dot(d0, d1) = (l0 + d0.x)*(l1 + d1.x) + d0.y*d1.y + d0.z*d1.z.
            // l + d.x cancels badly when d points almost along -x (an edge seen along the ray),
            // so there we use l + d.x = (d.y^2 + d.z^2) / (l - d.x). On a grid the (y,z) sums are exact integers.
            auto const l_plus_x = [](qvec3 v, double l)
            {
                if (v.x >= 0)
                    return l + double(v.x);
                return robust::dot_yz(v, v) / (l - double(v.x));
            };
            double const yz_dot = robust::dot_yz(v0, v1);
            double const denom = l_plus_x(v0, l0) * l_plus_x(v1, l1) + yz_dot;

            double contrib;
            if (num != 0)
            {
                contrib = std::atan2(double(num), denom);
            }
            else if (v0.y == 0 && v0.z == 0 && v0.x < 0 && (v1.y != 0 || v1.z != 0))
            {
                // The ray passes through pos0, where num and denom both vanish.
                // Take the limit for the perturbed query, as the edge predicate does.
                contrib = v1.z != 0 ? std::atan2(double(v1.z), -double(v1.y)) : (v1.y > 0 ? -pi<double> : 0.0);
            }
            else if (v1.y == 0 && v1.z == 0 && v1.x < 0 && (v0.y != 0 || v0.z != 0))
            {
                // Same when the ray passes through pos1.
                contrib = v0.z != 0 ? std::atan2(-double(v0.z), -double(v0.y)) : (v0.y > 0 ? pi<double> : 0.0);
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
    [[nodiscard]] T eval(vec3<T> p) const { return fractional(p) + static_cast<T>(signed_intersection_count(p)); }

private:
    [[nodiscard]] static dvec3 to_d(vec3<T> p) { return {double(p.x), double(p.y), double(p.z)}; }

    // The query or vertex as the predicates see it: on the grid, or the doubles themselves for exact_float.
    [[nodiscard]] qvec3 quantize(dvec3 p) const
    {
        if constexpr (robust::is_grid_policy<precision>)
        {
            // Clamp far away queries so the determinants cannot overflow.
            // The clamp sits at about 500 mesh sizes and never moves a query across the mesh's bounding box,
            // so the ray crossings are unchanged; the fractional term stops decaying there,
            // which costs up to ~1.5e-7 absolute beyond it.
            static constexpr double max_q = double(robust::i64(1) << (precision::grid_bits + 10));
            auto const q = (p - m_center) * m_scale;
            auto const round = [](double v) { return coord(std::llround(std::clamp(v, -max_q, max_q))); };
            return {round(q.x), round(q.y), round(q.z)};
        }
        else
        {
            return p;
        }
    }

    // Only the grid policies have a frame; exact_float keeps the defaults, center 0 and scale 1.
    void build_quantization(std::span<vec3<T> const> vertices)
    {
        if constexpr (robust::is_grid_policy<precision>)
        {
            if (vertices.empty())
                return;

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

            // target a grid_bits magnitude so the 2x2 determinants cannot overflow
            double const half = 0.5 * std::max({ext.x, ext.y, ext.z});
            constexpr double max_coord = double((robust::i64(1) << precision::grid_bits) - 1);
            m_scale = half > 0.0 ? max_coord / half : 1.0;
        }
    }

    void build_embree()
    {
        m_geom_data.faces_pred = m_faces_pred.data();
        m_geom_data.faces_double = m_faces_d.data();
        m_geom_data.face_count = m_faces_pred.size();
        // Two quantization cells of AABB slack.
        // One cell covers the ~half-cell a counted query can sit outside the true triangle box.
        // (The predicate uses the quantized query, hence that slack.)
        // The second cell is headroom.
        // 1/m_scale is one cell in world units, so this auto-scales with the mesh.
        // exact_float tests the query itself, so it needs none.
        m_geom_data.pad = robust::is_grid_policy<precision> && m_scale > 0.0 ? 2.0 / m_scale : 0.0;

        m_device = rtcNewDevice(nullptr);
        m_scene = rtcNewScene(m_device);
        rtcSetSceneBuildQuality(m_scene, RTC_BUILD_QUALITY_HIGH);
        rtcSetSceneFlags(m_scene, RTC_SCENE_FLAG_ROBUST);

        auto geom = rtcNewGeometry(m_device, RTC_GEOMETRY_TYPE_USER);
        rtcSetGeometryUserPrimitiveCount(geom, unsigned(m_faces_pred.size()));
        rtcSetGeometryUserData(geom, &m_geom_data);
        rtcSetGeometryBoundsFunction(geom, detail::robust_bounds_fn<coord>, nullptr);
        rtcSetGeometryOccludedFunction(geom, detail::robust_occluded_fn<coord>);
        rtcCommitGeometry(geom);
        rtcAttachGeometry(m_scene, geom);
        rtcReleaseGeometry(geom);
        rtcCommitScene(m_scene);
    }

    struct qsegment
    {
        qvec3 pos0;
        qvec3 pos1;
        double weight;
    };

    dvec3 m_center{};
    double m_scale = 1.0;
    std::vector<detail::robust_ptriangle<coord>> m_faces_pred;
    std::vector<detail::robust_dtriangle> m_faces_d;
    std::vector<qsegment> m_boundary;
    detail::robust_geom_data<coord> m_geom_data;
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
 * @tparam Precision  Precision policy of the evaluator.
 * @param  dispatcher Parallel-for backend.
 * @param  robust     Shared robust evaluator; safe to query concurrently.
 * @param  positions  Input query points.
 * @param  out_wnrs   Output buffer; must have the same size as `positions`.
 */
template <class Dispatcher, class T, class Precision>
void eval_gwnr_mesh_batch_robust(Dispatcher& dispatcher,
                                 RobustMeshGwn<T, Precision> const& robust,
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
