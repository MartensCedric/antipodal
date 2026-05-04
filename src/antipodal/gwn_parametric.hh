#pragma once

/**
 * @file gwn_parametric.hh
 * @brief Generalized winding number for parametric surfaces — fractional,
 *        integer, and full kernels (single + batch), mirroring `gwn_mesh.hh`.
 *
 * The fractional kernel implements the antipodal paper's `parametric-wnr`
 * algorithm:
 *
 *     frac(p) = (1 / 4π) · Σᵢ ∫₀¹ (wᵢ(t) − dηᵢ(t)) dt
 *
 * where each boundary curve `φᵢ : [0, 1] → R³` contributes a smooth scalar
 * integrand built from
 *
 *   - the unit-sphere projection  `Γ = (φ − p) / ‖φ − p‖`
 *   - the vector field            `X = x0 − (x0·Γ) Γ`
 *   - the geodesic tangent        `v = Γ − (x1·Γ) x1`,  with `x1 = −x0`
 *
 * First derivatives `Γ'` flow through `fwd_diff` autodiff, then
 * everything collapses to plain `vec3<T>` math.
 *
 * ## Duck-typed concepts
 *
 * ### Curve
 * Any callable
 *
 *     curve(fwd_diff<T>) -> vec3<fwd_diff<T>>;
 *
 * with parameter domain `t ∈ [0, 1]`. Curves must be oriented so the surface
 * lies on the left, matching the convention used by `build_boundary_segments`
 * in `<antipodal/math/common.hh>`.
 *
 * ### ForEachBoundary
 * Any callable that fires a visitor once per boundary curve:
 *
 *     for_each_boundary([&](auto&& curve) { ... });
 *
 * Curves passed to the visitor may be of heterogeneous types — that's the
 * whole reason this isn't a `std::span`. Visitation order is irrelevant.
 *
 * ### Integrator
 * Any callable matching
 *
 *     integrator(F&& fun, T a, T b, integrate_config const&) -> T;
 *
 * where `fun` has signature `T(T)`. Provided implementations live in
 * `<antipodal/math/integrate_gk15.hh>` (`IntegratorGK15`, the default) and
 * `<antipodal/math/integrate_gl7.hh>` (`IntegratorGL7`).
 *
 * ### Intersector
 * Same concept as `<antipodal/intersector/intersector.hh>` — a parametric
 * surface needs a ray–surface signed-intersection-count primitive for the
 * integer term, and the contract is identical to the mesh case.
 */

#include <cassert>
#include <span>

#include <antipodal/dispatcher/dispatcher.hh>
#include <antipodal/math/common.hh>
#include <antipodal/math/fwd_diff.hh>
#include <antipodal/math/integrate.hh>
#include <antipodal/math/integrate_gk15.hh>

namespace antipodal
{
namespace detail
{
// Single-sample integrand for the parametric fractional GWN. Returns
// `w(t) - d_eta(t)`; the 1/(4π) normalization is applied once by the caller
// after summing all boundary integrals.
template <class T, class Curve>
[[nodiscard]] T parametric_gwn_integrand(Curve const& curve, vec3<T> p, vec3<T> x0, T t)
{
    auto const x1 = -x0;

    // Curve point + autodiff first derivative, projected onto S^2.
    auto const phi = curve(fwd_diff<T>::input(t));
    auto const r = phi - p;
    auto const G_fd = normalize_fd(r);

    // Drop to plain T for the geometry: only G and G' are needed downstream.
    vec3<T> const G = value_of(G_fd);
    vec3<T> const Gp = d_value_of(G_fd);

    // Vector field X = x0 - (x0·G) G, with derivative
    //     X' = -(x0·G') G - (x0·G) G'.
    T const a = dot(x0, G);
    T const ap = dot(x0, Gp);
    vec3<T> const X = x0 - a * G;
    vec3<T> const Xp = (-ap) * G - a * Gp;
    T const d_eta = dot(G, cross(X, Xp)) / dot(X, X);

    // Geodesic tangent v = G - (x1·G) x1, derivative v' = G' - (x1·G') x1.
    T const b = dot(x1, G);
    T const bp = dot(x1, Gp);
    vec3<T> const v = G - b * x1;
    vec3<T> const vp = Gp - bp * x1;
    T const w = dot(x1, cross(v, vp)) / dot(v, v);

    return w - d_eta;
}
} // namespace detail

/**
 * @brief Fractional GWN for a parametric surface at a single query point.
 *
 * Sums the per-boundary-curve integrals of the smooth integrand `(w − dη)`
 * and divides by `4π`. The integer ray-crossing term is *not* included — see
 * `eval_gwnr_parametric_single` for the full GWN.
 *
 * @tparam T                Scalar type (typically `float` or `double`).
 * @tparam ForEachBoundary  Boundary-iteration callable (see file docs).
 * @tparam Integrator       1D integrator callable; defaults to `IntegratorGK15`.
 * @param  p                Query point.
 * @param  x0               Antipodal reference direction (unit-length). The
 *                          same `x0` must be used across all related
 *                          fractional / integer / full evaluations.
 * @param  for_each_boundary  Visitor source — fires once per boundary curve.
 * @param  cfg              Integrator configuration; the default tolerance
 *                          (`1e-6`) suffices for the typically very smooth
 *                          parametric integrand.
 * @param  integrator       Integrator instance.
 * @return Fractional GWN contribution (dimensionless, in turns).
 *
 * @note `x0` is not validated; passing a non-unit vector silently corrupts
 *       the result.
 */
template <class T, class ForEachBoundary, class Integrator = IntegratorGK15>
[[nodiscard]] T eval_gwnr_parametric_single_fractional(vec3<T> p,
                                                       vec3<T> x0,
                                                       ForEachBoundary const& for_each_boundary,
                                                       integrate_config const& cfg = {},
                                                       Integrator const& integrator = {})
{
    T sum = T(0);
    for_each_boundary(
        [&](auto&& curve)
        {
            sum += integrator([&](T t) { return detail::parametric_gwn_integrand<T>(curve, p, x0, t); }, T(0), T(1), cfg);
        });
    return sum / (T(4) * pi<T>);
}

/**
 * @brief Batch fractional GWN over many query points, parallelized through
 *        a Dispatcher.
 *
 * Equivalent to calling `eval_gwnr_parametric_single_fractional` once per
 * element of `positions`, writing results into the matching slot of
 * `out_wnrs`. The boundary visitor is shared across query points; each
 * worker thread re-runs the boundary loop.
 *
 * @tparam Dispatcher       Dispatcher concept.
 * @tparam T                Scalar type.
 * @tparam ForEachBoundary  Boundary-iteration callable.
 * @tparam Integrator       1D integrator callable; defaults to `IntegratorGK15`.
 * @param  dispatcher       Parallel-for backend.
 * @param  positions        Query points.
 * @param  out_wnrs         Output buffer; must be the same size as `positions`.
 * @param  x0               Antipodal reference direction (unit-length).
 * @param  for_each_boundary  Visitor source. Must be safe to invoke
 *                            concurrently from multiple worker threads.
 * @param  cfg              Integrator configuration.
 * @param  integrator       Integrator instance.
 */
template <class Dispatcher, class T, class ForEachBoundary, class Integrator = IntegratorGK15>
void eval_gwnr_parametric_batch_fractional(Dispatcher& dispatcher,
                                           std::span<vec3<T> const> positions,
                                           std::span<T> out_wnrs,
                                           vec3<T> x0,
                                           ForEachBoundary const& for_each_boundary,
                                           integrate_config const& cfg = {},
                                           Integrator const& integrator = {})
{
    assert(positions.size() == out_wnrs.size());
    auto const cnt = static_cast<int>(positions.size());
    dispatcher.parallel_for( //
        0, cnt,
        [&](int i)
        {
            //
            out_wnrs[i] = eval_gwnr_parametric_single_fractional<T>(positions[i], x0, for_each_boundary, cfg, integrator);
        });
}

/**
 * @brief Integer GWN at a single query point — the signed ray-surface
 *        intersection count along the ray `(p, x0)`.
 *
 * Thin forwarder to the intersector. The intersector concept is identical to
 * the one used by the mesh kernels — the only difference is that for a
 * parametric surface the ray-crossing primitive may be implemented over a
 * different acceleration structure (e.g. Newton iteration on patches).
 *
 * @tparam Intersector  Intersector concept.
 * @tparam T            Scalar type.
 * @param  p            Query point.
 * @param  x0           Antipodal reference direction (unit-length).
 * @param  intersector  Intersector built over the parametric surface.
 * @return Signed crossing count.
 */
template <class Intersector, class T>
[[nodiscard]] int eval_gwnr_parametric_single_integer(vec3<T> p, vec3<T> x0, Intersector const& intersector)
{
    return intersector.signed_intersection_count(p, x0);
}

/**
 * @brief Batch integer GWN over many query points, parallelized through a
 *        Dispatcher.
 */
template <class Dispatcher, class Intersector, class T>
void eval_gwnr_parametric_batch_integer(Dispatcher& dispatcher,
                                        Intersector const& intersector,
                                        std::span<vec3<T> const> positions,
                                        std::span<int> out_counts,
                                        vec3<T> x0)
{
    assert(positions.size() == out_counts.size());
    auto const cnt = static_cast<int>(positions.size());
    dispatcher.parallel_for( //
        0, cnt,
        [&](int i)
        {
            //
            out_counts[i] = intersector.signed_intersection_count(positions[i], x0);
        });
}

/**
 * @brief Full generalized winding number at a single query point.
 *
 * Combines the fractional and integer terms:
 *
 *     GWN(p) = frac(p, x0, ∂M) + SignedIntersectionNumber(p, x0, M)
 *
 * For closed parametric surfaces (no boundary curves) the fractional term is
 * zero and the result reduces to the integer winding number.
 */
template <class Intersector, class T, class ForEachBoundary, class Integrator = IntegratorGK15>
[[nodiscard]] T eval_gwnr_parametric_single(vec3<T> p,
                                            vec3<T> x0,
                                            ForEachBoundary const& for_each_boundary,
                                            Intersector const& intersector,
                                            integrate_config const& cfg = {},
                                            Integrator const& integrator = {})
{
    auto const frac = eval_gwnr_parametric_single_fractional<T>(p, x0, for_each_boundary, cfg, integrator);
    auto const intg = intersector.signed_intersection_count(p, x0);
    return frac + static_cast<T>(intg);
}

/**
 * @brief Batch full GWN over many query points, parallelized through a
 *        Dispatcher.
 *
 * The fractional and integer contributions are fused per-point (single
 * `parallel_for`) so boundary integration and ray traversal share the same
 * parallel decomposition.
 */
template <class Dispatcher, class Intersector, class T, class ForEachBoundary, class Integrator = IntegratorGK15>
void eval_gwnr_parametric_batch(Dispatcher& dispatcher,
                                Intersector const& intersector,
                                std::span<vec3<T> const> positions,
                                std::span<T> out_wnrs,
                                vec3<T> x0,
                                ForEachBoundary const& for_each_boundary,
                                integrate_config const& cfg = {},
                                Integrator const& integrator = {})
{
    assert(positions.size() == out_wnrs.size());
    auto const cnt = static_cast<int>(positions.size());
    dispatcher.parallel_for( //
        0, cnt,
        [&](int i)
        {
            auto const frac = eval_gwnr_parametric_single_fractional<T>(positions[i], x0, for_each_boundary, cfg, integrator);
            auto const intg = intersector.signed_intersection_count(positions[i], x0);
            out_wnrs[i] = frac + static_cast<T>(intg);
        });
}
} // namespace antipodal
