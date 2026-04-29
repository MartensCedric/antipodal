#pragma once

#include "common.hh"
#include "dispatcher.hh"
#include "intersector.hh"

#include <cassert>
#include <span>

namespace antipodal
{
/**
 * @brief Fractional generalized winding number for a triangle mesh at a
 *        single query point (paper Algorithm `wnr-unorm`).
 *
 * Sums the unnormalized Van Oosterom–Strackee contribution of every weighted
 * boundary edge segment and divides by @f$2\pi@f$. The integer ray-crossing
 * term is @em not included — see `eval_gwnr_mesh_single` for the full GWN.
 *
 * @tparam T        Scalar type (typically `float` or `double`).
 * @param  p        Query point in world space.
 * @param  x0       Antipodal reference direction. Must be a unit vector; the
 *                  same `x0` must be used across all related fractional /
 *                  integer / full evaluations on the same mesh.
 * @param  boundary Weighted edge segments of the mesh's open boundary
 *                  @f$\partial M@f$. Pass an empty span for closed meshes
 *                  (the result is then exactly zero).
 * @return Fractional GWN contribution (dimensionless, in turns).
 *
 * @note `x0` is not validated; passing a non-unit vector silently corrupts the
 *       result.
 * @see eval_gwnr_mesh_single
 */
template <class T>
[[nodiscard]] T eval_gwnr_mesh_single_fractional(vec3<T> p, vec3<T> x0, std::span<weighted_segment3<T> const> boundary)
{
    auto const x1 = -x0;
    T area = T(0);
    for (auto const& ws : boundary)
    {
        auto const v0 = ws.segment.pos0 - p;
        auto const v1 = ws.segment.pos1 - p;
        area += ws.weight * signed_spherical_tri_area_half_unorm(x1, v0, v1);
    }
    return area / (T(2) * pi<T>);
}

/**
 * @brief Batch fractional GWN over many query points, parallelized through a
 *        Dispatcher.
 *
 * Equivalent to calling `eval_gwnr_mesh_single_fractional` once per element of
 * `positions`, writing results into the matching slot of `out_wnrs`.
 *
 * @tparam Dispatcher Any type satisfying the dispatcher concept (see
 *                    @ref dispatcher.hh).
 * @tparam T          Scalar type.
 * @param  dispatcher Parallel-for backend; passed by ref so non-`const` state
 *                    in the dispatcher is allowed.
 * @param  positions  Input query points.
 * @param  out_wnrs   Output buffer; must have the same size as `positions`.
 * @param  x0         Antipodal reference direction (unit-length).
 * @param  boundary   Weighted boundary edge segments.
 *
 * @see eval_gwnr_mesh_single_fractional
 */
template <class Dispatcher, class T>
void eval_gwnr_mesh_batch_fractional(Dispatcher& dispatcher,
                                     std::span<vec3<T> const> positions,
                                     std::span<T> out_wnrs,
                                     vec3<T> x0,
                                     std::span<weighted_segment3<T> const> boundary)
{
    assert(positions.size() == out_wnrs.size());
    auto const cnt = static_cast<int>(positions.size());
    dispatcher.parallel_for( //
        0, cnt,
        [&](int i)
        {
            //
            out_wnrs[i] = eval_gwnr_mesh_single_fractional<T>(positions[i], x0, boundary);
        });
}

/**
 * @brief Integer GWN at a single query point — the signed ray-surface
 *        intersection count along @f$(\vec{p}, \vec{x_0})@f$.
 *
 * Thin forwarder to the intersector; this is the `SignedIntersectionNumber`
 * term of the antipodal paper.
 *
 * @tparam Intersector Any type satisfying the intersector concept (see
 *                     @ref intersector.hh).
 * @tparam T           Scalar type.
 * @param  p           Query point.
 * @param  x0          Antipodal reference direction (unit-length).
 * @param  intersector Intersector built over the full mesh.
 * @return Signed crossing count (an integer).
 *
 * @see eval_gwnr_mesh_single
 */
template <class Intersector, class T>
[[nodiscard]] int eval_gwnr_mesh_single_integer(vec3<T> p, vec3<T> x0, Intersector const& intersector)
{
    return intersector.signed_intersection_count(p, x0);
}

/**
 * @brief Batch integer GWN over many query points, parallelized through a
 *        Dispatcher.
 *
 * Equivalent to calling `eval_gwnr_mesh_single_integer` once per element of
 * `positions`, writing the integer result into the matching slot of
 * `out_counts`.
 *
 * @tparam Dispatcher  Dispatcher concept.
 * @tparam Intersector Intersector concept.
 * @tparam T           Scalar type.
 * @param  dispatcher  Parallel-for backend.
 * @param  intersector Shared intersector; must be safe to query concurrently.
 * @param  positions   Input query points.
 * @param  out_counts  Output buffer; must have the same size as `positions`.
 * @param  x0          Antipodal reference direction (unit-length).
 *
 * @see eval_gwnr_mesh_single_integer
 */
template <class Dispatcher, class Intersector, class T>
void eval_gwnr_mesh_batch_integer(Dispatcher& dispatcher,
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
 * @f[
 *     \mathrm{GWN}(\vec{p}) =
 *         \mathrm{frac}(\vec{p}, \vec{x_0}, \partial M) +
 *         \mathrm{SignedIntersectionNumber}(\vec{p}, \vec{x_0}, M).
 * @f]
 * For watertight meshes the fractional term is zero and the result is the
 * classical integer winding number.
 *
 * @tparam Intersector Intersector concept.
 * @tparam T           Scalar type.
 * @param  p           Query point.
 * @param  x0          Antipodal reference direction (unit-length).
 * @param  boundary    Weighted boundary edge segments (empty span for closed
 *                     meshes).
 * @param  intersector Intersector built over the mesh.
 * @return Full GWN value (dimensionless, in turns).
 *
 * @see eval_gwnr_mesh_single_fractional
 * @see eval_gwnr_mesh_single_integer
 */
template <class Intersector, class T>
[[nodiscard]] T eval_gwnr_mesh_single(vec3<T> p,
                                      vec3<T> x0,
                                      std::span<weighted_segment3<T> const> boundary,
                                      Intersector const& intersector)
{
    auto const frac = eval_gwnr_mesh_single_fractional<T>(p, x0, boundary);
    auto const intg = intersector.signed_intersection_count(p, x0);
    return frac + static_cast<T>(intg);
}

/**
 * @brief Batch full GWN over many query points, parallelized through a
 *        Dispatcher.
 *
 * Equivalent to calling `eval_gwnr_mesh_single` once per element of
 * `positions`. The fractional and integer contributions are fused per-point
 * (single `parallel_for`) so that boundary traversal and ray traversal share
 * the same parallel decomposition.
 *
 * @tparam Dispatcher  Dispatcher concept.
 * @tparam Intersector Intersector concept.
 * @tparam T           Scalar type.
 * @param  dispatcher  Parallel-for backend.
 * @param  intersector Shared intersector; must be safe to query concurrently.
 * @param  positions   Input query points.
 * @param  out_wnrs    Output buffer; must have the same size as `positions`.
 * @param  x0          Antipodal reference direction (unit-length).
 * @param  boundary    Weighted boundary edge segments.
 *
 * @see eval_gwnr_mesh_single
 */
template <class Dispatcher, class Intersector, class T>
void eval_gwnr_mesh_batch(Dispatcher& dispatcher,
                          Intersector const& intersector,
                          std::span<vec3<T> const> positions,
                          std::span<T> out_wnrs,
                          vec3<T> x0,
                          std::span<weighted_segment3<T> const> boundary)
{
    assert(positions.size() == out_wnrs.size());
    auto const cnt = static_cast<int>(positions.size());
    dispatcher.parallel_for( //
        0, cnt,
        [&](int i)
        {
            auto const frac = eval_gwnr_mesh_single_fractional<T>(positions[i], x0, boundary);
            auto const intg = intersector.signed_intersection_count(positions[i], x0);
            out_wnrs[i] = frac + static_cast<T>(intg);
        });
}
} // namespace antipodal
