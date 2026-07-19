#include "doctest.h"
#include "gwn_baseline.hh"

#include <algorithm>
#include <cmath>
#include <span>
#include <type_traits>
#include <vector>

#include <antipodal/math/common.hh>
#include <antipodal/dispatcher/dispatcher.hh>
#include <antipodal/gwn_mesh.hh>
#include <antipodal/gwn_mesh_robust.hh>
#include <antipodal/intersector/intersector_embree.hh>

// The robust path is Embree-backed; the whole suite is a no-op without it.
#if defined(ANTIPODAL_HAS_EMBREE) && ANTIPODAL_HAS_EMBREE

namespace
{
// The robust path quantizes coordinates to ~20 bits, so its value precision is ~1e-6 regardless of T.
// The exactness it buys is in the integer/fractional sign agreement, not in accurate mantissa bits.
// Tolerances reflect that quantization floor, not the tighter float/double kernel tolerances.
template <class T>
constexpr T eps_for()
{
    if constexpr (std::is_same_v<T, float>)
        return T(1e-4);
    else
        return T(1e-5);
}

template <class T>
std::vector<antipodal::vec3<T>> unit_cube_vertices()
{
    using V = antipodal::vec3<T>;
    return {
        V{T(-0.5), T(-0.5), T(-0.5)},
        V{T(+0.5), T(-0.5), T(-0.5)},
        V{T(-0.5), T(+0.5), T(-0.5)},
        V{T(+0.5), T(+0.5), T(-0.5)},
        V{T(-0.5), T(-0.5), T(+0.5)},
        V{T(+0.5), T(-0.5), T(+0.5)},
        V{T(-0.5), T(+0.5), T(+0.5)},
        V{T(+0.5), T(+0.5), T(+0.5)},
    };
}

inline std::vector<int> unit_cube_indices_closed()
{
    return {
        0, 2, 1, 1, 2, 3, // -Z
        4, 5, 6, 6, 5, 7, // +Z
        0, 4, 2, 2, 4, 6, // -X
        1, 3, 5, 5, 3, 7, // +X
        0, 1, 4, 4, 1, 5, // -Y
        2, 6, 3, 3, 6, 7, // +Y
    };
}

// Cube with the +Y face removed.
inline std::vector<int> unit_cube_indices_open_y()
{
    return {
        0, 2, 1, 1, 2, 3, //
        4, 5, 6, 6, 5, 7, //
        0, 4, 2, 2, 4, 6, //
        1, 3, 5, 5, 3, 7, //
        0, 1, 4, 4, 1, 5, //
    };
}

// Open boundary of `unit_cube_indices_open_y` (see test_gwn_mesh.cc for the
// orientation rationale).
template <class T>
std::vector<antipodal::weighted_segment3<T>> unit_cube_open_y_boundary()
{
    using V = antipodal::vec3<T>;
    using S = antipodal::segment3<T>;
    using W = antipodal::weighted_segment3<T>;
    auto const v2 = V{T(-0.5), T(0.5), T(-0.5)};
    auto const v3 = V{T(+0.5), T(0.5), T(-0.5)};
    auto const v6 = V{T(-0.5), T(0.5), T(+0.5)};
    auto const v7 = V{T(+0.5), T(0.5), T(+0.5)};
    return {
        W{S{v6, v2}, T(1)},
        W{S{v7, v6}, T(1)},
        W{S{v3, v7}, T(1)},
        W{S{v2, v3}, T(1)},
    };
}

// Probes including several that lie exactly on the y+z=0 diagonal.
// That diagonal is shared between the two triangles of the +X and -X faces.
// The robust ray is fixed to -x, so these rays graze the shared diagonal / pass through vertices.
// Those are exactly the cases the floating-point Möller-Trumbore path double-counts on.
// The robust evaluator must still match the baseline.
template <class T>
std::vector<antipodal::vec3<T>> sample_points()
{
    using V = antipodal::vec3<T>;
    return {
        V{T(0), T(0), T(0)},          // interior, on the y+z=0 diagonal
        V{T(0.1), T(-0.2), T(0.3)},   // interior, generic
        V{T(-0.4), T(0.4), T(-0.4)},  // interior, on the diagonal
        V{T(0.49), T(0), T(0)},       // interior, near +X face, on the diagonal
        V{T(2), T(0), T(0)},          // exterior, ray grazes both face diagonals
        V{T(-2), T(-2), T(-2)},       // exterior, generic
        V{T(0), T(1.5), T(0)},        // exterior, above the cube
        V{T(1), T(1), T(1)},          // exterior, generic
    };
}
} // namespace

TEST_CASE_TEMPLATE("RobustMeshGwn: closed cube matches CPU baseline (grazing rays)", T, float, double)
{
    using namespace antipodal;
    auto const tol = eps_for<T>();
    auto const verts = unit_cube_vertices<T>();
    auto const idx = unit_cube_indices_closed();
    std::span<weighted_segment3<T> const> empty_boundary{};

    RobustMeshGwn<T> robust{std::span<vec3<T> const>{verts}, std::span<int const>{idx}, empty_boundary};

    for (auto const& p : sample_points<T>())
    {
        auto const ref = antipodal_test::eval_gwn_baseline<T>(p, verts, idx);
        CHECK(robust.eval(p) == doctest::Approx(ref).epsilon(tol));
    }
}

TEST_CASE_TEMPLATE("RobustMeshGwn: open cube (+Y removed) with boundary matches CPU baseline", T, float, double)
{
    using namespace antipodal;
    auto const tol = eps_for<T>();
    auto const verts = unit_cube_vertices<T>();
    auto const idx = unit_cube_indices_open_y();
    auto const boundary = unit_cube_open_y_boundary<T>();

    RobustMeshGwn<T> robust{std::span<vec3<T> const>{verts}, std::span<int const>{idx},
                            std::span<weighted_segment3<T> const>{boundary}};

    for (auto const& p : sample_points<T>())
    {
        auto const ref = antipodal_test::eval_gwn_baseline<T>(p, verts, idx);
        CHECK(robust.eval(p) == doctest::Approx(ref).epsilon(tol));
    }
}

TEST_CASE_TEMPLATE("eval_gwnr_mesh_batch_robust: matches CPU baseline (single-thread)", T, float, double)
{
    using namespace antipodal;
    auto const tol = eps_for<T>();
    auto const verts = unit_cube_vertices<T>();
    auto const idx = unit_cube_indices_closed();
    std::span<weighted_segment3<T> const> empty_boundary{};

    RobustMeshGwn<T> robust{std::span<vec3<T> const>{verts}, std::span<int const>{idx}, empty_boundary};

    auto const points = sample_points<T>();

    SinglethreadDispatcher disp;
    std::vector<T> batch_out(points.size());
    eval_gwnr_mesh_batch_robust(disp, robust, std::span<vec3<T> const>{points}, std::span<T>{batch_out});

    for (std::size_t i = 0; i < points.size(); ++i)
    {
        auto const ref = antipodal_test::eval_gwn_baseline<T>(points[i], verts, idx);
        CHECK(batch_out[i] == doctest::Approx(ref).epsilon(tol));
    }
}

// The payoff test.
// A non-robust GWN computes the fractional term in double and the integer term with Embree (float).
// Both run along the same fixed -x ray the robust path uses, so it is the same method, only non-robust.
// Its two terms are therefore decided by different discretizations of the query.
// Where the ray grazes a boundary edge, the float integer crossing and the double atan2 branch cut misalign.
// The misalignment here is ~1 float ULP (~1e-8).
// So the total GWN jumps by ~1 across that plateau even though the true GWN is smooth.
// The robust evaluator decides both terms from one quantized predicate, so it stays smooth.
// We sweep ~1000 queries across such a grazing and compare the largest jump between neighboring queries.
TEST_CASE("RobustMeshGwn: consistent across a grazing edge where the float frac/int split jumps")
{
    using namespace antipodal;

    // Single triangle in the x=0 plane (normal +x); its three edges are the open boundary.
    // The -x ray from a far query grazes the hypotenuse edge (y+z=1).
    std::vector<vec3<double>> const verts = {{0, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    std::vector<int> const idx = {0, 1, 2};
    auto const boundary = build_boundary_segments<double>(verts, idx);

    RobustMeshGwn<double> const robust{verts, idx, boundary};

    // Non-robust reference: double fractional + Embree (float) integer.
    // Both run along the robust fixed axis, so it is the same method, only non-robust.
    std::vector<fvec3> const fverts = {{0, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    EmbreeIntersector const embree{fverts, idx};
    auto const x0 = RobustMeshGwn<double>::axis(); // (-1, 0, 0)
    fvec3 const x0f{float(x0.x), float(x0.y), float(x0.z)};

    auto const nonrobust = [&](vec3<double> p)
    {
        double const frac = eval_gwnr_mesh_single_fractional<double>(p, x0, boundary);
        fvec3 const pf{float(p.x), float(p.y), float(p.z)};
        return frac + double(embree.signed_intersection_count(pf, x0f));
    };

    // Query ~100 away from the triangle so the true GWN is tiny and smooth.
    // z is fixed at 0.3 so the hypotenuse is grazed at y = 0.7.
    // 0.7 / 0.3 are not exactly representable in float.
    // So float(query) displaces the Embree crossing ~1e-8 from the double fractional's branch cut.
    constexpr int N = 1001;
    constexpr double center = 0.7;      // y at which the -x ray grazes y+z=1
    constexpr double half_width = 5e-7; // sweep ±5e-7 around it (steps ~1e-9)

    double max_jump_robust = 0.0;
    double max_jump_nonrobust = 0.0;
    double prev_r = 0.0;
    double prev_nr = 0.0;
    for (int i = 0; i < N; ++i)
    {
        double const s = center - half_width + 2.0 * half_width * (double(i) / double(N - 1));
        vec3<double> const p{100.0, s, 0.3};

        double const r = robust.eval(p);
        double const nr = nonrobust(p);
        if (i > 0)
        {
            max_jump_robust = std::max(max_jump_robust, std::abs(r - prev_r));
            max_jump_nonrobust = std::max(max_jump_nonrobust, std::abs(nr - prev_nr));
        }
        prev_r = r;
        prev_nr = nr;
    }

    // The true GWN varies by < 1e-4 across the whole 1e-6 sweep.
    // So consecutive robust samples must never jump by anything close to the ~1 spurious step.
    CHECK(max_jump_robust < 0.1);
    // ...whereas the non-robust split provokes the spurious ~1 jump.
    CHECK(max_jump_nonrobust > 0.5);
}

#endif // ANTIPODAL_HAS_EMBREE
