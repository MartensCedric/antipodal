#include "doctest.h"
#include "gwn_baseline.hh"

#include <cmath>
#include <type_traits>
#include <vector>

#include <antipodal/common.hh>
#include <antipodal/dispatcher/dispatcher.hh>
#include <antipodal/dispatcher/dispatcher_tbb.hh>
#include <antipodal/gwn_mesh.hh>
#include <antipodal/intersector/intersector.hh>

namespace
{
template <class T>
constexpr T eps_for()
{
    if constexpr (std::is_same_v<T, float>)
        return T(1e-4);
    else
        return T(1e-9);
}

// Re-declared locally; same layout as in test_intersector.cc but kept independent.
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
        0, 2, 1, 1, 2, 3,
        4, 5, 6, 6, 5, 7,
        0, 4, 2, 2, 4, 6,
        1, 3, 5, 5, 3, 7,
        0, 1, 4, 4, 1, 5,
    };
}

// Open boundary of `unit_cube_indices_open_y`, oriented as the surrounding
// mesh traverses it (each edge in the direction inherited from the adjacent
// face triangle that survived the +Y removal): v6→v2 (-X), v7→v6 (+Z),
// v3→v7 (+X), v2→v3 (-Z). This is the orientation required for
// `eval_gwnr_mesh_single_fractional` to recover the open-mesh GWN.
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

template <class T>
antipodal::vec3<T> unit_x0()
{
    using namespace antipodal;
    return normalize(vec3<T>{T(0.31), T(0.57), T(0.76)});
}

// Mix of interior and exterior probes; chosen so the rays cast along `unit_x0`
// avoid the diagonal shared between the two triangles of each cube face (which
// would trip Möller-Trumbore's edge double-count).
template <class T>
std::vector<antipodal::vec3<T>> sample_points()
{
    using V = antipodal::vec3<T>;
    return {
        V{T(0), T(0), T(0)},
        V{T(0.1), T(-0.2), T(0.3)},
        V{T(-0.4), T(0.4), T(-0.4)},
        V{T(0.49), T(0), T(0)},
        V{T(2), T(0), T(0)},
        V{T(-2), T(-2), T(-2)},
        V{T(0), T(1.5), T(0)},
        V{T(1), T(1), T(1)},
    };
}
} // namespace

TEST_CASE_TEMPLATE("eval_gwnr_mesh_single: closed cube matches CPU baseline", T, float, double)
{
    using namespace antipodal;
    auto const tol = eps_for<T>();
    auto const verts = unit_cube_vertices<T>();
    auto const idx = unit_cube_indices_closed();
    NaiveIntersector<T> intersector{verts, idx};
    auto const x0 = unit_x0<T>();
    std::span<weighted_segment3<T> const> empty_boundary{};

    for (auto const& p : sample_points<T>())
    {
        auto const ref = antipodal_test::eval_gwn_baseline<T>(p, verts, idx);
        auto const got = eval_gwnr_mesh_single<NaiveIntersector<T>, T>(p, x0, empty_boundary, intersector);
        CHECK(got == doctest::Approx(ref).epsilon(tol));
    }
}

TEST_CASE_TEMPLATE("eval_gwnr_mesh_single: open cube (+Y removed) with boundary correction matches CPU baseline",
                   T,
                   float,
                   double)
{
    using namespace antipodal;
    auto const tol = eps_for<T>();
    auto const verts = unit_cube_vertices<T>();
    auto const idx = unit_cube_indices_open_y();
    NaiveIntersector<T> intersector{verts, idx};
    auto const boundary = unit_cube_open_y_boundary<T>();
    auto const x0 = unit_x0<T>();

    for (auto const& p : sample_points<T>())
    {
        auto const ref = antipodal_test::eval_gwn_baseline<T>(p, verts, idx);
        auto const got = eval_gwnr_mesh_single<NaiveIntersector<T>, T>(p, x0, boundary, intersector);
        CHECK(got == doctest::Approx(ref).epsilon(tol));
    }
}

TEST_CASE_TEMPLATE("eval_gwnr_mesh_batch: matches CPU baseline (single-thread)", T, float, double)
{
    using namespace antipodal;
    auto const tol = eps_for<T>();
    auto const verts = unit_cube_vertices<T>();
    auto const idx = unit_cube_indices_closed();
    NaiveIntersector<T> intersector{verts, idx};
    auto const x0 = unit_x0<T>();
    std::span<weighted_segment3<T> const> empty_boundary{};

    auto const points = sample_points<T>();

    SinglethreadDispatcher disp;
    std::vector<T> batch_out(points.size());
    eval_gwnr_mesh_batch(disp, intersector, std::span<vec3<T> const>{points}, std::span<T>{batch_out}, x0, empty_boundary);

    for (std::size_t i = 0; i < points.size(); ++i)
    {
        auto const ref = antipodal_test::eval_gwn_baseline<T>(points[i], verts, idx);
        CHECK(batch_out[i] == doctest::Approx(ref).epsilon(tol));
    }
}

#if defined(ANTIPODAL_HAS_TBB) && ANTIPODAL_HAS_TBB
TEST_CASE_TEMPLATE("eval_gwnr_mesh_batch: TBB dispatcher matches CPU baseline", T, float, double)
{
    using namespace antipodal;
    auto const tol = eps_for<T>();
    auto const verts = unit_cube_vertices<T>();
    auto const idx = unit_cube_indices_closed();
    NaiveIntersector<T> intersector{verts, idx};
    auto const x0 = unit_x0<T>();
    std::span<weighted_segment3<T> const> empty_boundary{};

    std::vector<vec3<T>> points;
    points.reserve(64);
    for (int i = 0; i < 64; ++i)
    {
        auto const f = T(i) / T(63);
        points.push_back({T(-1) + T(2) * f, T(-1) + T(2) * (T(1) - f), T(-1) + T(2) * f * f});
    }

    TbbDispatcher tbb;
    std::vector<T> tbb_out(points.size());
    eval_gwnr_mesh_batch(tbb, intersector, std::span<vec3<T> const>{points}, std::span<T>{tbb_out}, x0, empty_boundary);

    for (std::size_t i = 0; i < points.size(); ++i)
    {
        auto const ref = antipodal_test::eval_gwn_baseline<T>(points[i], verts, idx);
        CHECK(tbb_out[i] == doctest::Approx(ref).epsilon(tol));
    }
}
#endif
