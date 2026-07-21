#include "doctest.h"

#include <antipodal/dispatcher/dispatcher.hh>
#include <antipodal/dispatcher/dispatcher_tbb.hh>
#include <antipodal/gwn_mesh.hh>
#include <antipodal/gwn_parametric.hh>
#include <antipodal/intersector/intersector.hh>
#include <antipodal/math/common.hh>
#include <antipodal/math/fwd_diff.hh>
#include <antipodal/math/integrate_gl7.hh>

#include <random>
#include <type_traits>
#include <vector>

namespace
{
template <class T>
constexpr T eps_for()
{
    if constexpr (std::is_same_v<T, float>)
        return T(1e-4);
    else
        return T(1e-6);
}

// Same cube/boundary/x0 helpers used by test_gwn_mesh.cc — duplicated locally
// to keep test files self-contained, matching the existing project style.
template <class T>
std::vector<antipodal::vec3<T>> unit_cube_vertices()
{
    using V = antipodal::vec3<T>;
    return {
        V{T(-0.5), T(-0.5), T(-0.5)}, V{T(+0.5), T(-0.5), T(-0.5)}, V{T(-0.5), T(+0.5), T(-0.5)},
        V{T(+0.5), T(+0.5), T(-0.5)}, V{T(-0.5), T(-0.5), T(+0.5)}, V{T(+0.5), T(-0.5), T(+0.5)},
        V{T(-0.5), T(+0.5), T(+0.5)}, V{T(+0.5), T(+0.5), T(+0.5)},
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

inline std::vector<int> unit_cube_indices_open_y()
{
    return {
        0, 2, 1, 1, 2, 3, 4, 5, 6, 6, 5, 7, 0, 4, 2, 2, 4, 6, 1, 3, 5, 5, 3, 7, 0, 1, 4, 4, 1, 5,
    };
}

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

// Straight-line segment p0 -> p1 as a parametric curve over t in [0, 1].
// Callable with fwd_diff<T> so the GWN kernel can autodiff through it.
template <class T>
struct LineCurve
{
    antipodal::vec3<T> p0;
    antipodal::vec3<T> p1;

    template <class S>
    [[nodiscard]] antipodal::vec3<S> operator()(S t) const
    {
        return p0 * (T(1) - t) + p1 * t;
    }
};

template <class T>
std::vector<LineCurve<T>> open_y_curves()
{
    auto const segs = unit_cube_open_y_boundary<T>();
    std::vector<LineCurve<T>> curves;
    curves.reserve(segs.size());
    for (auto const& ws : segs)
    {
        // weights are all 1 in this fixture (verified by the fixture itself)
        curves.push_back(LineCurve<T>{ws.segment.pos0, ws.segment.pos1});
    }
    return curves;
}

// Visitor source backed by a vector<LineCurve<T>>.
template <class T>
struct LineCurvesBoundary
{
    std::vector<LineCurve<T>> const* curves;

    template <class Visit>
    void operator()(Visit&& visit) const
    {
        for (auto const& c : *curves)
            visit(c);
    }
};

// Empty visitor source for the closed-mesh case.
struct EmptyBoundary
{
    template <class Visit>
    void operator()(Visit&&) const
    {
    }
};

// ~100 random query points in [-1.5, 1.5]^3, mixing inside / outside the cube.
template <class T>
std::vector<antipodal::vec3<T>> random_query_points(std::size_t n = 100, std::uint32_t seed = 0xC0FFEEu)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-1.5, 1.5);
    std::vector<antipodal::vec3<T>> pts;
    pts.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        pts.push_back({T(dist(rng)), T(dist(rng)), T(dist(rng))});
    return pts;
}
} // namespace

TEST_CASE_TEMPLATE("eval_gwnr_parametric_single_fractional: empty boundary returns zero", T, float, double)
{
    using namespace antipodal;
    auto const x0 = unit_x0<T>();
    auto const points = random_query_points<T>(16);

    EmptyBoundary const empty;
    for (auto const& p : points)
    {
        auto const got = eval_gwnr_parametric_single_fractional<T>(p, x0, empty);
        CHECK(got == T(0));
    }
}

TEST_CASE_TEMPLATE("eval_gwnr_parametric_single: closed cube matches mesh GWN", T, float, double)
{
    using namespace antipodal;
    auto const tol = eps_for<T>();
    auto const verts = unit_cube_vertices<T>();
    auto const idx = unit_cube_indices_closed();
    NaiveIntersector<T> intersector{verts, idx};
    auto const x0 = unit_x0<T>();
    std::span<weighted_segment3<T> const> empty_mesh_boundary{};

    EmptyBoundary const empty_param_boundary;
    auto const points = random_query_points<T>(100);
    for (auto const& p : points)
    {
        auto const ref = eval_gwnr_mesh_single<NaiveIntersector<T>, T>(p, x0, empty_mesh_boundary, intersector);
        auto const got = eval_gwnr_parametric_single<NaiveIntersector<T>, T>(p, x0, empty_param_boundary, intersector);
        CHECK(got == doctest::Approx(ref).epsilon(tol));
    }
}

TEST_CASE_TEMPLATE("eval_gwnr_parametric_single: open cube (+Y removed) matches mesh GWN", T, float, double)
{
    using namespace antipodal;
    auto const tol = eps_for<T>();
    auto const verts = unit_cube_vertices<T>();
    auto const idx = unit_cube_indices_open_y();
    NaiveIntersector<T> intersector{verts, idx};
    auto const mesh_boundary = unit_cube_open_y_boundary<T>();
    auto const x0 = unit_x0<T>();

    auto const curves = open_y_curves<T>();
    LineCurvesBoundary<T> const param_boundary{&curves};

    auto const points = random_query_points<T>(100);
    for (auto const& p : points)
    {
        auto const ref = eval_gwnr_mesh_single<NaiveIntersector<T>, T>(p, x0, mesh_boundary, intersector);
        auto const got = eval_gwnr_parametric_single<NaiveIntersector<T>, T>(p, x0, param_boundary, intersector);
        CHECK(got == doctest::Approx(ref).epsilon(tol));
    }
}

TEST_CASE_TEMPLATE("eval_gwnr_parametric_single: open cube matches mesh GWN with GL7 integrator", T, float, double)
{
    using namespace antipodal;
    auto const tol = eps_for<T>();
    auto const verts = unit_cube_vertices<T>();
    auto const idx = unit_cube_indices_open_y();
    NaiveIntersector<T> intersector{verts, idx};
    auto const mesh_boundary = unit_cube_open_y_boundary<T>();
    auto const x0 = unit_x0<T>();

    auto const curves = open_y_curves<T>();
    LineCurvesBoundary<T> const param_boundary{&curves};

    auto const points = random_query_points<T>(32, 0xBEEFu);
    integrate_config const cfg{};
    IntegratorGL7 const gl7;
    for (auto const& p : points)
    {
        auto const ref = eval_gwnr_mesh_single<NaiveIntersector<T>, T>(p, x0, mesh_boundary, intersector);
        auto const got
            = eval_gwnr_parametric_single<NaiveIntersector<T>, T>(p, x0, param_boundary, intersector, cfg, gl7);
        CHECK(got == doctest::Approx(ref).epsilon(tol));
    }
}

TEST_CASE_TEMPLATE("eval_gwnr_parametric_batch: open cube matches mesh GWN (single-thread)", T, float, double)
{
    using namespace antipodal;
    auto const tol = eps_for<T>();
    auto const verts = unit_cube_vertices<T>();
    auto const idx = unit_cube_indices_open_y();
    NaiveIntersector<T> intersector{verts, idx};
    auto const mesh_boundary = unit_cube_open_y_boundary<T>();
    auto const x0 = unit_x0<T>();

    auto const curves = open_y_curves<T>();
    LineCurvesBoundary<T> const param_boundary{&curves};

    auto const points = random_query_points<T>(100);

    SinglethreadDispatcher disp;
    std::vector<T> mesh_out(points.size());
    std::vector<T> param_out(points.size());
    eval_gwnr_mesh_batch(disp, intersector, std::span<vec3<T> const>{points}, std::span<T>{mesh_out}, x0,
                         std::span<weighted_segment3<T> const>{mesh_boundary});
    eval_gwnr_parametric_batch(disp, intersector, std::span<vec3<T> const>{points}, std::span<T>{param_out}, x0,
                               param_boundary);

    for (std::size_t i = 0; i < points.size(); ++i)
        CHECK(param_out[i] == doctest::Approx(mesh_out[i]).epsilon(tol));
}

#if defined(ANTIPODAL_HAS_TBB) && ANTIPODAL_HAS_TBB
TEST_CASE_TEMPLATE("eval_gwnr_parametric_batch: open cube matches mesh GWN (TBB)", T, float, double)
{
    using namespace antipodal;
    auto const tol = eps_for<T>();
    auto const verts = unit_cube_vertices<T>();
    auto const idx = unit_cube_indices_open_y();
    NaiveIntersector<T> intersector{verts, idx};
    auto const mesh_boundary = unit_cube_open_y_boundary<T>();
    auto const x0 = unit_x0<T>();

    auto const curves = open_y_curves<T>();
    LineCurvesBoundary<T> const param_boundary{&curves};

    auto const points = random_query_points<T>(128, 0xFEEDu);

    TbbDispatcher tbb;
    std::vector<T> mesh_out(points.size());
    std::vector<T> param_out(points.size());
    eval_gwnr_mesh_batch(tbb, intersector, std::span<vec3<T> const>{points}, std::span<T>{mesh_out}, x0,
                         std::span<weighted_segment3<T> const>{mesh_boundary});
    eval_gwnr_parametric_batch(tbb, intersector, std::span<vec3<T> const>{points}, std::span<T>{param_out}, x0,
                               param_boundary);

    for (std::size_t i = 0; i < points.size(); ++i)
        CHECK(param_out[i] == doctest::Approx(mesh_out[i]).epsilon(tol));
}
#endif
