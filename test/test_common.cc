#include "doctest.h"

#include <cmath>
#include <type_traits>

#include <antipodal/math/common.hh>

namespace
{
template <class T>
constexpr T eps_for()
{
    if constexpr (std::is_same_v<T, float>)
        return T(1e-5);
    else
        return T(1e-12);
}

template <class T>
antipodal::vec3<T> v(T x, T y, T z)
{
    return {x, y, z};
}
} // namespace

TEST_CASE_TEMPLATE("signed_spherical_tri_area_half_unorm: degenerate cases are zero", T, float, double)
{
    using namespace antipodal;
    auto const tol = eps_for<T>();
    auto const x1 = v<T>(0, 0, 1);
    auto const a = v<T>(1, 0, 0);

    SUBCASE("v0 == v1")
    {
        CHECK(signed_spherical_tri_area_half_unorm<T>(x1, a, a) == doctest::Approx(T(0)).epsilon(tol));
    }
    SUBCASE("v0 parallel to x1")
    {
        CHECK(signed_spherical_tri_area_half_unorm<T>(x1, x1, a) == doctest::Approx(T(0)).epsilon(tol));
    }
    SUBCASE("v1 parallel to x1")
    {
        CHECK(signed_spherical_tri_area_half_unorm<T>(x1, a, x1) == doctest::Approx(T(0)).epsilon(tol));
    }
}

TEST_CASE_TEMPLATE("signed_spherical_tri_area_half_unorm: octant has magnitude pi/4", T, float, double)
{
    using namespace antipodal;
    auto const tol = eps_for<T>();
    auto const x1 = v<T>(0, 0, 1);
    auto const x = v<T>(1, 0, 0);
    auto const y = v<T>(0, 1, 0);

    auto const a01 = signed_spherical_tri_area_half_unorm<T>(x1, x, y);
    auto const a10 = signed_spherical_tri_area_half_unorm<T>(x1, y, x);

    // Magnitude is one octant of the unit sphere (4*pi/8 = pi/2), halved (the
    // helper drops the leading factor of 2) -> pi/4.
    CHECK(std::abs(a01) == doctest::Approx(pi<T> / T(4)).epsilon(tol));
    CHECK(std::abs(a10) == doctest::Approx(pi<T> / T(4)).epsilon(tol));

    // Swapping v0 and v1 negates the signed area.
    CHECK(a01 == doctest::Approx(-a10).epsilon(tol));
}

TEST_CASE_TEMPLATE("signed_spherical_tri_area_half_unorm: flipping x1 negates the area", T, float, double)
{
    using namespace antipodal;
    auto const tol = eps_for<T>();
    auto const x1 = v<T>(0, 0, 1);
    auto const x = v<T>(1, 0, 0);
    auto const y = v<T>(0, 1, 0);

    auto const a_pos = signed_spherical_tri_area_half_unorm<T>(x1, x, y);
    auto const a_neg = signed_spherical_tri_area_half_unorm<T>(-x1, x, y);
    CHECK(a_pos == doctest::Approx(-a_neg).epsilon(tol));
}

TEST_CASE_TEMPLATE("signed_spherical_tri_area_half_unorm: equator loop sums to half a hemisphere", T, float, double)
{
    using namespace antipodal;
    auto const tol = eps_for<T>();
    auto const x1 = v<T>(0, 0, 1);

    // Four equator points; each consecutive pair plus x1 forms a quadrant of the
    // upper hemisphere. The signed sum should equal half the hemisphere area
    // (the helper returns half each per-edge contribution): 2*pi / 2 = pi.
    vec3<T> const corners[4] = {
        v<T>(1, 0, 0),
        v<T>(0, 1, 0),
        v<T>(-1, 0, 0),
        v<T>(0, -1, 0),
    };

    T sum = T(0);
    for (int i = 0; i < 4; ++i)
    {
        auto const v0 = corners[i];
        auto const v1 = corners[(i + 1) % 4];
        sum += signed_spherical_tri_area_half_unorm<T>(x1, v0, v1);
    }
    CHECK(std::abs(sum) == doctest::Approx(pi<T>).epsilon(tol));
}

TEST_CASE_TEMPLATE("signed_spherical_tri_area_half_unorm: invariant under uniform scaling of v0/v1", T, float, double)
{
    using namespace antipodal;
    auto const tol = eps_for<T>();
    auto const x1 = v<T>(0, 0, 1);
    auto const v0 = v<T>(T(1.3), T(-0.4), T(0.2));
    auto const v1 = v<T>(T(-0.7), T(1.1), T(0.5));

    auto const baseline = signed_spherical_tri_area_half_unorm<T>(x1, v0, v1);

    // Independently scale each vector — the unnormalized formulation only
    // depends on direction, not magnitude.
    CHECK(signed_spherical_tri_area_half_unorm<T>(x1, T(7) * v0, T(0.01) * v1) == doctest::Approx(baseline).epsilon(tol));
    CHECK(signed_spherical_tri_area_half_unorm<T>(x1, T(0.001) * v0, T(1000) * v1) == doctest::Approx(baseline).epsilon(tol));
}
