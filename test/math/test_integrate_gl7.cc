#include "doctest.h"

#include <cmath>
#include <numbers>
#include <set>

#include <antipodal/math/fwd_diff_diff.hh>
#include <antipodal/math/integrate_gl7.hh>

using antipodal::integrate_config;
using antipodal::integrate_gl7;
using antipodal::integrate_mode;

// ============================================================================
// closed-form integrals (evaluate mode)
// ============================================================================

TEST_CASE("gl7 - integrates sin(t) on [0, pi] to 2")
{
    integrate_config const cfg{.min_depth = 0, .max_depth = 12, .tol = 1e-10};
    double err = -1.0;
    int n = -1;
    auto const r = integrate_gl7([](int, double t) { return std::sin(t); }, 0.0, std::numbers::pi_v<double>, cfg, &err, &n);
    CHECK(r == doctest::Approx(2.0).epsilon(1e-12));
    CHECK(err >= 0.0);
    CHECK(n >= 1);
}

TEST_CASE("gl7 - integrates t^2 on [0, 1] to 1/3")
{
    integrate_config const cfg{.min_depth = 0, .max_depth = 12, .tol = 1e-10};
    auto const r = integrate_gl7([](int, double t) { return t * t; }, 0.0, 1.0, cfg);
    CHECK(r == doctest::Approx(1.0 / 3.0).epsilon(1e-14));
}

TEST_CASE("gl7 - integrates exp(t) on [0, 1] to e - 1")
{
    integrate_config const cfg{.min_depth = 0, .max_depth = 15, .tol = 1e-12};
    auto const r = integrate_gl7([](int, double t) { return std::exp(t); }, 0.0, 1.0, cfg);
    CHECK(r == doctest::Approx(std::numbers::e_v<double> - 1.0).epsilon(1e-12));
}

// ============================================================================
// degenerate interval
// ============================================================================

TEST_CASE("gl7 - a == b returns 0 and zeroes out params")
{
    integrate_config const cfg{};
    double err = 42.0;
    int n = 7;
    auto const r = integrate_gl7([](int, double) { return 1.0; }, 0.5, 0.5, cfg, &err, &n);
    CHECK(r == 0.0);
    CHECK(err == 0.0);
    CHECK(n == 0);
}

// ============================================================================
// min_depth forces refinement even when err == 0
// ============================================================================

TEST_CASE("gl7 - min_depth forces 2^min_depth subintervals on exact integrands")
{
    // gl7 is exact for polynomials up to degree 13, so err == 0 here.
    integrate_config const cfg{.min_depth = 3, .max_depth = 10, .tol = 1.0};
    int n = 0;
    auto const r = integrate_gl7([](int, double t) { return t; }, 0.0, 1.0, cfg, nullptr, &n);
    CHECK(r == doctest::Approx(0.5).epsilon(1e-14));
    CHECK(n >= 8);
}

// ============================================================================
// single_level mode
// ============================================================================

TEST_CASE("gl7 - single_level returns the GL7 estimate without recursion")
{
    integrate_config const cfg{};
    int call_count = 0;
    auto const r = integrate_gl7<integrate_mode::single_level>(
        [&](int, double t)
        {
            ++call_count;
            return std::sin(t);
        },
        0.0, std::numbers::pi_v<double>, cfg);
    CHECK(call_count == 7);
    CHECK(r == doctest::Approx(2.0).epsilon(1e-4));
}

// ============================================================================
// preinvoke covers every index that evaluate may query
// ============================================================================

TEST_CASE("gl7 - preinvoke index set is a superset of evaluate's")
{
    integrate_config const cfg{.min_depth = 0, .max_depth = 3, .tol = 0.0};

    std::set<int> pre_indices;
    integrate_gl7<integrate_mode::preinvoke>(
        [&](int idx, double) { pre_indices.insert(idx); return 0.0; },
        0.0, 1.0, cfg);

    std::set<int> eval_indices;
    (void)integrate_gl7(
        [&](int idx, double t)
        {
            eval_indices.insert(idx);
            return std::sin(t * 30.0);
        },
        0.0, 1.0, cfg);

    for (int idx : eval_indices)
        CHECK(pre_indices.contains(idx));
    CHECK(pre_indices.size() >= eval_indices.size());
}

// ============================================================================
// fwd_diff_diff propagation through integration
// ============================================================================

TEST_CASE("gl7 - integrand t * x propagates derivatives in x")
{
    using antipodal::fdd64;

    auto const x = fdd64::input(2.0);
    integrate_config const cfg{.min_depth = 0, .max_depth = 6, .tol = 1e-10};

    auto const r = integrate_gl7(
        [&](int, fdd64 t) { return t * x; },
        fdd64(0.0), fdd64(1.0), cfg);

    CHECK(r.value == doctest::Approx(1.0).epsilon(1e-12));
    CHECK(r.d_value == doctest::Approx(0.5).epsilon(1e-12));
    CHECK(r.dd_value == doctest::Approx(0.0).epsilon(1e-12));
}
