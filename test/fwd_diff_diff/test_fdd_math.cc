#include "doctest.h"
#include "fdd_test_utils.hh"

#include <antipodal/math/fwd_diff_diff.hh>

#include <cmath>

using antipodal_test::fd_first;
using antipodal_test::fd_second;
using antipodal_test::fuzz_run;
using antipodal_test::uniform;

// ============================================================================
// sqrt correctness
// ============================================================================

TEST_CASE("fdd - sqrt 2")
{
    auto v = sqrt(antipodal::fdd64::input(2.0));
    CHECK(v.value == doctest::Approx(std::sqrt(2.0)).epsilon(1e-12));
    CHECK(v.d_value == doctest::Approx(1 / (2 * std::sqrt(2.0))).epsilon(1e-12));
    CHECK(v.dd_value == doctest::Approx(-1 / (4 * std::sqrt(2.0) * 2)).epsilon(1e-12));
}

TEST_CASE("fdd - sqrt differential")
{
    auto const check = [](double x0)
    {
        auto const f = [](double t) { return std::sqrt(t); };
        auto const x = antipodal::fdd64::input(x0);
        auto const result = sqrt(x);

        CHECK(result.value == doctest::Approx(f(x0)).epsilon(1e-12));
        CHECK(result.d_value == doctest::Approx(fd_first(f, x0)).epsilon(1e-9));
        CHECK(result.dd_value == doctest::Approx(fd_second(f, x0)).epsilon(1e-6));
    };
    check(0.25);
    check(1.0);
    check(4.0);
}

// ============================================================================
// exp and log
// ============================================================================

TEST_CASE("fdd - exp differential")
{
    auto const check = [](double x0)
    {
        auto const f = [](double t) { return std::exp(t); };
        auto const x = antipodal::fdd64::input(x0);
        auto const result = exp(x);

        CHECK(result.value == doctest::Approx(f(x0)).epsilon(1e-12));
        CHECK(result.d_value == doctest::Approx(fd_first(f, x0)).epsilon(1e-9));
        CHECK(result.dd_value == doctest::Approx(fd_second(f, x0)).epsilon(1e-6));
    };
    check(-3.0);
    check(-1.0);
    check(0.0);
    check(1.0);
    check(3.0);
}

TEST_CASE("fdd - log differential")
{
    auto const check = [](double x0)
    {
        auto const f = [](double t) { return std::log(t); };
        auto const x = antipodal::fdd64::input(x0);
        auto const result = log(x);

        CHECK(result.value == doctest::Approx(f(x0)).epsilon(1e-12));
        CHECK(result.d_value == doctest::Approx(fd_first(f, x0)).epsilon(1e-8));
        CHECK(result.dd_value == doctest::Approx(fd_second(f, x0)).epsilon(1e-4));
    };
    check(0.25);
    check(1.0);
    check(4.0);
}

// ============================================================================
// pow
// ============================================================================

TEST_CASE("fdd - pow with constant exponent")
{
    auto const check = [](double x0, double n, double expected_d, double expected_dd)
    {
        auto const x = antipodal::fdd64::input(x0);
        auto const result = pow(x, n);

        auto const expected_val = std::pow(x0, n);
        CHECK(result.value == doctest::Approx(expected_val).epsilon(1e-12));
        CHECK(result.d_value == doctest::Approx(expected_d).epsilon(1e-10));
        CHECK(result.dd_value == doctest::Approx(expected_dd).epsilon(1e-9));
    };

    // pow(x, 2): f' = 2x, f'' = 2
    check(0.25, 2.0, 0.5, 2.0);
    check(1.0, 2.0, 2.0, 2.0);
    check(2.0, 2.0, 4.0, 2.0);

    // pow(x, 3): f' = 3x^2, f'' = 6x
    check(0.25, 3.0, 3.0 * 0.0625, 1.5);
    check(1.0, 3.0, 3.0, 6.0);
    check(2.0, 3.0, 12.0, 12.0);

    // pow(x, 0.5) = sqrt(x): f' = 0.5/sqrt(x), f'' = -0.25/x^1.5
    check(1.0, 0.5, 0.5, -0.25);
    check(4.0, 0.5, 0.25, -0.03125);
}

TEST_CASE("fdd - pow(fwd, fwd) x^x differential")
{
    auto const check = [](double x0)
    {
        auto const f = [](double t) { return std::pow(t, t); };
        auto const x = antipodal::fdd64::input(x0);
        auto const result = pow(x, x);

        CHECK(result.value == doctest::Approx(f(x0)).epsilon(1e-10));
        CHECK(result.d_value == doctest::Approx(fd_first(f, x0)).epsilon(1e-7));
        CHECK(result.dd_value == doctest::Approx(fd_second(f, x0)).epsilon(1e-4));
    };
    check(0.5);
    check(1.2);
    check(2.0);
}

TEST_CASE("fdd - pow(fwd, fwd) x^(2x+1) differential")
{
    auto const check = [](double x0)
    {
        auto const f = [](double t) { return std::pow(t, 2.0 * t + 1.0); };
        auto const x = antipodal::fdd64::input(x0);
        auto const result = pow(x, 2.0 * x + 1.0);

        CHECK(result.value == doctest::Approx(f(x0)).epsilon(1e-10));
        CHECK(result.d_value == doctest::Approx(fd_first(f, x0)).epsilon(1e-6));
        CHECK(result.dd_value == doctest::Approx(fd_second(f, x0)).epsilon(1e-3));
    };
    check(0.5);
    check(1.2);
    check(2.0);
}

// ============================================================================
// trigonometric functions: sin, cos, tan
// (operate on raw fwd_diff_diff<T> values interpreted as radians)
// ============================================================================

TEST_CASE("fdd - sin differential")
{
    auto const check = [](double x0)
    {
        auto const f = [](double t) { return std::sin(t); };
        auto const x = antipodal::fdd64::input(x0);
        auto const result = sin(x);

        CHECK(result.value == doctest::Approx(f(x0)).epsilon(1e-12));
        CHECK(result.d_value == doctest::Approx(fd_first(f, x0)).epsilon(1e-8));
        CHECK(result.dd_value == doctest::Approx(fd_second(f, x0)).epsilon(1e-5));
    };
    check(-1.0);
    check(-0.5);
    check(0.0);
    check(0.5);
    check(1.0);
}

TEST_CASE("fdd - cos differential")
{
    auto const check = [](double x0)
    {
        auto const f = [](double t) { return std::cos(t); };
        auto const x = antipodal::fdd64::input(x0);
        auto const result = cos(x);

        CHECK(result.value == doctest::Approx(f(x0)).epsilon(1e-12));
        CHECK(result.d_value == doctest::Approx(fd_first(f, x0)).epsilon(1e-8));
        CHECK(result.dd_value == doctest::Approx(fd_second(f, x0)).epsilon(1e-5));
    };
    check(-1.0);
    check(-0.5);
    check(0.0);
    check(0.5);
    check(1.0);
}

TEST_CASE("fdd - tan differential")
{
    auto const check = [](double x0)
    {
        auto const f = [](double t) { return std::tan(t); };
        auto const x = antipodal::fdd64::input(x0);
        auto const result = tan(x);

        CHECK(result.value == doctest::Approx(f(x0)).epsilon(1e-12));
        CHECK(result.d_value == doctest::Approx(fd_first(f, x0)).epsilon(1e-8));
        CHECK(result.dd_value == doctest::Approx(fd_second(f, x0)).epsilon(1e-5));
    };
    check(-1.0);
    check(-0.5);
    check(0.0);
    check(0.5);
    check(1.0);
}

// ============================================================================
// inverse trig: asin, acos, atan
// (return plain fwd_diff_diff<T> in radians)
// ============================================================================

TEST_CASE("fdd - asin differential")
{
    auto const check = [](double x0)
    {
        auto const f = [](double t) { return std::asin(t); };
        auto const x = antipodal::fdd64::input(x0);
        auto const result = asin(x);

        CHECK(result.value == doctest::Approx(f(x0)).epsilon(1e-12));
        CHECK(result.d_value == doctest::Approx(fd_first(f, x0)).epsilon(1e-8));
        CHECK(result.dd_value == doctest::Approx(fd_second(f, x0)).epsilon(1e-5));
    };
    check(-0.8);
    check(-0.3);
    check(0.0);
    check(0.3);
    check(0.8);
}

TEST_CASE("fdd - acos differential")
{
    auto const check = [](double x0)
    {
        auto const f = [](double t) { return std::acos(t); };
        auto const x = antipodal::fdd64::input(x0);
        auto const result = acos(x);

        CHECK(result.value == doctest::Approx(f(x0)).epsilon(1e-12));
        CHECK(result.d_value == doctest::Approx(fd_first(f, x0)).epsilon(1e-8));
        CHECK(result.dd_value == doctest::Approx(fd_second(f, x0)).epsilon(1e-5));
    };
    check(-0.8);
    check(-0.3);
    check(0.0);
    check(0.3);
    check(0.8);
}

TEST_CASE("fdd - atan differential")
{
    auto const check = [](double x0)
    {
        auto const f = [](double t) { return std::atan(t); };
        auto const x = antipodal::fdd64::input(x0);
        auto const result = atan(x);

        CHECK(result.value == doctest::Approx(f(x0)).epsilon(1e-12));
        CHECK(result.d_value == doctest::Approx(fd_first(f, x0)).epsilon(1e-8));
        CHECK(result.dd_value == doctest::Approx(fd_second(f, x0)).epsilon(1e-5));
    };
    check(-2.0);
    check(-0.5);
    check(0.0);
    check(0.5);
    check(2.0);
}

// ============================================================================
// atan2
// ============================================================================

TEST_CASE("fdd - atan2(fwd, fwd) on circle differential")
{
    // (x(t), y(t)) = (cos(t), sin(t)), atan2(y,x) = t
    auto const check = [](double t0)
    {
        auto const f = [](double t) { return std::atan2(std::sin(t), std::cos(t)); };

        auto const t = antipodal::fdd64::input(t0);
        auto const x = cos(t);
        auto const y = sin(t);
        auto const result = atan2(y, x);

        CHECK(result.value == doctest::Approx(f(t0)).epsilon(1e-12));
        CHECK(result.d_value == doctest::Approx(fd_first(f, t0)).epsilon(1e-7));
        CHECK(result.dd_value == doctest::Approx(fd_second(f, t0)).epsilon(1e-4));
    };
    check(0.5);
    check(1.0);
    check(2.0);
}

TEST_CASE("fdd - atan2(fwd, fwd) on line differential")
{
    // (x(t), y(t)) = (t, 1+t)
    auto const check = [](double t0)
    {
        auto const f = [](double t) { return std::atan2(1.0 + t, t); };

        auto const t = antipodal::fdd64::input(t0);
        auto const x = t;
        auto const y = t + 1.0;
        auto const result = atan2(y, x);

        CHECK(result.value == doctest::Approx(f(t0)).epsilon(1e-12));
        CHECK(result.d_value == doctest::Approx(fd_first(f, t0)).epsilon(1e-7));
        CHECK(result.dd_value == doctest::Approx(fd_second(f, t0)).epsilon(1e-4));
    };
    check(0.5);
    check(1.0);
    check(2.0);
}

TEST_CASE("fdd - atan2 mixed overloads")
{
    // atan2(fwd, T) and atan2(T, fwd)
    auto const check_fwd_T = [](double t0)
    {
        auto const f = [](double t) { return std::atan2(t * t, 2.0); };
        auto const t = antipodal::fdd64::input(t0);
        auto const result = atan2(t * t, 2.0);

        CHECK(result.value == doctest::Approx(f(t0)).epsilon(1e-12));
        CHECK(result.d_value == doctest::Approx(fd_first(f, t0)).epsilon(1e-7));
        CHECK(result.dd_value == doctest::Approx(fd_second(f, t0)).epsilon(1e-4));
    };

    auto const check_T_fwd = [](double t0)
    {
        auto const f = [](double t) { return std::atan2(2.0, t * t + 1.0); };
        auto const t = antipodal::fdd64::input(t0);
        auto const result = atan2(2.0, t * t + 1.0);

        CHECK(result.value == doctest::Approx(f(t0)).epsilon(1e-12));
        CHECK(result.d_value == doctest::Approx(fd_first(f, t0)).epsilon(1e-7));
        CHECK(result.dd_value == doctest::Approx(fd_second(f, t0)).epsilon(1e-4));
    };

    check_fwd_T(0.5);
    check_fwd_T(1.0);
    check_T_fwd(0.5);
    check_T_fwd(1.0);
}

// ============================================================================
// log2 / log10 vs log
// ============================================================================

TEST_CASE("fdd - log2 equals log / log(2) (fuzz)")
{
    fuzz_run(0xA1062DEADBEEFULL,
             [](std::mt19937_64& rng)
             {
                 auto const x0 = uniform(rng, 0.1, 10.0);
                 auto const x = antipodal::fdd64::input(x0);

                 auto const result_log2 = log2(x);
                 auto const result_log = log(x);
                 auto const scale = 1.0 / std::log(2.0);

                 CHECK(result_log2.value == doctest::Approx(result_log.value * scale).epsilon(1e-12));
                 CHECK(result_log2.d_value == doctest::Approx(result_log.d_value * scale).epsilon(1e-12));
                 CHECK(result_log2.dd_value == doctest::Approx(result_log.dd_value * scale).epsilon(1e-12));
             });
}

TEST_CASE("fdd - log10 equals log / log(10) (fuzz)")
{
    fuzz_run(0xA10610C0FFEEULL,
             [](std::mt19937_64& rng)
             {
                 auto const x0 = uniform(rng, 0.1, 10.0);
                 auto const x = antipodal::fdd64::input(x0);

                 auto const result_log10 = log10(x);
                 auto const result_log = log(x);
                 auto const scale = 1.0 / std::log(10.0);

                 CHECK(result_log10.value == doctest::Approx(result_log.value * scale).epsilon(1e-12));
                 CHECK(result_log10.d_value == doctest::Approx(result_log.d_value * scale).epsilon(1e-12));
                 CHECK(result_log10.dd_value == doctest::Approx(result_log.dd_value * scale).epsilon(1e-12));
             });
}

// ============================================================================
// exp / log inverse relation
// ============================================================================

TEST_CASE("fdd - log(exp(x)) roundtrip (fuzz)")
{
    fuzz_run(0xE2FE2F10C0FFEEULL,
             [](std::mt19937_64& rng)
             {
                 auto const x0 = uniform(rng, -5.0, 5.0);
                 auto const x = antipodal::fdd64::input(x0);

                 auto const result = log(exp(x));

                 CHECK(result.value == doctest::Approx(x0).epsilon(1e-12));
                 CHECK(result.d_value == doctest::Approx(1.0).epsilon(1e-10));
                 CHECK(result.dd_value == doctest::Approx(0.0).epsilon(1e-8));
             });
}

TEST_CASE("fdd - exp(log(x)) roundtrip (fuzz)")
{
    fuzz_run(0xF10E2FE2FBEEULL,
             [](std::mt19937_64& rng)
             {
                 auto const x0 = uniform(rng, 0.1, 10.0);
                 auto const x = antipodal::fdd64::input(x0);

                 auto const result = exp(log(x));

                 CHECK(result.value == doctest::Approx(x0).epsilon(1e-12));
                 CHECK(result.d_value == doctest::Approx(1.0).epsilon(1e-10));
                 CHECK(result.dd_value == doctest::Approx(0.0).epsilon(1e-8));
             });
}

// ============================================================================
// composition tests
// ============================================================================

TEST_CASE("fdd - exp(sin(x)) differential")
{
    auto const check = [](double x0)
    {
        auto const f = [](double t) { return std::exp(std::sin(t)); };
        auto const x = antipodal::fdd64::input(x0);
        auto const result = exp(sin(x));

        CHECK(result.value == doctest::Approx(f(x0)).epsilon(1e-12));
        CHECK(result.d_value == doctest::Approx(fd_first(f, x0)).epsilon(1e-7));
        CHECK(result.dd_value == doctest::Approx(fd_second(f, x0)).epsilon(1e-4));
    };
    check(-1.0);
    check(0.0);
    check(0.5);
    check(1.0);
}

TEST_CASE("fdd - log(1 + x^2) differential")
{
    auto const check = [](double x0)
    {
        auto const f = [](double t) { return std::log(1.0 + t * t); };
        auto const x = antipodal::fdd64::input(x0);
        auto const result = log(1.0 + x * x);

        CHECK(result.value == doctest::Approx(f(x0)).epsilon(1e-12));
        CHECK(result.d_value == doctest::Approx(fd_first(f, x0)).epsilon(1e-8));
        CHECK(result.dd_value == doctest::Approx(fd_second(f, x0)).epsilon(1e-5));
    };
    check(-2.0);
    check(-0.5);
    check(0.0);
    check(0.5);
    check(2.0);
}

// ============================================================================
// second-derivative symmetry sanity
// ============================================================================

TEST_CASE("fdd - adding constant does not affect second derivative (fuzz)")
{
    fuzz_run(0xC0BBCABDADBADULL,
             [](std::mt19937_64& rng)
             {
                 auto const x0 = uniform(rng, -5.0, 5.0);
                 auto const c = uniform(rng, -100.0, 100.0);
                 auto const x = antipodal::fdd64::input(x0);

                 auto const f1 = x * x * x;
                 auto const f2 = x * x * x + c;

                 CHECK(f1.dd_value == doctest::Approx(f2.dd_value).epsilon(1e-12));
             });
}
