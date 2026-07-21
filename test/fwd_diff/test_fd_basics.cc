#include "doctest.h"

#include <antipodal/math/common.hh>
#include <antipodal/math/fwd_diff.hh>

#include <cmath>
#include <limits>

// ============================================================================
// input / constant construction semantics
// ============================================================================

TEST_CASE("fd - constant construction")
{
    auto const c0 = antipodal::fd64::constant(0.0);
    CHECK(c0.value == 0.0);
    CHECK(c0.d_value == 0.0);

    auto const c1 = antipodal::fd64::constant(1.5);
    CHECK(c1.value == 1.5);
    CHECK(c1.d_value == 0.0);

    auto const c2 = antipodal::fd64::constant(-2.3);
    CHECK(c2.value == -2.3);
    CHECK(c2.d_value == 0.0);
}

TEST_CASE("fd - input construction")
{
    auto const x0 = antipodal::fd64::input(0.0);
    CHECK(x0.value == 0.0);
    CHECK(x0.d_value == 1.0);

    auto const x1 = antipodal::fd64::input(1.5);
    CHECK(x1.value == 1.5);
    CHECK(x1.d_value == 1.0);

    auto const x2 = antipodal::fd64::input(-2.3);
    CHECK(x2.value == -2.3);
    CHECK(x2.d_value == 1.0);
}

// ============================================================================
// unary +/- preserve / negate derivatives
// ============================================================================

TEST_CASE("fd - unary plus preserves all components")
{
    auto const x = antipodal::fd64{1.5, 2.3};
    auto const px = +x;
    CHECK(px.value == x.value);
    CHECK(px.d_value == x.d_value);
}

TEST_CASE("fd - unary minus negates all components")
{
    auto const x = antipodal::fd64{1.5, 2.3};
    auto const mx = -x;
    CHECK(mx.value == -1.5);
    CHECK(mx.d_value == -2.3);
}

// ============================================================================
// binary arithmetic rules on simple polynomials
// ============================================================================

TEST_CASE("fd - linear polynomial f(x) = 3x + 2")
{
    auto const check = [](double x0)
    {
        auto const x = antipodal::fd64::input(x0);
        auto const f = 3.0 * x + 2.0;
        CHECK(f.value == doctest::Approx(3.0 * x0 + 2.0).epsilon(1e-12));
        CHECK(f.d_value == doctest::Approx(3.0).epsilon(1e-12));
    };
    check(-2.0);
    check(0.5);
    check(3.0);
}

TEST_CASE("fd - quadratic f(x) = x^2")
{
    auto const check = [](double x0)
    {
        auto const x = antipodal::fd64::input(x0);
        auto const f = x * x;
        CHECK(f.value == doctest::Approx(x0 * x0).epsilon(1e-12));
        CHECK(f.d_value == doctest::Approx(2.0 * x0).epsilon(1e-12));
    };
    check(-2.0);
    check(0.5);
    check(3.0);
}

TEST_CASE("fd - cubic f(x) = x^3 + 2x")
{
    auto const check = [](double x0)
    {
        auto const x = antipodal::fd64::input(x0);
        auto const f = x * x * x + 2.0 * x;
        CHECK(f.value == doctest::Approx(x0 * x0 * x0 + 2.0 * x0).epsilon(1e-12));
        CHECK(f.d_value == doctest::Approx(3.0 * x0 * x0 + 2.0).epsilon(1e-12));
    };
    check(-2.0);
    check(0.5);
    check(3.0);
}

// ============================================================================
// division correctness
// ============================================================================

TEST_CASE("fd - division f(x) = (x^2 + 1) / (2x + 3)")
{
    auto const check = [](double x0)
    {
        auto const x = antipodal::fd64::input(x0);
        auto const f = (x * x + 1.0) / (2.0 * x + 3.0);

        // f(x) = (x^2 + 1) / (2x + 3)
        auto const num = x0 * x0 + 1.0;
        auto const den = 2.0 * x0 + 3.0;
        auto const expected_val = num / den;

        // f'(x) = (2x^2 + 6x - 2) / (2x+3)^2
        auto const expected_d = (2.0 * x0 * x0 + 6.0 * x0 - 2.0) / (den * den);

        CHECK(f.value == doctest::Approx(expected_val).epsilon(1e-12));
        CHECK(f.d_value == doctest::Approx(expected_d).epsilon(1e-10));
    };
    check(-0.5);
    check(1.0);
    check(2.0);
}

TEST_CASE("fd - division T / fwd: g(x) = 5 / (x^2 + 1)")
{
    auto const check = [](double x0)
    {
        auto const x = antipodal::fd64::input(x0);
        auto const g = 5.0 / (x * x + 1.0);

        auto const den = x0 * x0 + 1.0;
        auto const expected_val = 5.0 / den;
        // g'(x) = -10x / (x^2+1)^2
        auto const expected_d = -10.0 * x0 / (den * den);

        CHECK(g.value == doctest::Approx(expected_val).epsilon(1e-12));
        CHECK(g.d_value == doctest::Approx(expected_d).epsilon(1e-10));
    };
    check(-1.5);
    check(0.0);
    check(2.0);
}

// ============================================================================
// min / max selection semantics
// ============================================================================

TEST_CASE("fd - min returns smaller value with its derivative")
{
    auto const a = antipodal::fd64{2.0, 1.0};
    auto const b = antipodal::fd64{3.0, 2.0};

    auto const m = min(a, b);
    CHECK(m.value == 2.0);
    CHECK(m.d_value == 1.0);

    auto const m2 = min(b, a);
    CHECK(m2.value == 2.0);
    CHECK(m2.d_value == 1.0);
}

TEST_CASE("fd - max returns larger value with its derivative")
{
    auto const a = antipodal::fd64{2.0, 1.0};
    auto const b = antipodal::fd64{3.0, 2.0};

    auto const m = max(a, b);
    CHECK(m.value == 3.0);
    CHECK(m.d_value == 2.0);

    auto const m2 = max(b, a);
    CHECK(m2.value == 3.0);
    CHECK(m2.d_value == 2.0);
}

TEST_CASE("fd - min/max tie uses <= / >= semantics")
{
    auto const a = antipodal::fd64{2.0, 1.0};
    auto const b = antipodal::fd64{2.0, 3.0}; // same value, different derivative

    // min uses <=, so a should be returned (first operand on tie)
    auto const m = min(a, b);
    CHECK(m.d_value == 1.0);

    // max uses >=, so a should be returned (first operand on tie)
    auto const mx = max(a, b);
    CHECK(mx.d_value == 1.0);
}

TEST_CASE("fd - min/max with T")
{
    auto const a = antipodal::fd64{2.0, 1.0};

    auto const m1 = min(a, 3.0);
    CHECK(m1.value == 2.0);
    CHECK(m1.d_value == 1.0);

    auto const m2 = min(a, 1.0);
    CHECK(m2.value == 1.0);
    CHECK(m2.d_value == 0.0);

    auto const m3 = max(a, 1.0);
    CHECK(m3.value == 2.0);
    CHECK(m3.d_value == 1.0);

    auto const m4 = max(a, 3.0);
    CHECK(m4.value == 3.0);
    CHECK(m4.d_value == 0.0);
}

// ============================================================================
// abs sign handling
// ============================================================================

TEST_CASE("fd - abs flips sign for negative values")
{
    auto const x_pos = antipodal::fd64{2.0, 1.5};
    auto const x_neg = antipodal::fd64{-2.0, 1.5};

    auto const abs_pos = abs(x_pos);
    CHECK(abs_pos.value == 2.0);
    CHECK(abs_pos.d_value == 1.5);

    auto const abs_neg = abs(x_neg);
    CHECK(abs_neg.value == 2.0);
    CHECK(abs_neg.d_value == -1.5);
}

TEST_CASE("fd - abs of function that never crosses zero")
{
    // f(x) = x^2 + 1 is always positive, so abs(f) = f
    auto const check = [](double x0)
    {
        auto const f = [](double t) { return t * t + 1.0; };
        auto const x = antipodal::fd64::input(x0);
        auto const result = abs(x * x + 1.0);

        CHECK(result.value == doctest::Approx(f(x0)).epsilon(1e-12));
        CHECK(result.d_value == doctest::Approx(2.0 * x0).epsilon(1e-12));
    };
    check(-2.0);
    check(0.0);
    check(1.5);
}

// ============================================================================
// comparisons ignore derivatives
// ============================================================================

TEST_CASE("fd - comparisons depend only on value")
{
    auto const a = antipodal::fd64{2.0, 1.0};
    auto const b = antipodal::fd64{2.0, 3.0}; // same value, different derivative
    auto const c = antipodal::fd64{3.0, 0.0};

    CHECK(a == b);
    CHECK(!(a != b));
    CHECK(a < c);
    CHECK(a <= c);
    CHECK(c > a);
    CHECK(c >= a);
    CHECK(a <= b);
    CHECK(a >= b);
}

TEST_CASE("fd - comparisons with T")
{
    auto const a = antipodal::fd64{2.0, 1.0};

    CHECK(a == 2.0);
    CHECK(2.0 == a);
    CHECK(a != 3.0);
    CHECK(3.0 != a);
    CHECK(a < 3.0);
    CHECK(1.0 < a);
    CHECK(a <= 2.0);
    CHECK(2.0 <= a);
    CHECK(a > 1.0);
    CHECK(3.0 > a);
    CHECK(a >= 2.0);
    CHECK(2.0 >= a);
}

// ============================================================================
// vec projections: value_of, d_value_of
// ============================================================================

TEST_CASE("fd - vec3 projections")
{
    using antipodal::fd64;
    antipodal::vec3<fd64> const v{fd64{1.0, 2.0}, fd64{4.0, 5.0}, fd64{7.0, 8.0}};

    auto const val = value_of(v);
    CHECK(val.x == 1.0);
    CHECK(val.y == 4.0);
    CHECK(val.z == 7.0);

    auto const d_val = d_value_of(v);
    CHECK(d_val.x == 2.0);
    CHECK(d_val.y == 5.0);
    CHECK(d_val.z == 8.0);
}

TEST_CASE("fd - vec2 projections")
{
    using antipodal::fd64;
    antipodal::vec2<fd64> const v{fd64{1.0, 2.0}, fd64{4.0, 5.0}};

    auto const val = value_of(v);
    CHECK(val.x == 1.0);
    CHECK(val.y == 4.0);

    auto const d_val = d_value_of(v);
    CHECK(d_val.x == 2.0);
    CHECK(d_val.y == 5.0);
}

// ============================================================================
// is_finite / is_nan / is_inf
// ============================================================================

TEST_CASE("fd - is_finite/is_nan/is_inf depend only on value")
{
    auto const nan_d = std::numeric_limits<double>::quiet_NaN();
    auto const inf_d = std::numeric_limits<double>::infinity();

    auto const finite = antipodal::fd64{1.0, 2.0};
    auto const nan_val = antipodal::fd64{nan_d, 1.0};
    auto const inf_val = antipodal::fd64{inf_d, 1.0};
    auto const neg_inf_val = antipodal::fd64{-inf_d, 1.0};

    CHECK(is_finite(finite));
    CHECK(!is_nan(finite));
    CHECK(!is_inf(finite));

    CHECK(!is_finite(nan_val));
    CHECK(is_nan(nan_val));
    CHECK(!is_inf(nan_val));

    CHECK(!is_finite(inf_val));
    CHECK(!is_nan(inf_val));
    CHECK(is_inf(inf_val));

    CHECK(!is_finite(neg_inf_val));
    CHECK(!is_nan(neg_inf_val));
    CHECK(is_inf(neg_inf_val));
}

// ============================================================================
// mixed ops with vec3<T> and vec3<fd>
// ============================================================================

TEST_CASE("fd - dot(vec3<fd>, vec3<T>)")
{
    using antipodal::fd64;
    antipodal::vec3<fd64> const v_fd{fd64{1.0, 0.5}, fd64{2.0, 0.25}, fd64{3.0, 0.75}};
    antipodal::dvec3 const v_dbl{2.0, 4.0, 4.0};

    auto const result = dot(v_fd, v_dbl);

    // dot value: 1*2 + 2*4 + 3*4 = 22
    CHECK(result.value == 22.0);
    // dot d_value: 0.5*2 + 0.25*4 + 0.75*4 = 5
    CHECK(result.d_value == 5.0);
}

TEST_CASE("fd - vec3<T> * fd")
{
    antipodal::dvec3 const v{1.0, 2.0, 4.0};
    auto const s = antipodal::fd64{2.0, 0.5};

    auto const result = v * s;

    CHECK(result.x.value == 2.0);
    CHECK(result.x.d_value == 0.5);

    CHECK(result.y.value == 4.0);
    CHECK(result.y.d_value == 1.0);

    CHECK(result.z.value == 8.0);
    CHECK(result.z.d_value == 2.0);
}

TEST_CASE("fd - vec3<T> + vec3<fd>")
{
    using antipodal::fd64;
    antipodal::dvec3 const p{1.0, 2.0, 3.0};
    antipodal::vec3<fd64> const v{fd64{1.0, 0.5}, fd64{2.0, 0.75}, fd64{3.0, 0.875}};

    auto const result = p + v;

    CHECK(result.x.value == 2.0);
    CHECK(result.x.d_value == 0.5);

    CHECK(result.y.value == 4.0);
    CHECK(result.y.d_value == 0.75);

    CHECK(result.z.value == 6.0);
    CHECK(result.z.d_value == 0.875);
}

TEST_CASE("fd - vec3<fd> - vec3<T>")
{
    using antipodal::fd64;
    antipodal::vec3<fd64> const p1{fd64{2.0, 0.5}, fd64{4.0, 0.75}, fd64{6.0, 0.875}};
    antipodal::dvec3 const p2{1.0, 2.0, 3.0};

    auto const result = p1 - p2;

    CHECK(result.x.value == 1.0);
    CHECK(result.x.d_value == 0.5);

    CHECK(result.y.value == 2.0);
    CHECK(result.y.d_value == 0.75);

    CHECK(result.z.value == 3.0);
    CHECK(result.z.d_value == 0.875);
}

TEST_CASE("fd - normalize_fd: constant direction has zero derivatives")
{
    // p(t) = (t, 2t, 2t), ||p|| = 3|t|, normalized = (1/3, 2/3, 2/3) for t > 0.
    // Since the normalized direction is constant in t, the derivative is 0.
    using antipodal::fd64;
    auto const t = fd64::input(1.0);
    antipodal::vec3<fd64> const v{t, 2.0 * t, 2.0 * t};
    auto const n = normalize_fd(v);

    CHECK(value_of(n).x == doctest::Approx(1.0 / 3.0).epsilon(1e-12));
    CHECK(value_of(n).y == doctest::Approx(2.0 / 3.0).epsilon(1e-12));
    CHECK(value_of(n).z == doctest::Approx(2.0 / 3.0).epsilon(1e-12));

    CHECK(d_value_of(n).x == doctest::Approx(0.0).epsilon(1e-10));
    CHECK(d_value_of(n).y == doctest::Approx(0.0).epsilon(1e-10));
    CHECK(d_value_of(n).z == doctest::Approx(0.0).epsilon(1e-10));
}

TEST_CASE("fd - normalize_fd: non-trivial derivatives at t=0 for v=(1,t,0)")
{
    // v(t) = (1, t, 0), ||v|| = sqrt(1+t^2)
    // n(t) = (1/sqrt(1+t^2), t/sqrt(1+t^2), 0)
    // At t=0: n = (1, 0, 0), n' = (0, 1, 0)
    using antipodal::fd64;
    auto const t = fd64::input(0.0);
    antipodal::vec3<fd64> const v{fd64::constant(1.0), t, fd64::constant(0.0)};
    auto const n = normalize_fd(v);

    CHECK(value_of(n).x == doctest::Approx(1.0).epsilon(1e-12));
    CHECK(value_of(n).y == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(value_of(n).z == doctest::Approx(0.0).epsilon(1e-12));

    CHECK(d_value_of(n).x == doctest::Approx(0.0).epsilon(1e-10));
    CHECK(d_value_of(n).y == doctest::Approx(1.0).epsilon(1e-10));
    CHECK(d_value_of(n).z == doctest::Approx(0.0).epsilon(1e-10));
}

// ============================================================================
// float sanity tests (fd32)
// ============================================================================

TEST_CASE("fd32 - basic operations")
{
    auto const x = antipodal::fd32::input(2.0f);

    auto const f = x * x;
    CHECK(f.value == doctest::Approx(4.0f).epsilon(1e-6f));
    CHECK(f.d_value == doctest::Approx(4.0f).epsilon(1e-6f));
}

TEST_CASE("fd32 - exp")
{
    auto const x = antipodal::fd32::input(1.0f);
    auto const result = exp(x);

    auto const e = std::exp(1.0f);
    CHECK(result.value == doctest::Approx(e).epsilon(1e-5f));
    CHECK(result.d_value == doctest::Approx(e).epsilon(1e-5f));
}

TEST_CASE("fd32 - log")
{
    auto const x = antipodal::fd32::input(2.0f);
    auto const result = log(x);

    CHECK(result.value == doctest::Approx(std::log(2.0f)).epsilon(1e-5f));
    CHECK(result.d_value == doctest::Approx(0.5f).epsilon(1e-5f));
}

TEST_CASE("fd32 - sin")
{
    auto const x = antipodal::fd32::input(0.5f);
    auto const result = sin(x);

    CHECK(result.value == doctest::Approx(std::sin(0.5f)).epsilon(1e-5f));
    CHECK(result.d_value == doctest::Approx(std::cos(0.5f)).epsilon(1e-5f));
}

TEST_CASE("fd32 - pow(x, 2)")
{
    auto const x = antipodal::fd32::input(3.0f);
    auto const result = pow(x, 2.0f);

    CHECK(result.value == doctest::Approx(9.0f).epsilon(1e-5f));
    CHECK(result.d_value == doctest::Approx(6.0f).epsilon(1e-5f));
}
