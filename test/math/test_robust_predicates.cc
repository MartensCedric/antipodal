#include "doctest.h"

#include <antipodal/math/robust_predicates.hh>

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>

using namespace antipodal;
using namespace antipodal::robust;

// ivec3 keeps returning i64 like before.
static_assert(std::is_same_v<decltype(atan2_num(ivec3{}, ivec3{}, ivec3{})), i64>);
static_assert(std::is_same_v<decltype(atan2_num(lvec3{}, lvec3{}, lvec3{})), i128>);

TEST_CASE("robust predicates: exact at 50-bit coordinates where int64 determinants would overflow")
{
    constexpr std::int64_t A = std::int64_t(1) << 50;

    // Products here are around 2^98, far beyond int64.
    lvec3 const q0{0, 0, 0};
    lvec3 const q1{0, A, A - 2};
    lvec3 const on{0, A / 2, A / 2 - 1};
    CHECK(atan2_num(on, q0, q1) == 0);
    // on the edge, so the perturbation decides
    CHECK(edge_sign(on, q0, q1) == -1);
    CHECK(sign_num_perturbed(on, q0, q1) == 1);

    // one unit off the edge on each side
    lvec3 const above{0, A / 2, A / 2};
    lvec3 const below{0, A / 2, A / 2 - 2};
    CHECK(double(atan2_num(above, q0, q1)) == -double(A));
    CHECK(double(atan2_num(below, q0, q1)) == double(A));
    CHECK(edge_sign(above, q0, q1) == 1);
    CHECK(edge_sign(below, q0, q1) == -1);

    lvec3 const q2{0, 0, A};
    CHECK(inside_tri_sign(lvec3{0, 1, 1}, q0, q1, q2) != 0);
    CHECK(inside_tri_sign(lvec3{0, -1, 1}, q0, q1, q2) == 0);
}

TEST_CASE("robust predicates: int32 and int64 coordinates agree on the 20-bit grid")
{
    std::mt19937_64 rng(1234);
    std::uniform_int_distribution<std::int32_t> coord(-(1 << 20), 1 << 20);
    auto const rand_i = [&] { return ivec3{coord(rng), coord(rng), coord(rng)}; };
    auto const widen = [](ivec3 v) { return lvec3{v.x, v.y, v.z}; };

    for (int i = 0; i < 10000; ++i)
    {
        ivec3 const q = rand_i(), a = rand_i(), b = rand_i(), c = rand_i();
        CHECK(edge_sign(q, a, b) == edge_sign(widen(q), widen(a), widen(b)));
        CHECK(double(atan2_num(q, a, b)) == double(atan2_num(widen(q), widen(a), widen(b))));
        CHECK(inside_tri_sign(q, a, b, c) == inside_tri_sign(widen(q), widen(a), widen(b), widen(c)));
    }
}

TEST_CASE("robust::int128: portable arithmetic")
{
    constexpr auto lo = std::numeric_limits<std::int64_t>::min();
    constexpr auto hi = std::numeric_limits<std::int64_t>::max();

    CHECK(int128::mul(3, -4) == int128(-12));
    CHECK(int128::mul(-3, -4) == int128(12));
    CHECK(int128::mul(0, lo) == int128(0));
    CHECK(int128::mul(hi, hi) > int128(hi));
    CHECK(int128::mul(lo, hi) < int128(lo));
    CHECK(int128::mul(hi, hi) - int128::mul(hi, hi) == int128(0));
    CHECK(-int128(5) == int128(-5));
    CHECK(double(int128::mul(std::int64_t(1) << 40, std::int64_t(1) << 40)) == std::ldexp(1.0, 80));
    CHECK(double(int128::mul(-(std::int64_t(1) << 40), std::int64_t(1) << 40)) == -std::ldexp(1.0, 80));
    // small negative results must stay negative
    CHECK(double(int128::mul(hi, 2) - int128::mul(hi, 2) - int128(5)) == -5.0);
}

TEST_CASE("robust::int128: mul matches the portable schoolbook multiply")
{
    // On MSVC mul() is the hardware instruction; elsewhere both are the same code, and this is cheap.
    constexpr auto lo = std::numeric_limits<std::int64_t>::min();
    constexpr auto hi = std::numeric_limits<std::int64_t>::max();
    std::int64_t const edge[]
        = {0, 1, -1, 2, -2, lo, lo + 1, hi, hi - 1, std::int64_t(1) << 32, -(std::int64_t(1) << 32)};
    for (auto a : edge)
        for (auto b : edge)
            CHECK(int128::mul(a, b) == int128::mul_portable(a, b));

    std::mt19937_64 rng(7);
    for (int i = 0; i < 100000; ++i)
    {
        auto const a = std::int64_t(rng()), b = std::int64_t(rng());
        CHECK(int128::mul(a, b) == int128::mul_portable(a, b));
    }

    static_assert(int128::mul(-3, 4) == int128(-12), "mul stays usable in constant expressions");
}

#if defined(__SIZEOF_INT128__) && !defined(_MSC_VER)
TEST_CASE("robust::int128: matches the native __int128")
{
    __extension__ typedef __int128 native;
    auto const same = [](int128 a, native b) { return a.lo == std::uint64_t(b) && a.hi == std::int64_t(b >> 64); };

    constexpr auto lo = std::numeric_limits<std::int64_t>::min();
    constexpr auto hi = std::numeric_limits<std::int64_t>::max();
    std::int64_t const edge[]
        = {0, 1, -1, 2, -2, lo, lo + 1, hi, hi - 1, std::int64_t(1) << 32, -(std::int64_t(1) << 32)};

    auto const check = [&](std::int64_t a, std::int64_t b, std::int64_t c, std::int64_t d)
    {
        int128 const p = int128::mul(a, b) - int128::mul(c, d);
        native const n = native(a) * native(b) - native(c) * native(d);
        CHECK(same(p, n));
        CHECK((p > 0) == (n > 0));
        CHECK((p == 0) == (n == 0));
        // rounding can differ in the last bit
        CHECK(double(p) == doctest::Approx(double(n)).epsilon(1e-15));
    };

    for (auto a : edge)
        for (auto b : edge)
            check(a, b, 1, 1);

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<std::int64_t> any(lo / 2, hi / 2);
    for (int i = 0; i < 100000; ++i)
        check(any(rng), any(rng), any(rng), any(rng));
}
#endif

static_assert(std::is_same_v<decltype(atan2_num(dvec3{}, dvec3{}, dvec3{})), double>);
static_assert(std::is_same_v<default_precision<float>, int64_grid>);
static_assert(std::is_same_v<default_precision<double>, exact_float>);

TEST_CASE("robust predicates: exact_float signs match exact integers, including collinear and off-by-one cases")
{
    // Points are integers times 2^-20, so each double is exact and the integer edge_det on lvec3 is the reference.
    // Mixing magnitudes (around 2^50 and around 2^10) makes the differences round in double,
    // and the collinear and one-unit-off cases are exactly the ones the double filter cannot decide.
    std::mt19937_64 rng(99);
    std::uniform_int_distribution<std::int64_t> big(-(std::int64_t(1) << 50), std::int64_t(1) << 50);
    std::uniform_int_distribution<std::int64_t> small(-1024, 1024);
    auto const to_d = [](lvec3 v)
    { return dvec3{std::ldexp(double(v.x), -20), std::ldexp(double(v.y), -20), std::ldexp(double(v.z), -20)}; };

    int fallbacks = 0;
    for (int i = 0; i < 100000; ++i)
    {
        lvec3 const q0{0, big(rng), big(rng)};
        lvec3 const dir{0, small(rng), small(rng)};
        lvec3 const q1{0, q0.y + dir.y, q0.z + dir.z};
        std::int64_t const k = small(rng);
        lvec3 q{0, q0.y + k * dir.y, q0.z + k * dir.z}; // on the line through q0, q1
        int const nudge = int(rng() % 3) - 1;           // and then 0 or one unit off it
        q.y += nudge;
        if (i % 4 == 0)
            q = lvec3{0, small(rng), small(rng)}; // a small query far from a big edge

        int const ref = edge_sign(q, q0, q1);
        int const got = edge_sign(to_d(q), to_d(q0), to_d(q1));
        CHECK(got == ref);
        CHECK((atan2_num(to_d(q), to_d(q0), to_d(q1)) > 0) == (atan2_num(q, q0, q1) > 0));
        CHECK((atan2_num(to_d(q), to_d(q0), to_d(q1)) == 0) == (atan2_num(q, q0, q1) == 0));
        if (edge_det(q, q0, q1) == 0)
            ++fallbacks;
    }
    CHECK(fallbacks > 1000); // the exactly-collinear case really was exercised
}

TEST_CASE("robust predicates: exact_float triangle weights carry the exact sign")
{
    // A sliver far from the origin: from q1 the edges to q2 and back to q0 differ in slope by 2^-21,
    // and q sits strictly between them, so it is inside and every weight must share the triangle's sign.
    // All coordinates are exact doubles (they span at most 52 bits).
    double const b = std::ldexp(1.0, 28);
    double const t = std::ldexp(1.0, -21);
    dvec3 const q0{0, b, b};
    dvec3 const q1{0, b + 2, b + 1};
    dvec3 const q2{0, b + 4, b + 2 + 2 * t};
    dvec3 const q{0, b + 3, b + 1.5 + 1.25 * t}; // between z = b+1.5+t (edge q1q2) and z = b+1.5+1.5t (edge q0q2)

    auto const hit = project_into_tri(q, q0, q1, q2);
    REQUIRE(hit.sign != 0);
    CHECK(hit.w0 * hit.sign > 0);
    CHECK(hit.w1 * hit.sign > 0);
    CHECK(hit.w2 * hit.sign > 0);

    // the same triangle with q just outside the edge q1q2 is rejected
    CHECK(inside_tri_sign(dvec3{0, b + 3, b + 1.5 + 0.75 * t}, q0, q1, q2) == 0);
}
