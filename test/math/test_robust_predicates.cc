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

#if defined(__SIZEOF_INT128__) && !defined(_MSC_VER)
TEST_CASE("robust::int128: matches the native __int128")
{
    __extension__ typedef __int128 native;
    auto const same = [](int128 a, native b) { return a.lo == std::uint64_t(b) && a.hi == std::int64_t(b >> 64); };

    constexpr auto lo = std::numeric_limits<std::int64_t>::min();
    constexpr auto hi = std::numeric_limits<std::int64_t>::max();
    std::int64_t const edge[] = {0, 1, -1, 2, -2, lo, lo + 1, hi, hi - 1, std::int64_t(1) << 32, -(std::int64_t(1) << 32)};

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
