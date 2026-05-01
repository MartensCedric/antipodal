#pragma once

#include <cstdint>
#include <random>

namespace antipodal_test
{
// Central finite difference, first derivative.
template <class F>
double fd_first(F&& f, double x, double h = 1e-5)
{
    return (f(x + h) - f(x - h)) / (2 * h);
}

// Central finite difference, second derivative.
template <class F>
double fd_second(F&& f, double x, double h = 1e-4)
{
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h);
}

inline double uniform(std::mt19937_64& rng, double lo, double hi)
{
    return std::uniform_real_distribution<double>(lo, hi)(rng);
}

// Fixed-seed fuzz loop — emulates the project's old FUZZ_TEST harness.
// `body(rng)` is invoked iters times with a deterministic, seeded rng.
template <class F>
void fuzz_run(std::uint64_t seed, F&& body, int iters = 10000)
{
    std::mt19937_64 rng(seed);
    for (int i = 0; i < iters; ++i)
        body(rng);
}
} // namespace antipodal_test
