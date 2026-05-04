#pragma once

namespace antipodal
{
/// Common configuration for adaptive 1D integrators in <antipodal/math/>
/// (`integrate_gk15`, `integrate_gl7`, ...). Sharing one config struct lets
/// callers be parameterised on the integrator function.
///
/// Per-method special fields may be added here later — this struct is the
/// canonical extension point.
struct integrate_config
{
    /// Minimum recursion depth before tolerance-based acceptance kicks in.
    /// 0 means "accept the root segment if it already converges".
    int min_depth = 0;

    /// Hard cap on recursion depth.
    int max_depth = 10;

    /// Local error tolerance, halved at each refinement level.
    double tol = 1e-6;
};

/// Mode shared by every `integrate_*` function in <antipodal/math/>.
///
/// - `evaluate`     — full adaptive integration; returns the integral.
/// - `preinvoke`    — walk the recursion tree to `cfg.max_depth` and call `fun`
///                    at every node without computing the integral. Useful for
///                    pre-populating an external cache that a later `evaluate`
///                    pass will hit by index.
///                    `cfg.min_depth` and `cfg.tol` are ignored.
/// - `single_level` — evaluate the rule once on `[a, b]` with no recursion and
///                    no error estimate. The fastest path; the recursion lambda
///                    is `if constexpr`'d out entirely.
///                    `cfg.min_depth`, `cfg.max_depth`, `cfg.tol`, `out_err`,
///                    and `out_intervals` are all ignored.
enum class integrate_mode
{
    evaluate,
    preinvoke,
    single_level,
};
} // namespace antipodal
