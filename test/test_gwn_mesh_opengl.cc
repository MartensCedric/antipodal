#include "doctest.h"

#if defined(ANTIPODAL_HAS_OPENGL) && ANTIPODAL_HAS_OPENGL

#include <antipodal/dispatcher/dispatcher.hh>
#include <antipodal/gwn_mesh.hh>
#include <antipodal/intersector/intersector.hh>
#include <antipodal/math/common.hh>
#include <antipodal/gwn_mesh_opengl.hh>
#include <antipodal/opengl/gl_headless_context.hh>

#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#if defined(ANTIPODAL_HAS_EMBREE) && ANTIPODAL_HAS_EMBREE
#include <antipodal/intersector/intersector_embree.hh>
#endif
#if defined(ANTIPODAL_HAS_TBB) && ANTIPODAL_HAS_TBB
#include <antipodal/dispatcher/dispatcher_tbb.hh>
#endif

namespace
{
using antipodal::fvec3;

std::vector<fvec3> unit_cube_vertices()
{
    return {
        fvec3{-0.5f, -0.5f, -0.5f}, fvec3{+0.5f, -0.5f, -0.5f}, fvec3{-0.5f, +0.5f, -0.5f}, fvec3{+0.5f, +0.5f, -0.5f},
        fvec3{-0.5f, -0.5f, +0.5f}, fvec3{+0.5f, -0.5f, +0.5f}, fvec3{-0.5f, +0.5f, +0.5f}, fvec3{+0.5f, +0.5f, +0.5f},
    };
}

std::vector<int> unit_cube_indices_closed()
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

// Cube with the +Y face removed (open boundary along the top rim).
std::vector<int> unit_cube_indices_open_y()
{
    return {
        0, 2, 1, 1, 2, 3, //
        4, 5, 6, 6, 5, 7, //
        0, 4, 2, 2, 4, 6, //
        1, 3, 5, 5, 3, 7, //
        0, 1, 4, 4, 1, 5, //
    };
}

// Squared distance from `p` to triangle (a, b, c) (Ericson, closest-point).
float point_tri_dist_sqr(fvec3 p, fvec3 a, fvec3 b, fvec3 c)
{
    using namespace antipodal;
    auto const ab = b - a;
    auto const ac = c - a;
    auto const ap = p - a;
    float const d1 = dot(ab, ap);
    float const d2 = dot(ac, ap);
    if (d1 <= 0 && d2 <= 0)
        return length_sqr(ap);
    auto const bp = p - b;
    float const d3 = dot(ab, bp);
    float const d4 = dot(ac, bp);
    if (d3 >= 0 && d4 <= d3)
        return length_sqr(bp);
    float const vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0)
        return length_sqr(p - (a + ab * (d1 / (d1 - d3))));
    auto const cp = p - c;
    float const d5 = dot(ab, cp);
    float const d6 = dot(ac, cp);
    if (d6 >= 0 && d5 <= d6)
        return length_sqr(cp);
    float const vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0)
        return length_sqr(p - (a + ac * (d2 / (d2 - d6))));
    float const va = d3 * d6 - d5 * d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0)
        return length_sqr(p - (b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))));
    float const denom = 1.0f / (va + vb + vc);
    return length_sqr(p - (a + ab * (vb * denom) + ac * (vc * denom)));
}

// True when `p` is within `skin` of the mesh surface (any triangle, including edges and the open rim).
// There the GPU rasterization and the exact CPU ray cast may legitimately disagree by a full ±1 in the integer term (the crossing is snapped to a grid cell).
// Those cells are excluded from the check.
bool near_mesh_surface(fvec3 p, std::vector<fvec3> const& verts, std::vector<int> const& idx, float skin)
{
    float const skin2 = skin * skin;
    for (std::size_t t = 0; t < idx.size() / 3; ++t)
        if (point_tri_dist_sqr(p, verts[idx[3 * t]], verts[idx[3 * t + 1]], verts[idx[3 * t + 2]]) < skin2)
            return true;
    return false;
}

// A fixed right-handed orthonormal frame, generically rotated off the axes.
// The grid is laid out along this frame so integration rays do not align with the cube's axis-aligned faces / shared face diagonals.
// That alignment makes the rasterizer double-count along shared triangle edges, which is exactly why the repo's demo mesh is a *rotated* cube.
struct Basis
{
    fvec3 ex;
    fvec3 ey;
    fvec3 ez;
};

Basis rotated_basis()
{
    using namespace antipodal;
    auto const ex = normalize(fvec3{0.60f, 0.70f, 0.39f});
    auto const ez = normalize(cross(ex, fvec3{-0.40f, 0.20f, 0.90f}));
    auto const ey = cross(ez, ex); // unit-length, completes a right-handed frame
    return {ex, ey, ez};
}
} // namespace

TEST_CASE("OpenGLMeshGwn: dense grid roughly matches the Embree+TBB float CPU path")
{
    using namespace antipodal;

    auto ctx = HeadlessGlContext::create();
    if (!ctx)
    {
        WARN_MESSAGE(false, "no headless OpenGL 4.3 context available — skipping GPU test");
        return;
    }

    OpenGLMeshGwn gpu(ctx->loader());

    auto const basis = rotated_basis();

    // Lays out an nx*ny*nz grid along the rotated frame `basis` (cell step `e`), centered on the origin.
    // Evaluates it on the GPU and against the CPU baseline, then checks every cell that is not too close to the surface.
    auto check_grid = [&](std::vector<fvec3> const& verts, std::vector<int> const& idx,
                          std::vector<weighted_fsegment3> const& boundary, float e, int nx, int ny, int nz)
    {
        auto const cells = std::size_t(nx) * std::size_t(ny) * std::size_t(nz);

        // Grid basis along the rotated frame.
        // A single-cell axis still gets a finite step; its magnitude does not affect a 1-cell integration axis.
        fvec3 const dx = basis.ex * e;
        fvec3 const dy = basis.ey * e;
        fvec3 const dz = basis.ez * e;
        fvec3 const start = dx * (-0.5f * float(nx - 1)) + dy * (-0.5f * float(ny - 1)) + dz * (-0.5f * float(nz - 1));

        // Grid lattice points in the caller's ((iz*ny)+iy)*nx+ix ordering.
        std::vector<fvec3> positions(cells);
        for (int iz = 0; iz < nz; ++iz)
            for (int iy = 0; iy < ny; ++iy)
                for (int ix = 0; ix < nx; ++ix)
                {
                    auto const p = start + dx * float(ix) + dy * float(iy) + dz * float(iz);
                    positions[(std::size_t(iz) * ny + iy) * nx + ix] = p;
                }

        // GPU evaluation.
        gpu.set_scene(verts, idx, boundary);
        std::vector<float> got(cells);
        gpu.eval_grid(start, dx, dy, dz, nx, ny, nz, std::span<float>{got});

        // CPU baseline: the Embree+TBB float path (falls back when unavailable).
        // The full GWN is invariant to x0, so a non-axis-aligned direction keeps the Embree ray off the cube's shared face diagonals (where Möller-Trumbore would double-count).
        // That is a CPU-side artifact, independent of the GPU's own (axis-aligned) integration.
        auto const x0 = normalize(fvec3{0.31f, 0.57f, 0.76f});
        std::span<weighted_fsegment3 const> const bspan{boundary};
        std::vector<float> ref(cells);
#if defined(ANTIPODAL_HAS_EMBREE) && ANTIPODAL_HAS_EMBREE
        EmbreeIntersector intersector{std::span<fvec3 const>{verts}, std::span<int const>{idx}};
#else
        NaiveIntersector<float> intersector{std::span<fvec3 const>{verts}, std::span<int const>{idx}};
#endif
#if defined(ANTIPODAL_HAS_TBB) && ANTIPODAL_HAS_TBB
        TbbDispatcher disp;
#else
        SinglethreadDispatcher disp;
#endif
        eval_gwnr_mesh_batch(disp, intersector, std::span<fvec3 const>{positions}, std::span<float>{ref}, x0, bspan);

        float const tol = 3e-3f;
        // Skip cells within ~2 grid steps of the surface: there the rasterized crossing (snapped to a cell) legitimately differs from the exact ray.
        float const skin = 2.0f * e;
        int checked = 0;
        int mismatches = 0;
        float max_err = 0.0f;
        for (std::size_t i = 0; i < cells; ++i)
        {
            if (near_mesh_surface(positions[i], verts, idx, skin))
                continue;
            ++checked;
            float const err = std::abs(got[i] - ref[i]);
            max_err = std::max(max_err, err);
            if (err > tol)
                ++mismatches;
        }

        CHECK(checked > 0);
        CHECK(mismatches == 0);
        CHECK(max_err < tol);
    };

    auto const verts = unit_cube_vertices();

    SUBCASE("closed cube, full 3D grid")
    {
        auto const idx = unit_cube_indices_closed();
        std::vector<weighted_fsegment3> const boundary; // closed -> no boundary
        check_grid(verts, idx, boundary, 0.15f, 20, 20, 20);
    }

    SUBCASE("open cube (+Y removed) with boundary correction, full 3D grid")
    {
        auto const idx = unit_cube_indices_open_y();
        auto const boundary = build_boundary_segments<float>(std::span<fvec3 const>{verts}, std::span<int const>{idx});
        check_grid(verts, idx, boundary, 0.15f, 20, 20, 20);
    }

    SUBCASE("2D slice (ny = 1) exercises the skip-integrate fast path")
    {
        auto const idx = unit_cube_indices_closed();
        std::vector<weighted_fsegment3> const boundary;
        // ny = 1 -> the integration axis is y (a single cell), so the prefix-sum (integrate) pass is skipped.
        // The slice cuts through the cube center.
        check_grid(verts, idx, boundary, 0.13f, 24, 1, 24);
    }

    SUBCASE("smallest axis (z) exercises the axis permutation")
    {
        auto const idx = unit_cube_indices_closed();
        std::vector<weighted_fsegment3> const boundary;
        // Only 6 cells along z -> z becomes the integration axis, exercising the permute/un-permute path.
        // z = 0 stays an interior slab.
        check_grid(verts, idx, boundary, 0.15f, 20, 20, 6);
    }
}

#endif
