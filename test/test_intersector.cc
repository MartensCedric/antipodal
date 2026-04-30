#include "doctest.h"

#include <vector>

#include <antipodal/common.hh>
#include <antipodal/intersector/intersector.hh>
#include <antipodal/intersector/intersector_embree.hh>

namespace
{
// Unit cube [-0.5, 0.5]^3 as 8 vertices + 12 triangles, all outward-facing
// (CCW when viewed from outside, so cross(e1, e2) gives the outward normal).
template <class T>
std::vector<antipodal::vec3<T>> unit_cube_vertices()
{
    using V = antipodal::vec3<T>;
    return {
        V{T(-0.5), T(-0.5), T(-0.5)}, // 0
        V{T(+0.5), T(-0.5), T(-0.5)}, // 1
        V{T(-0.5), T(+0.5), T(-0.5)}, // 2
        V{T(+0.5), T(+0.5), T(-0.5)}, // 3
        V{T(-0.5), T(-0.5), T(+0.5)}, // 4
        V{T(+0.5), T(-0.5), T(+0.5)}, // 5
        V{T(-0.5), T(+0.5), T(+0.5)}, // 6
        V{T(+0.5), T(+0.5), T(+0.5)}, // 7
    };
}

inline std::vector<int> unit_cube_indices()
{
    return {
        // -Z face (n = -z)
        0, 2, 1, 1, 2, 3,
        // +Z face (n = +z)
        4, 5, 6, 6, 5, 7,
        // -X face (n = -x)
        0, 4, 2, 2, 4, 6,
        // +X face (n = +x)
        1, 3, 5, 5, 3, 7,
        // -Y face (n = -y)
        0, 1, 4, 4, 1, 5,
        // +Y face (n = +y)
        2, 6, 3, 3, 6, 7,
    };
}
} // namespace

TEST_CASE_TEMPLATE("NaiveIntersector: closed cube — inside has signed count 1, outside has 0", T, float, double)
{
    using namespace antipodal;
    auto const verts = unit_cube_vertices<T>();
    auto const idx = unit_cube_indices();
    NaiveIntersector<T> intersector{verts, idx};

    // Generic (non-axis-aligned) directions only — NaiveIntersector is plain
    // Möller-Trumbore, which double-counts when a ray hits a triangle edge or
    // vertex exactly. Axis-aligned rays from the cube center land on the
    // diagonal shared by the two triangles of each face, which is exactly that
    // degenerate case; we deliberately steer clear of it here.
    vec3<T> const dirs[] = {
        {T(0.31), T(0.57), T(0.76)},
        {T(-0.6), T(0.4), T(-0.69)},
        {T(0.83), T(-0.21), T(0.51)},
    };

    for (auto const& d : dirs)
    {
        // Inside: each ray exits through exactly one face (outward-pointing).
        CHECK(intersector.signed_intersection_count(vec3<T>{T(0), T(0), T(0)}, d) == 1);
        CHECK(intersector.signed_intersection_count(vec3<T>{T(0.1), T(-0.2), T(0.3)}, d) == 1);

        // Outside: ray enters and exits in pairs (or misses entirely) → net 0.
        CHECK(intersector.signed_intersection_count(vec3<T>{T(2), T(0), T(0)}, d) == 0);
        CHECK(intersector.signed_intersection_count(vec3<T>{T(-2), T(-2), T(-2)}, d) == 0);
    }
}

#if defined(ANTIPODAL_HAS_EMBREE) && ANTIPODAL_HAS_EMBREE
TEST_CASE("EmbreeIntersector: agrees with NaiveIntersector<float> on the unit cube")
{
    using namespace antipodal;
    auto const verts = unit_cube_vertices<float>();
    auto const idx = unit_cube_indices();

    NaiveIntersector<float> naive{verts, idx};
    EmbreeIntersector embree{verts, idx};

    fvec3 const samples[] = {
        {0.0f, 0.0f, 0.0f},
        {0.1f, -0.2f, 0.3f},
        {0.49f, 0.0f, 0.0f}, // near the +X face
        {2.0f, 0.0f, 0.0f},
        {-2.0f, -2.0f, -2.0f},
        {0.0f, 1.5f, 0.0f},
    };
    fvec3 const dirs[] = {
        {1.0f, 0.0f, 0.0f},
        {0.31f, 0.57f, 0.76f},
        {-0.6f, 0.4f, -0.69f},
    };

    for (auto const& p : samples)
        for (auto const& d : dirs)
            CHECK(naive.signed_intersection_count(p, d) == embree.signed_intersection_count(p, d));
}
#endif
