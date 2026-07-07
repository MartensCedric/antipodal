// Standalone example: load an .obj triangle mesh, evaluate the mesh generalized
// winding number (GWN) on a 2D slice plane, and write the slice to a PNG.
//
// Usage:
//   antipodal-example-slice <input.obj> <output.png> <resolution>
//                           [slice_pos] [nx ny nz] [extent]
//
//   input.obj    path to an OBJ mesh (only `v` and `f` lines are read; polygonal
//                faces are fan-triangulated)
//   output.png   path of the PNG to write (square, resolution x resolution)
//   resolution   image side length in pixels, e.g. 256
//   slice_pos    relative position of the slice along the normal, in [0,1]
//                (0 = min extent, 1 = max extent). Default 0.5.
//   nx ny nz     slice-plane normal (need not be unit). Default 0 1 0.
//   extent       relative size of the (square) slice window: 1.0 is tight on the
//                mesh's larger in-plane dimension, 1.2 leaves >=20% slack.
//                Default 1.2.
//
// The winding number is mapped to a diverging colormap: blue is ~0 (outside),
// yellow trends to +1 (inside), and orange-red marks negative values. For the
// shipped `rotated_cube_open.obj` (a cube missing one face) the open side shows
// the characteristic smooth fractional-term transition rather than a hard edge.

#include <antipodal/dispatcher/dispatcher_threadpool.hh>
#include <antipodal/gwn_mesh.hh>
#include <antipodal/intersector/intersector.hh>
#include <antipodal/math/common.hh>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <span>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using antipodal::vec3;
using T = double;

namespace
{
// Minimal OBJ reader: parses `v x y z` and `f ...` lines only. Face tokens may be
// `v`, `v/vt`, `v//vn`, or `v/vt/vn`; we take the leading vertex index (1-based in
// OBJ, converted to 0-based) and fan-triangulate any polygon. All other lines are
// ignored. Not a general-purpose importer -- enough for simple demo meshes.
bool load_obj(std::string const& path, std::vector<vec3<T>>& verts, std::vector<int>& indices)
{
    std::ifstream in(path);
    if (!in)
        return false;

    std::string line;
    while (std::getline(in, line))
    {
        std::istringstream ls(line);
        std::string tag;
        ls >> tag;
        if (tag == "v")
        {
            vec3<T> p;
            ls >> p.x >> p.y >> p.z;
            verts.push_back(p);
        }
        else if (tag == "f")
        {
            std::vector<int> face;
            std::string tok;
            while (ls >> tok)
            {
                // keep only the vertex index (before the first '/')
                auto const slash = tok.find('/');
                std::string const vtok = (slash == std::string::npos) ? tok : tok.substr(0, slash);
                if (vtok.empty())
                    continue;
                int const vi = std::stoi(vtok); // 1-based, may be negative (relative)
                face.push_back(vi > 0 ? vi - 1 : int(verts.size()) + vi);
            }
            // fan triangulation
            for (std::size_t i = 2; i < face.size(); ++i)
            {
                indices.push_back(face[0]);
                indices.push_back(face[i - 1]);
                indices.push_back(face[i]);
            }
        }
    }
    return true;
}

// Map a winding-number value to an RGB triplet (diverging, clamped to [-1, 1]):
//   0 -> #2F5994 (blue, outside)   1 -> #F3E676 (yellow, inside)
//                                 -1 -> #D2552E (orange-red)
void colormap(T w, unsigned char& r, unsigned char& g, unsigned char& b)
{
    struct rgb
    {
        T r, g, b;
    };
    constexpr rgb c_out{47, 89, 148};    // #2F5994 at w = 0
    constexpr rgb c_in{243, 230, 118};   // #F3E676 at w = 1
    constexpr rgb c_neg{210, 85, 46};    // #D2552E at w = -1

    w = std::clamp(w, T(-1), T(1));
    rgb const& target = (w >= 0) ? c_in : c_neg;
    T const t = std::abs(w); // blend from c_out toward the target
    auto to_byte = [](T v) { return (unsigned char)std::lround(std::clamp(v, T(0), T(255))); };
    r = to_byte(c_out.r + (target.r - c_out.r) * t);
    g = to_byte(c_out.g + (target.g - c_out.g) * t);
    b = to_byte(c_out.b + (target.b - c_out.b) * t);
}
} // namespace

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::fprintf(stderr,
                     "usage: %s <input.obj> <output.png> <resolution> "
                     "[slice_pos=0.5] [nx ny nz = 0 1 0] [extent=1.2]\n",
                     argv[0]);
        return 1;
    }

    std::string const obj_path = argv[1];
    std::string const png_path = argv[2];
    int const res = std::atoi(argv[3]);
    T const slice_pos = (argc > 4) ? std::atof(argv[4]) : T(0.5);
    vec3<T> normal = (argc > 7) ? vec3<T>{std::atof(argv[5]), std::atof(argv[6]), std::atof(argv[7])} : vec3<T>{0, 1, 0};
    T const extent = (argc > 8) ? std::atof(argv[8]) : T(1.2);

    if (res <= 0)
    {
        std::fprintf(stderr, "error: resolution must be positive\n");
        return 1;
    }

    std::vector<vec3<T>> verts;
    std::vector<int> indices;
    if (!load_obj(obj_path, verts, indices))
    {
        std::fprintf(stderr, "error: could not open '%s'\n", obj_path.c_str());
        return 1;
    }
    if (verts.empty() || indices.empty())
    {
        std::fprintf(stderr, "error: no triangles loaded from '%s'\n", obj_path.c_str());
        return 1;
    }
    std::printf("loaded %zu vertices, %zu triangles\n", verts.size(), indices.size() / 3);

    // In-plane orthonormal basis (u, v) for the slice, with n as the plane normal.
    vec3<T> const n = antipodal::normalize(normal);
    vec3<T> const helper = (std::abs(n.x) < T(0.9)) ? vec3<T>{1, 0, 0} : vec3<T>{0, 1, 0};
    vec3<T> const u = antipodal::normalize(antipodal::cross(helper, n));
    vec3<T> const v = antipodal::cross(n, u); // already unit: n, u orthonormal

    // Project all vertices onto (n, u, v) to find the slice offset range and the
    // in-plane 2D bounding box.
    T dmin = std::numeric_limits<T>::max(), dmax = std::numeric_limits<T>::lowest();
    T amin = dmin, amax = dmax, bmin = dmin, bmax = dmax;
    for (auto const& p : verts)
    {
        T const d = antipodal::dot(p, n);
        T const a = antipodal::dot(p, u);
        T const b = antipodal::dot(p, v);
        dmin = std::min(dmin, d), dmax = std::max(dmax, d);
        amin = std::min(amin, a), amax = std::max(amax, a);
        bmin = std::min(bmin, b), bmax = std::max(bmax, b);
    }

    T const off = dmin + slice_pos * (dmax - dmin); // plane: dot(x, n) == off
    vec3<T> const p0 = off * n;                     // an anchor point on the plane

    T const ac = (amin + amax) / 2, bc = (bmin + bmax) / 2;
    T const larger = std::max(amax - amin, bmax - bmin);
    T const side = extent * (larger > T(0) ? larger : T(1)); // guard degenerate meshes

    // Build the grid of query points on the slice plane.
    std::vector<vec3<T>> points(std::size_t(res) * res);
    for (int j = 0; j < res; ++j)
    {
        T const bv = bc + ((j + T(0.5)) / res - T(0.5)) * side;
        for (int i = 0; i < res; ++i)
        {
            T const au = ac + ((i + T(0.5)) / res - T(0.5)) * side;
            points[std::size_t(j) * res + i] = p0 + au * u + bv * v;
        }
    }

    // Evaluate the full GWN (fractional boundary term + integer crossing term).
    // For simplicity this example uses the always-available NaiveIntersector and
    // PoolDispatcher. In production you'd typically use a faster intersector
    // (e.g. EmbreeIntersector for the integer ray-crossing term) and TbbDispatcher
    // for the parallel batch — see intersector_embree.hh / dispatcher_tbb.hh. Note
    // Embree's geometry API is float-only, so going that route means running the
    // example with T = float; we keep T = double here to stay simple.
    antipodal::NaiveIntersector<T> isect{verts, indices};
    auto const boundary = antipodal::build_boundary_segments<T>(verts, indices);
    auto const x0 = antipodal::normalize(vec3<T>{T(0.31), T(0.57), T(0.76)}); // any fixed unit dir
    std::printf("open boundary: %zu segments\n", boundary.size());

    std::vector<T> wn(points.size());
    antipodal::PoolDispatcher disp{int(std::max(1u, std::thread::hardware_concurrency()))};
    antipodal::eval_gwnr_mesh_batch(disp, isect, std::span<vec3<T> const>{points}, std::span<T>{wn}, x0,
                                    std::span<antipodal::weighted_segment3<T> const>{boundary});

    // Colorize. Flip rows so +v points up in the image (row 0 = top = max v).
    std::vector<unsigned char> rgb(std::size_t(res) * res * 3);
    for (int j = 0; j < res; ++j)
    {
        int const src_row = res - 1 - j;
        for (int i = 0; i < res; ++i)
        {
            auto const w = wn[std::size_t(src_row) * res + i];
            auto* px = &rgb[(std::size_t(j) * res + i) * 3];
            colormap(w, px[0], px[1], px[2]);
        }
    }

    if (!stbi_write_png(png_path.c_str(), res, res, 3, rgb.data(), res * 3))
    {
        std::fprintf(stderr, "error: failed to write '%s'\n", png_path.c_str());
        return 1;
    }
    std::printf("wrote %dx%d slice to '%s'\n", res, res, png_path.c_str());
    return 0;
}
