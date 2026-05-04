# The Antipodal Method

A header-only C++20 library for computing **3D generalized winding numbers** (GWN) on triangle meshes and parametric surfaces, accompanying the paper *"The Antipodal Method: Fast, Accurate, and Robust 3D Generalized Winding Numbers"*.

The GWN is a continuous, robust measure of how "inside" a point is with respect to a 3D surface, including surfaces that are open, self-intersecting, or non-manifold. 
It is the workhorse of many geometry-processing pipelines (insidedness queries, fast voxelization, repair, reconstruction, ...).

This library implements the antipodal formulation, which splits the GWN into:

1. an **integer** term — the signed number of ray–surface intersections along a chosen direction, and
2. a **fractional** term — a boundary integral expressed as a sum of signed spherical-triangle areas over the surface's projection onto the unit sphere.

That decomposition avoids expensive surface integrals and spherical arrangements, while remaining accurate to arbitrary precision. 
The algorithm scales from interactive single-point queries to high-throughput batch evaluation.

## API cheat sheet

All functions live in namespace `antipodal`. Single-point variants take a query
point; batch variants take a `Dispatcher&` plus matching input/output spans
(see [`gwn_mesh.hh`](src/antipodal/gwn_mesh.hh) for full Doxygen).

```cpp
// Fractional GWN (boundary integral; zero for closed meshes)
T    eval_gwnr_mesh_single_fractional(vec3<T> p, vec3<T> x0,
                                      span<weighted_segment3<T> const> boundary);
void eval_gwnr_mesh_batch_fractional (Dispatcher&,
                                      span<vec3<T> const> positions,
                                      span<T> out_wnrs,
                                      vec3<T> x0,
                                      span<weighted_segment3<T> const> boundary);

// Integer GWN (signed ray-surface crossing count)
int  eval_gwnr_mesh_single_integer(vec3<T> p, vec3<T> x0, Intersector const&);
void eval_gwnr_mesh_batch_integer (Dispatcher&, Intersector const&,
                                   span<vec3<T> const> positions,
                                   span<int> out_counts,
                                   vec3<T> x0);

// Full GWN (fractional + integer, fused per-point)
T    eval_gwnr_mesh_single(vec3<T> p, vec3<T> x0,
                           span<weighted_segment3<T> const> boundary,
                           Intersector const&);
void eval_gwnr_mesh_batch (Dispatcher&, Intersector const&,
                           span<vec3<T> const> positions,
                           span<T> out_wnrs,
                           vec3<T> x0,
                           span<weighted_segment3<T> const> boundary);

// Helpers (common.hh): build the open-boundary edge list from a triangle mesh
vector<weighted_segment3<T>> build_boundary_segments(span<vec3<T> const> vertices,
                                                    span<int const> indices);
```

**Dispatcher** is any duck-typed `parallel_for(int begin, int end, F&& f)`
backend; ships with:

| Type                        | Header                            | Gated on            |
|-----------------------------|-----------------------------------|---------------------|
| `SinglethreadDispatcher`    | `dispatcher.hh`                   | always available    |
| `PoolDispatcher`            | `dispatcher_threadpool.hh`        | always available    |
| `TbbDispatcher`             | `dispatcher_tbb.hh`               | `ANTIPODAL_HAS_TBB` |
| `OpenMPDispatcher`          | `dispatcher_openmp.hh`            | `_OPENMP`           |

**Intersector** is any duck-typed `signed_intersection_count(p, dir) const`
backend; ships with `NaiveIntersector<T>` (always) and `EmbreeIntersector`
(float-only, gated on `ANTIPODAL_HAS_EMBREE`).

Parametric-surface counterparts (`eval_gwnr_parametric_*`) are stubbed in
[`gwn_parametric.hh`](src/antipodal/gwn_parametric.hh) and will land in
subsequent commits.

## Dependencies

Both optional, **default ON**, both *user-provided* (the project does not bundle them):

- **TBB** — provides `antipodal::TbbDispatcher`. Disable with `-DANTIPODAL_WITH_TBB=OFF`.
- **Embree 4** — provides `antipodal::EmbreeIntersector` (ray-based integer term). Disable with `-DANTIPODAL_WITH_EMBREE=OFF`.

If an enabled dependency cannot be found, CMake configuration will fail with a `find_package` error — install the package or turn the option off.

`SinglethreadDispatcher`, `PoolDispatcher`, and `NaiveIntersector` are always available regardless of either option. `OpenMPDispatcher` is enabled automatically whenever the compiler defines `_OPENMP` (e.g. `-fopenmp` / `/openmp`); no CMake option is involved.

## Quickstart with vcpkg

If you don't already have TBB and Embree installed, [vcpkg](https://github.com/microsoft/vcpkg) is the easiest way to get them on Windows or Linux. The repo ships a [`vcpkg.json`](vcpkg.json) manifest, so the deps install automatically the first time you configure.

Bootstrap vcpkg once:

```sh
git clone https://github.com/microsoft/vcpkg C:/vcpkg
C:/vcpkg/bootstrap-vcpkg.bat
```

Then configure and build using the project's presets, pointing CMake at the vcpkg toolchain:

```sh
cmake --preset x64-windows-msvc-ninja-release ^
  -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build --preset release-msvc
```

(On Linux, replace the paths with `~/vcpkg/...`, use `./bootstrap-vcpkg.sh`, and pick the `x64-linux-clang-ninja-*` / `*-linux-clang` presets.)

## Build

The convenience wrapper `build.py` configures with the default preset
(`x64-windows-msvc-ninja-relwithdebinfo` / `x64-linux-clang-ninja-relwithdebinfo`),
threads the vcpkg toolchain through, and on Windows sources `vcvarsall.bat`
automatically:

```sh
python build.py        # configure + build
python build.py -t     # configure + build + run antipodal-tests
python build.py -p <preset>   # pick a different preset
```

Or invoke CMake directly:

```sh
cmake -S . -B build
cmake --build build
```

## Run tests

```sh
ctest --test-dir build --output-on-failure
```

## Use as a library

Via `add_subdirectory` or `FetchContent`:

```cmake
add_subdirectory(extern/antipodal)
target_link_libraries(my_app PRIVATE antipodal::antipodal)
```

Then in code:

```cpp
#include <antipodal/gwn_mesh.hh>
#include <antipodal/dispatcher_tbb.hh>

antipodal::TbbDispatcher disp;
antipodal::eval_gwnr_mesh_batch_fractional(disp, positions, out_wnrs, dir, boundary);
```
