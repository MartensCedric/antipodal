#pragma once

// Raw-OpenGL compute-shader evaluator for dense grids of mesh GWN values (`OpenGLMeshGwn`).
//
// A current OpenGL 4.3 context must be active on the calling thread for the object's whole lifetime.
// Entry points are resolved through a caller-supplied loader `void* (*)(char const*)` (wglGetProcAddress, eglGetProcAddress, glfwGetProcAddress, SDL_GL_GetProcAddress).
// The header declares its own GL types / enums / entry points, so it needs no glad / glew / glfw and never collides with a consumer's own <GL/gl.h>.
// The header links nothing itself; gl_headless_context.hh supplies an optional off-screen context.
//
// Header-only port of the research `volume-opengl` evaluator onto plain OpenGL 4.3 compute shaders.
// The four passes mirror the antipodal decomposition:
//   1. init      — clear the per-cell integer accumulator.
//   2. rasterize — scatter signed ray-crossings of every triangle.
//   3. integrate — prefix-sum those crossings into the integer GWN term.
//   4. wnr       — add the fractional boundary integral and store the full GWN.
// The fractional pass uses the unnormalized Van Oosterom-Strackee form of `signed_spherical_tri_area_half_unorm` (math/common.hh), so the hot loop needs no per-edge `normalize()`.
//
// Gated on `ANTIPODAL_HAS_OPENGL`; including it without that define is a no-op.

#include <antipodal/math/common.hh>

#if defined(ANTIPODAL_HAS_OPENGL) && ANTIPODAL_HAS_OPENGL

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace antipodal
{
// Minimal, self-contained OpenGL shim.
// Types, enums, and entry points are declared here, so the header needs no GL loader library and never collides with a consumer's own <GL/gl.h>.
namespace opengl_detail
{
// Loaded pointers must match the platform GL calling convention or calls corrupt the stack.
// That is __stdcall on Windows, the default convention elsewhere.
#ifdef _WIN32
#define ANTIPODAL_GLAPI __stdcall
#else
#define ANTIPODAL_GLAPI
#endif

using GLenum = unsigned int;
using GLbitfield = unsigned int;
using GLuint = unsigned int;
using GLint = int;
using GLsizei = int;
using GLboolean = unsigned char;
using GLchar = char;
using GLfloat = float;
using GLintptr = std::intptr_t;
using GLsizeiptr = std::intptr_t;

inline constexpr GLenum GL_COMPUTE_SHADER = 0x91B9;
inline constexpr GLenum GL_SHADER_STORAGE_BUFFER = 0x90D2;
inline constexpr GLenum GL_COMPILE_STATUS = 0x8B81;
inline constexpr GLenum GL_LINK_STATUS = 0x8B82;
inline constexpr GLenum GL_INFO_LOG_LENGTH = 0x8B84;
inline constexpr GLbitfield GL_SHADER_STORAGE_BARRIER_BIT = 0x00002000;
inline constexpr GLbitfield GL_ALL_BARRIER_BITS = 0xFFFFFFFFu;
inline constexpr GLenum GL_DYNAMIC_DRAW = 0x88E8;
inline constexpr GLboolean GL_FALSE = 0;

using GlLoader = void* (*)(char const*);

// Function-pointer typedefs for every entry point the evaluator needs.
using PFN_glCreateShader = GLuint(ANTIPODAL_GLAPI*)(GLenum);
using PFN_glShaderSource = void(ANTIPODAL_GLAPI*)(GLuint, GLsizei, GLchar const* const*, GLint const*);
using PFN_glCompileShader = void(ANTIPODAL_GLAPI*)(GLuint);
using PFN_glGetShaderiv = void(ANTIPODAL_GLAPI*)(GLuint, GLenum, GLint*);
using PFN_glGetShaderInfoLog = void(ANTIPODAL_GLAPI*)(GLuint, GLsizei, GLsizei*, GLchar*);
using PFN_glCreateProgram = GLuint(ANTIPODAL_GLAPI*)(void);
using PFN_glAttachShader = void(ANTIPODAL_GLAPI*)(GLuint, GLuint);
using PFN_glLinkProgram = void(ANTIPODAL_GLAPI*)(GLuint);
using PFN_glGetProgramiv = void(ANTIPODAL_GLAPI*)(GLuint, GLenum, GLint*);
using PFN_glGetProgramInfoLog = void(ANTIPODAL_GLAPI*)(GLuint, GLsizei, GLsizei*, GLchar*);
using PFN_glDeleteShader = void(ANTIPODAL_GLAPI*)(GLuint);
using PFN_glDeleteProgram = void(ANTIPODAL_GLAPI*)(GLuint);
using PFN_glUseProgram = void(ANTIPODAL_GLAPI*)(GLuint);
using PFN_glGetUniformLocation = GLint(ANTIPODAL_GLAPI*)(GLuint, GLchar const*);
using PFN_glUniform1i = void(ANTIPODAL_GLAPI*)(GLint, GLint);
using PFN_glUniform3f = void(ANTIPODAL_GLAPI*)(GLint, GLfloat, GLfloat, GLfloat);
using PFN_glUniformMatrix4fv = void(ANTIPODAL_GLAPI*)(GLint, GLsizei, GLboolean, GLfloat const*);
using PFN_glGenBuffers = void(ANTIPODAL_GLAPI*)(GLsizei, GLuint*);
using PFN_glBindBuffer = void(ANTIPODAL_GLAPI*)(GLenum, GLuint);
using PFN_glBufferData = void(ANTIPODAL_GLAPI*)(GLenum, GLsizeiptr, void const*, GLenum);
using PFN_glBindBufferBase = void(ANTIPODAL_GLAPI*)(GLenum, GLuint, GLuint);
using PFN_glGetBufferSubData = void(ANTIPODAL_GLAPI*)(GLenum, GLintptr, GLsizeiptr, void*);
using PFN_glDeleteBuffers = void(ANTIPODAL_GLAPI*)(GLsizei, GLuint const*);
using PFN_glDispatchCompute = void(ANTIPODAL_GLAPI*)(GLuint, GLuint, GLuint);
using PFN_glMemoryBarrier = void(ANTIPODAL_GLAPI*)(GLbitfield);

// Resolved GL entry points; populated once by `load`.
struct GlApi
{
    PFN_glCreateShader CreateShader{};
    PFN_glShaderSource ShaderSource{};
    PFN_glCompileShader CompileShader{};
    PFN_glGetShaderiv GetShaderiv{};
    PFN_glGetShaderInfoLog GetShaderInfoLog{};
    PFN_glCreateProgram CreateProgram{};
    PFN_glAttachShader AttachShader{};
    PFN_glLinkProgram LinkProgram{};
    PFN_glGetProgramiv GetProgramiv{};
    PFN_glGetProgramInfoLog GetProgramInfoLog{};
    PFN_glDeleteShader DeleteShader{};
    PFN_glDeleteProgram DeleteProgram{};
    PFN_glUseProgram UseProgram{};
    PFN_glGetUniformLocation GetUniformLocation{};
    PFN_glUniform1i Uniform1i{};
    PFN_glUniform3f Uniform3f{};
    PFN_glUniformMatrix4fv UniformMatrix4fv{};
    PFN_glGenBuffers GenBuffers{};
    PFN_glBindBuffer BindBuffer{};
    PFN_glBufferData BufferData{};
    PFN_glBindBufferBase BindBufferBase{};
    PFN_glGetBufferSubData GetBufferSubData{};
    PFN_glDeleteBuffers DeleteBuffers{};
    PFN_glDispatchCompute DispatchCompute{};
    PFN_glMemoryBarrier MemoryBarrier{};

    // Resolve every entry point via `loader`, throwing if any is missing.
    void load(GlLoader loader)
    {
        CreateShader = require<PFN_glCreateShader>(loader, "glCreateShader");
        ShaderSource = require<PFN_glShaderSource>(loader, "glShaderSource");
        CompileShader = require<PFN_glCompileShader>(loader, "glCompileShader");
        GetShaderiv = require<PFN_glGetShaderiv>(loader, "glGetShaderiv");
        GetShaderInfoLog = require<PFN_glGetShaderInfoLog>(loader, "glGetShaderInfoLog");
        CreateProgram = require<PFN_glCreateProgram>(loader, "glCreateProgram");
        AttachShader = require<PFN_glAttachShader>(loader, "glAttachShader");
        LinkProgram = require<PFN_glLinkProgram>(loader, "glLinkProgram");
        GetProgramiv = require<PFN_glGetProgramiv>(loader, "glGetProgramiv");
        GetProgramInfoLog = require<PFN_glGetProgramInfoLog>(loader, "glGetProgramInfoLog");
        DeleteShader = require<PFN_glDeleteShader>(loader, "glDeleteShader");
        DeleteProgram = require<PFN_glDeleteProgram>(loader, "glDeleteProgram");
        UseProgram = require<PFN_glUseProgram>(loader, "glUseProgram");
        GetUniformLocation = require<PFN_glGetUniformLocation>(loader, "glGetUniformLocation");
        Uniform1i = require<PFN_glUniform1i>(loader, "glUniform1i");
        Uniform3f = require<PFN_glUniform3f>(loader, "glUniform3f");
        UniformMatrix4fv = require<PFN_glUniformMatrix4fv>(loader, "glUniformMatrix4fv");
        GenBuffers = require<PFN_glGenBuffers>(loader, "glGenBuffers");
        BindBuffer = require<PFN_glBindBuffer>(loader, "glBindBuffer");
        BufferData = require<PFN_glBufferData>(loader, "glBufferData");
        BindBufferBase = require<PFN_glBindBufferBase>(loader, "glBindBufferBase");
        GetBufferSubData = require<PFN_glGetBufferSubData>(loader, "glGetBufferSubData");
        DeleteBuffers = require<PFN_glDeleteBuffers>(loader, "glDeleteBuffers");
        DispatchCompute = require<PFN_glDispatchCompute>(loader, "glDispatchCompute");
        MemoryBarrier = require<PFN_glMemoryBarrier>(loader, "glMemoryBarrier");
    }

private:
    template <class Fn>
    static Fn require(GlLoader loader, char const* name)
    {
        void* const p = loader(name);
        if (p == nullptr)
            throw std::runtime_error(std::string("OpenGLMeshGwn: failed to load GL entry point: ") + name);
        // Object-to-function-pointer casts are not standard, so memcpy carries the bits across.
        // That also silences the pedantic warning and matches how GL loaders do it in practice.
        Fn fn{};
        std::memcpy(&fn, &p, sizeof(fn));
        return fn;
    }
};

// Clears the integer accumulator to zero.
inline constexpr char const* k_src_init = R"(#version 430
layout(local_size_x = 1, local_size_y = 64, local_size_z = 1) in;

layout(std430, binding = 0) buffer bBufferWnrInt { int out_wnrs_int[]; };

uniform int uCountX;
uniform int uCountY;
uniform int uCountZ;

void main()
{
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    uint z = gl_GlobalInvocationID.z;

    if (x >= uint(uCountX) || y >= uint(uCountY) || z >= uint(uCountZ))
        return;

    uint idx = (z * uint(uCountY) + y) * uint(uCountX) + x;
    out_wnrs_int[idx] = 0;
}
)";

// Scatters signed ray-crossings of every triangle into the integer buffer.
// The integration axis is local x (the innermost `uCountX` dimension).
inline constexpr char const* k_src_rasterize = R"(#version 430
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) buffer bDataTriangles { float data_triangles[]; };
layout(std430, binding = 1) buffer bBufferWnrInt { int out_wnrs_int[]; };

uniform int uCountX;
uniform int uCountY;
uniform int uCountZ;
uniform int uCountTriangles;

uniform mat4 uInvTransform;

void swap_vec3(inout vec3 a, inout vec3 b)
{
    vec3 tmp = a;
    a = b;
    b = tmp;
}

struct yline
{
    float slope_x;
    float intercept_x;
    float slope_z;
    float intercept_z;
};

float line_x_at(yline l, float y) { return l.slope_x * y + l.intercept_x; }
float line_z_at(yline l, float y) { return l.slope_z * y + l.intercept_z; }

yline make_yline_from_pos(vec3 p0, vec3 p1)
{
    vec3 dir = p1 - p0;

    yline l;
    l.slope_x = dir.x / dir.y;
    l.intercept_x = p0.x - p0.y * l.slope_x;
    l.slope_z = dir.z / dir.y;
    l.intercept_z = p0.z - p0.y * l.slope_z;
    return l;
}

void on_pixel(int y, int z, float x, int inc)
{
    int ix = max(0, int(ceil(x)));

    // bounds checks
    if (ix >= uCountX)
        return;

    // NOTE: atomic because this is contended per tri
    int idx = (z * uCountY + y) * uCountX + ix;
    atomicAdd(out_wnrs_int[idx], inc);
}

void rasterize_depth_tri(vec3 p0, vec3 p1, vec3 p2, int x_min, int x_max, int y_min, int y_max)
{
    // sign is a constant for the tri
    int inc = cross(p1 - p0, p2 - p0).z < 0 ? 1 : -1;

    // sort by y
    if (p0.y > p1.y)
        swap_vec3(p0, p1);
    if (p0.y > p2.y)
        swap_vec3(p0, p2);
    if (p1.y > p2.y)
        swap_vec3(p1, p2);

    if (p0.y == p2.y)
        return; // degenerate tri on an y level

    yline l01 = make_yline_from_pos(p0, p1);
    yline l02 = make_yline_from_pos(p0, p2);
    yline l12 = make_yline_from_pos(p1, p2);

    int y_start = max(int(ceil(p0.y)), y_min);
    int y_end = min(int(floor(p2.y)), y_max);

    for (int y = y_start; y <= y_end; ++y)
    {
        yline ll = y < p1.y || p1.y == p2.y ? l01 : l12;

        float x0 = line_x_at(ll, y);
        float x1 = line_x_at(l02, y);

        float z0 = line_z_at(ll, y);
        float z1 = line_z_at(l02, y);

        int x_start = max(x_min, int(ceil(min(x0, x1))));
        int x_end = min(x_max, int(floor(max(x0, x1))));

        for (int x = x_start; x <= x_end; ++x)
        {
            //    x = mix(x0, x1, t) => t = (x - x0) / (x1 - x0)
            float t = (x - x0) / (x1 - x0);
            float z = z0 + (z1 - z0) * t;

            on_pixel(x, y, z, inc);
        }
    }
}

void main()
{
    uint i = gl_GlobalInvocationID.x;

    if (i >= uint(uCountTriangles))
        return;

    vec3 p0 = vec3(uInvTransform * vec4(data_triangles[i * 9 + 0], data_triangles[i * 9 + 1], data_triangles[i * 9 + 2], 1));
    vec3 p1 = vec3(uInvTransform * vec4(data_triangles[i * 9 + 3], data_triangles[i * 9 + 4], data_triangles[i * 9 + 5], 1));
    vec3 p2 = vec3(uInvTransform * vec4(data_triangles[i * 9 + 6], data_triangles[i * 9 + 7], data_triangles[i * 9 + 8], 1));

    // aabb check
    // NOTE: aabb must be half-open in negative x dir (which is the z coord in local)
    vec3 bb_min = min(p0, min(p1, p2));
    vec3 bb_max = max(p0, max(p1, p2));

    if (bb_min.x > uCountY - 1.0 || bb_min.y > uCountZ - 1.0 || bb_min.z > uCountX - 1.0 || bb_max.x < 0 || bb_max.y < 0)
        return; // out of bounds

    rasterize_depth_tri(p0, p1, p2, 0, uCountY - 1, 0, uCountZ - 1);
}
)";

// Prefix-sums crossings along the integration axis (local x) per (y, z) row.
inline constexpr char const* k_src_integrate = R"(#version 430
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) buffer bBufferWnrInt { int out_wnrs_int[]; };

uniform int uCountX;
uniform int uCountY;
uniform int uCountZ;

void main()
{
    // NOTE: this renaming matches the C++ invocation
    int y = int(gl_GlobalInvocationID.x);
    int z = int(gl_GlobalInvocationID.y);

    if (y >= uCountY || z >= uCountZ)
        return;

    int wnr = 0;
    for (int x = 0; x < uCountX; ++x)
    {
        int idx = (z * uCountY + y) * uCountX + x;
        wnr += out_wnrs_int[idx];
        out_wnrs_int[idx] = wnr;
    }
}
)";

// Adds the fractional boundary integral and stores the full GWN.
// Uses the unnormalized Van Oosterom-Strackee form (no per-edge normalize), matching signed_spherical_tri_area_half_unorm in math/common.hh.
inline constexpr char const* k_src_wnr = R"(#version 430
layout(local_size_x = 1, local_size_y = 64, local_size_z = 1) in;

layout(std430, binding = 0) buffer bDataSegments { float data_segments[]; };
layout(std430, binding = 1) buffer bBufferWnrInt { int out_wnrs_int[]; };
layout(std430, binding = 2) buffer bBufferWnr { float out_wnrs[]; };

uniform int uCountSegments;
uniform int uCountX;
uniform int uCountY;
uniform int uCountZ;
uniform vec3 uPoleDir;
uniform vec3 uDirX;
uniform vec3 uDirY;
uniform vec3 uDirZ;
uniform vec3 uStart;

void main()
{
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    uint z = gl_GlobalInvocationID.z;

    if (x >= uint(uCountX) || y >= uint(uCountY) || z >= uint(uCountZ))
        return;

    vec3 ref_pos = uStart + uDirX * x + uDirY * y + uDirZ * z;

    float area = 0.0;
    for (int i = 0; i < uCountSegments; ++i)
    {
        float seg_w = data_segments[i * 7 + 0];
        vec3 p0 = vec3(data_segments[i * 7 + 1], data_segments[i * 7 + 2], data_segments[i * 7 + 3]);
        vec3 p1 = vec3(data_segments[i * 7 + 4], data_segments[i * 7 + 5], data_segments[i * 7 + 6]);

        // NOT normalized: the length-scaled denominator matches the normalized formula while dropping two normalize() calls per edge.
        // Matches signed_spherical_tri_area_half_unorm(x1, v0, v1).
        // The rasterizer accumulates crossings from the -s_dx end, so its integer term is the ray in the -s_dx direction.
        // The consistent antipodal pole is then x1 = +s_dx (uPoleDir), not -uPoleDir.
        vec3 x1 = uPoleDir;
        vec3 v0 = p0 - ref_pos;
        vec3 v1 = p1 - ref_pos;
        float l0 = length(v0);
        float l1 = length(v1);

        float num = dot(x1, cross(v0, v1));
        float denom = l0 * l1 + l1 * dot(x1, v0) + l0 * dot(x1, v1) + dot(v0, v1);

        area += seg_w * 2.0 * atan(num, denom);
    }

    float pi = 3.141592653589793238462643383279;
    float wnr = area / (4 * pi);

    uint idx = (z * uint(uCountY) + y) * uint(uCountX) + x;
    out_wnrs[idx] = out_wnrs_int[idx] + wnr;
}
)";
} // namespace opengl_detail

// Evaluates the full mesh GWN over a dense 3D grid on the GPU.
//
// A current OpenGL 4.3 context must be current on the constructing / using thread.
// Non-copyable: owns GL program and buffer objects.
// All geometry is `float`.
//
// Typical use:
//   antipodal::OpenGLMeshGwn gwn(reinterpret_cast<void*(*)(char const*)>(wglGetProcAddress));
//   gwn.set_scene(vertices, indices, antipodal::build_boundary_segments<float>(vertices, indices));
//   std::vector<float> out(nx * ny * nz);
//   gwn.eval_grid(start, dx, dy, dz, nx, ny, nz, out);
struct OpenGLMeshGwn
{
    using GlLoader = opengl_detail::GlLoader;

    // Loads GL entry points via `loader` and compiles the four compute programs.
    // Throws std::runtime_error on a missing entry point or a shader failure.
    explicit OpenGLMeshGwn(GlLoader loader)
    {
        m_gl.load(loader);
        m_prog_init = compile_program(opengl_detail::k_src_init);
        m_prog_rasterize = compile_program(opengl_detail::k_src_rasterize);
        m_prog_integrate = compile_program(opengl_detail::k_src_integrate);
        m_prog_wnr = compile_program(opengl_detail::k_src_wnr);
    }

    ~OpenGLMeshGwn()
    {
        auto& gl = m_gl;
        if (m_buf_triangles)
            gl.DeleteBuffers(1, &m_buf_triangles);
        if (m_buf_segments)
            gl.DeleteBuffers(1, &m_buf_segments);
        if (m_buf_wnr_int)
            gl.DeleteBuffers(1, &m_buf_wnr_int);
        if (m_buf_wnr_float)
            gl.DeleteBuffers(1, &m_buf_wnr_float);
        if (m_prog_init)
            gl.DeleteProgram(m_prog_init);
        if (m_prog_rasterize)
            gl.DeleteProgram(m_prog_rasterize);
        if (m_prog_integrate)
            gl.DeleteProgram(m_prog_integrate);
        if (m_prog_wnr)
            gl.DeleteProgram(m_prog_wnr);
    }

    OpenGLMeshGwn(OpenGLMeshGwn const&) = delete;
    OpenGLMeshGwn& operator=(OpenGLMeshGwn const&) = delete;
    OpenGLMeshGwn(OpenGLMeshGwn&&) = delete;
    OpenGLMeshGwn& operator=(OpenGLMeshGwn&&) = delete;

    // Uploads the triangle soup and weighted boundary segments to the GPU.
    //
    // `indices.size()` must be a multiple of 3.
    // `boundary` is empty for closed meshes; build it with `build_boundary_segments` (math/common.hh).
    void set_scene(std::span<fvec3 const> vertices,
                   std::span<int const> indices,
                   std::span<weighted_fsegment3 const> boundary)
    {
        assert(indices.size() % 3 == 0);

        m_tri_count = static_cast<int>(indices.size() / 3);
        m_seg_count = static_cast<int>(boundary.size());

        // Flatten triangles to 9 floats each (p0.xyz, p1.xyz, p2.xyz).
        std::vector<float> data_triangles;
        data_triangles.reserve(std::size_t(m_tri_count) * 9);
        for (std::size_t t = 0; t < indices.size() / 3; ++t)
        {
            for (int c = 0; c < 3; ++c)
            {
                auto const& p = vertices[indices[3 * t + c]];
                data_triangles.push_back(p.x);
                data_triangles.push_back(p.y);
                data_triangles.push_back(p.z);
            }
        }

        // Flatten segments to 7 floats each (weight, p0.xyz, p1.xyz).
        std::vector<float> data_segments;
        data_segments.reserve(std::size_t(m_seg_count) * 7);
        for (auto const& ws : boundary)
        {
            data_segments.push_back(ws.weight);
            data_segments.push_back(ws.segment.pos0.x);
            data_segments.push_back(ws.segment.pos0.y);
            data_segments.push_back(ws.segment.pos0.z);
            data_segments.push_back(ws.segment.pos1.x);
            data_segments.push_back(ws.segment.pos1.y);
            data_segments.push_back(ws.segment.pos1.z);
        }

        upload_buffer(m_buf_triangles, data_triangles);
        upload_buffer(m_buf_segments, data_segments);
    }

    // Evaluates the full GWN at every cell of the grid `p(ix,iy,iz) = start + ix*dx + iy*dy + iz*dz`.
    //
    // `out` must have exactly `nx*ny*nz` elements.
    // `out` is filled in the caller's axis ordering, layout `((iz*ny)+iy)*nx + ix`.
    //
    // The axis with the fewest cells is chosen internally as the ray / integration axis; the full GWN is invariant to that choice.
    // That minimizes the prefix-sum work, and a single-cell axis (a 2D slice) skips the integrate pass entirely.
    void eval_grid(fvec3 start, fvec3 dx, fvec3 dy, fvec3 dz, int nx, int ny, int nz, std::span<float> out)
    {
        assert(nx > 0 && ny > 0 && nz > 0);
        assert(out.size() == std::size_t(nx) * std::size_t(ny) * std::size_t(nz));

        // Pick the integration axis = the axis with the fewest cells.
        fvec3 const dirs[3] = {dx, dy, dz};
        int const counts[3] = {nx, ny, nz};
        int ia = 0;
        if (counts[1] < counts[ia])
            ia = 1;
        if (counts[2] < counts[ia])
            ia = 2;

        // The other two axes become shader Y and Z.
        // Order them so the local frame (s_dy, s_dz, s_dx) is right-handed.
        // The rasterizer derives each triangle's crossing sign from the local-space normal, so a left-handed frame would negate the whole integer term.
        int ax_y = (ia == 0) ? 1 : 0;
        int ax_z = (ia == 2) ? 1 : 2;
        if (dot(dirs[ax_y], cross(dirs[ax_z], dirs[ia])) < 0.0f)
            std::swap(ax_y, ax_z);

        int const cx = counts[ia]; // integration axis -> local x (uCountX)
        int const cy = counts[ax_y];
        int const cz = counts[ax_z];

        fvec3 const s_dx = dirs[ia]; // integration direction
        fvec3 const s_dy = dirs[ax_y];
        fvec3 const s_dz = dirs[ax_z];

        auto const x0 = normalize(s_dx);

        ensure_grid_buffers(std::size_t(cx) * std::size_t(cy) * std::size_t(cz));

        auto& gl = m_gl;
        using namespace opengl_detail;

        // --- init: clear integer accumulator ---
        {
            gl.UseProgram(m_prog_init);
            gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_buf_wnr_int);
            set_uniform_i(m_prog_init, "uCountX", cx);
            set_uniform_i(m_prog_init, "uCountY", cy);
            set_uniform_i(m_prog_init, "uCountZ", cz);
            gl.DispatchCompute(GLuint(cx), GLuint(div_ceil(cy, 64)), GLuint(cz));
            gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }

        // --- rasterize: scatter signed crossings (skip if no triangles) ---
        if (m_tri_count > 0)
        {
            gl.UseProgram(m_prog_rasterize);
            gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_buf_triangles);
            gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_buf_wnr_int);

            // T : grid space -> world space, columns (s_dy, s_dz, s_dx, start).
            // s_dx becomes the local z (depth / integration) coordinate.
            float t_inv[16];
            affine_inverse(s_dy, s_dz, s_dx, start, t_inv);

            set_uniform_i(m_prog_rasterize, "uCountX", cx);
            set_uniform_i(m_prog_rasterize, "uCountY", cy);
            set_uniform_i(m_prog_rasterize, "uCountZ", cz);
            set_uniform_i(m_prog_rasterize, "uCountTriangles", m_tri_count);
            set_uniform_mat4(m_prog_rasterize, "uInvTransform", t_inv);

            gl.DispatchCompute(GLuint(div_ceil(m_tri_count, 64)), 1, 1);
            gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }

        // --- integrate: prefix-sum along the integration axis ---
        // Skipped when the integration axis is a single cell: the length-1 prefix sum is the identity, so the rasterize output already holds the integer term.
        if (m_tri_count > 0 && cx > 1)
        {
            gl.UseProgram(m_prog_integrate);
            gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_buf_wnr_int);
            set_uniform_i(m_prog_integrate, "uCountX", cx);
            set_uniform_i(m_prog_integrate, "uCountY", cy);
            set_uniform_i(m_prog_integrate, "uCountZ", cz);
            // inner loop is y, outer is z
            gl.DispatchCompute(GLuint(div_ceil(cy, 64)), GLuint(cz), 1);
            gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }

        // --- wnr: add fractional term, store full GWN ---
        {
            gl.UseProgram(m_prog_wnr);
            gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_buf_segments);
            gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_buf_wnr_int);
            gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_buf_wnr_float);

            set_uniform_i(m_prog_wnr, "uCountSegments", m_seg_count);
            set_uniform_i(m_prog_wnr, "uCountX", cx);
            set_uniform_i(m_prog_wnr, "uCountY", cy);
            set_uniform_i(m_prog_wnr, "uCountZ", cz);
            set_uniform_v3(m_prog_wnr, "uPoleDir", x0);
            set_uniform_v3(m_prog_wnr, "uDirX", s_dx);
            set_uniform_v3(m_prog_wnr, "uDirY", s_dy);
            set_uniform_v3(m_prog_wnr, "uDirZ", s_dz);
            set_uniform_v3(m_prog_wnr, "uStart", start);

            gl.DispatchCompute(GLuint(cx), GLuint(div_ceil(cy, 64)), GLuint(cz));
            gl.MemoryBarrier(GL_ALL_BARRIER_BITS);
        }

        // Read the permuted grid back and scatter into the caller's ordering.
        std::vector<float> gpu(std::size_t(cx) * std::size_t(cy) * std::size_t(cz));
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, m_buf_wnr_float);
        gl.GetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, GLsizeiptr(gpu.size() * sizeof(float)), gpu.data());

        for (int zs = 0; zs < cz; ++zs)
            for (int ys = 0; ys < cy; ++ys)
                for (int xs = 0; xs < cx; ++xs)
                {
                    int caller[3];
                    caller[ia] = xs;
                    caller[ax_y] = ys;
                    caller[ax_z] = zs;
                    std::size_t const dst = (std::size_t(caller[2]) * ny + caller[1]) * nx + caller[0];
                    std::size_t const src = (std::size_t(zs) * cy + ys) * cx + xs;
                    out[dst] = gpu[src];
                }
    }

private:
    using GLuint = opengl_detail::GLuint;
    using GLint = opengl_detail::GLint;
    using GLsizei = opengl_detail::GLsizei;
    using GLsizeiptr = opengl_detail::GLsizeiptr;

    static int div_ceil(int a, int b) { return (a + b - 1) / b; }

    GLuint compile_program(char const* source)
    {
        using namespace opengl_detail;
        auto& gl = m_gl;

        GLuint const shader = gl.CreateShader(GL_COMPUTE_SHADER);
        gl.ShaderSource(shader, 1, &source, nullptr);
        gl.CompileShader(shader);

        GLint ok = 0;
        gl.GetShaderiv(shader, GL_COMPILE_STATUS, &ok);
        if (!ok)
        {
            std::string const log = shader_log(shader);
            gl.DeleteShader(shader);
            throw std::runtime_error("OpenGLMeshGwn: compute shader compile failed:\n" + log);
        }

        GLuint const program = gl.CreateProgram();
        gl.AttachShader(program, shader);
        gl.LinkProgram(program);
        gl.DeleteShader(shader);

        gl.GetProgramiv(program, GL_LINK_STATUS, &ok);
        if (!ok)
        {
            std::string const log = program_log(program);
            gl.DeleteProgram(program);
            throw std::runtime_error("OpenGLMeshGwn: program link failed:\n" + log);
        }
        return program;
    }

    std::string shader_log(GLuint shader)
    {
        using namespace opengl_detail;
        GLint len = 0;
        m_gl.GetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
        if (len <= 0)
            return {};
        std::string log(std::size_t(len), '\0');
        m_gl.GetShaderInfoLog(shader, len, nullptr, log.data());
        return log;
    }

    std::string program_log(GLuint program)
    {
        using namespace opengl_detail;
        GLint len = 0;
        m_gl.GetProgramiv(program, GL_INFO_LOG_LENGTH, &len);
        if (len <= 0)
            return {};
        std::string log(std::size_t(len), '\0');
        m_gl.GetProgramInfoLog(program, len, nullptr, log.data());
        return log;
    }

    // Creates the buffer on first use, then (re)uploads the given float data.
    void upload_buffer(GLuint& buffer, std::vector<float> const& data)
    {
        using namespace opengl_detail;
        if (buffer == 0)
            m_gl.GenBuffers(1, &buffer);
        m_gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
        // A zero-size SSBO is invalid; upload a single-element placeholder.
        float const placeholder = 0.0f;
        void const* ptr = data.empty() ? static_cast<void const*>(&placeholder) : static_cast<void const*>(data.data());
        GLsizeiptr const bytes = data.empty() ? GLsizeiptr(sizeof(float)) : GLsizeiptr(data.size() * sizeof(float));
        m_gl.BufferData(GL_SHADER_STORAGE_BUFFER, bytes, ptr, GL_DYNAMIC_DRAW);
    }

    // (Re)allocates the integer and float grid buffers to hold `cells` elements.
    void ensure_grid_buffers(std::size_t cells)
    {
        using namespace opengl_detail;
        if (m_buf_wnr_int == 0)
            m_gl.GenBuffers(1, &m_buf_wnr_int);
        if (m_buf_wnr_float == 0)
            m_gl.GenBuffers(1, &m_buf_wnr_float);
        if (cells <= m_grid_capacity)
            return;

        m_grid_capacity = cells;
        m_gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, m_buf_wnr_int);
        m_gl.BufferData(GL_SHADER_STORAGE_BUFFER, GLsizeiptr(cells * sizeof(std::int32_t)), nullptr, GL_DYNAMIC_DRAW);
        m_gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, m_buf_wnr_float);
        m_gl.BufferData(GL_SHADER_STORAGE_BUFFER, GLsizeiptr(cells * sizeof(float)), nullptr, GL_DYNAMIC_DRAW);
    }

    void set_uniform_i(GLuint program, char const* name, int value)
    {
        m_gl.Uniform1i(m_gl.GetUniformLocation(program, name), GLint(value));
    }

    void set_uniform_v3(GLuint program, char const* name, fvec3 v)
    {
        m_gl.Uniform3f(m_gl.GetUniformLocation(program, name), v.x, v.y, v.z);
    }

    void set_uniform_mat4(GLuint program, char const* name, float const* col_major)
    {
        m_gl.UniformMatrix4fv(m_gl.GetUniformLocation(program, name), 1, opengl_detail::GL_FALSE, col_major);
    }

    // Column-major 4x4 inverse of the affine map with linear columns (c0,c1,c2) and translation `t` (last row 0,0,0,1).
    // Writes 16 floats for GL.
    static void affine_inverse(fvec3 c0, fvec3 c1, fvec3 c2, fvec3 t, float out[16])
    {
        // For M with columns (c0,c1,c2), the inverse rows are cross(c1,c2), cross(c2,c0), cross(c0,c1), scaled by 1/det.
        auto const r0 = cross(c1, c2);
        auto const r1 = cross(c2, c0);
        auto const r2 = cross(c0, c1);
        float const det = dot(c0, r0);
        float const inv = 1.0f / det;

        fvec3 const l0 = r0 * inv; // linear-inverse row 0
        fvec3 const l1 = r1 * inv;
        fvec3 const l2 = r2 * inv;

        // translation of the inverse = -Linv * t
        fvec3 const ti = {-dot(l0, t), -dot(l1, t), -dot(l2, t)};

        // column-major: out[col*4 + row]
        out[0] = l0.x;
        out[1] = l1.x;
        out[2] = l2.x;
        out[3] = 0.0f;
        out[4] = l0.y;
        out[5] = l1.y;
        out[6] = l2.y;
        out[7] = 0.0f;
        out[8] = l0.z;
        out[9] = l1.z;
        out[10] = l2.z;
        out[11] = 0.0f;
        out[12] = ti.x;
        out[13] = ti.y;
        out[14] = ti.z;
        out[15] = 1.0f;
    }

    opengl_detail::GlApi m_gl{};

    GLuint m_prog_init{};
    GLuint m_prog_rasterize{};
    GLuint m_prog_integrate{};
    GLuint m_prog_wnr{};

    GLuint m_buf_triangles{};
    GLuint m_buf_segments{};
    int m_tri_count{};
    int m_seg_count{};

    GLuint m_buf_wnr_int{};
    GLuint m_buf_wnr_float{};
    std::size_t m_grid_capacity{};
};
} // namespace antipodal

#undef ANTIPODAL_GLAPI

#endif
