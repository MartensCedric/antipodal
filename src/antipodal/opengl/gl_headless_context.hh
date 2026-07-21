#pragma once

// Optional helper that creates a headless OpenGL 4.3 context and exposes a loader for `OpenGLMeshGwn` (`HeadlessGlContext`).
//
// Provides an off-screen GL context without any windowing / loader dependency.
//   - Windows: a hidden-window WGL context (works out of the box).
//   - Linux:   an EGL surfaceless / pbuffer context (install the EGL dev package, e.g. `libegl1-mesa-dev`; nothing is bundled here).
//
// `HeadlessGlContext::create()` returns `nullptr` when no context can be made (no GPU, no driver), so callers such as the test suite can skip gracefully.
// The caller links the platform libs: Windows needs `opengl32 gdi32 user32`, Linux needs the GL and EGL libraries.
//
// Gated on `ANTIPODAL_HAS_OPENGL`; a no-op without it.

#include <antipodal/gwn_mesh_opengl.hh>

#if defined(ANTIPODAL_HAS_OPENGL) && ANTIPODAL_HAS_OPENGL

#include <cstring>
#include <memory>

#if defined(_WIN32)
#define ANTIPODAL_GL_CTX_WGL 1
#elif defined(__linux__)
#define ANTIPODAL_GL_CTX_EGL 1
#endif

#if defined(ANTIPODAL_GL_CTX_WGL)
// Keep <windows.h> from defining the min/max macros; they break std::min/std::max in any TU that includes this header.
// WIN32_LEAN_AND_MEAN trims the include for build speed.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
// clang-format off
#include <windows.h>
#include <GL/gl.h>
// clang-format on
#elif defined(ANTIPODAL_GL_CTX_EGL)
#include <EGL/egl.h>
#endif

namespace antipodal
{
// RAII headless GL context.
// Non-copyable; the underlying GL context stays current on the creating thread for the object's lifetime.
struct HeadlessGlContext
{
    using GlLoader = opengl_detail::GlLoader;

    HeadlessGlContext(HeadlessGlContext const&) = delete;
    HeadlessGlContext& operator=(HeadlessGlContext const&) = delete;

    // Attempts to create a headless GL 4.3 context; returns nullptr on failure.
    [[nodiscard]] static std::unique_ptr<HeadlessGlContext> create()
    {
        auto ctx = std::unique_ptr<HeadlessGlContext>(new HeadlessGlContext());
        if (!ctx->init())
            return nullptr;
        return ctx;
    }

    ~HeadlessGlContext() { destroy(); }

    // A loader suitable for `OpenGLMeshGwn` (glfw/SDL/wgl-style).
    [[nodiscard]] GlLoader loader() const { return &load_proc; }

private:
    HeadlessGlContext() = default;

#if defined(ANTIPODAL_GL_CTX_WGL)
    HWND m_hwnd{};
    HDC m_hdc{};
    HGLRC m_hglrc{};

    bool init()
    {
        WNDCLASSA wc{};
        wc.lpfnWndProc = DefWindowProcA;
        wc.hInstance = GetModuleHandleA(nullptr);
        wc.lpszClassName = "AntipodalHeadlessGL";
        RegisterClassA(&wc); // ignore "already registered"

        m_hwnd = CreateWindowA(wc.lpszClassName, "", WS_OVERLAPPEDWINDOW, 0, 0, 1, 1, nullptr, nullptr, wc.hInstance,
                               nullptr);
        if (!m_hwnd)
            return false;

        m_hdc = GetDC(m_hwnd);
        if (!m_hdc)
            return false;

        PIXELFORMATDESCRIPTOR pfd{};
        pfd.nSize = sizeof(pfd);
        pfd.nVersion = 1;
        pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
        pfd.iPixelType = PFD_TYPE_RGBA;
        pfd.cColorBits = 32;
        pfd.cDepthBits = 24;
        pfd.iLayerType = PFD_MAIN_PLANE;

        int const pf = ChoosePixelFormat(m_hdc, &pfd);
        if (pf == 0 || !SetPixelFormat(m_hdc, pf, &pfd))
            return false;

        // A compatibility context is enough: compute shaders are core in 4.3, and desktop drivers expose the highest supported version here.
        m_hglrc = wglCreateContext(m_hdc);
        if (!m_hglrc)
            return false;

        return wglMakeCurrent(m_hdc, m_hglrc) != FALSE;
    }

    void destroy()
    {
        if (m_hglrc)
        {
            wglMakeCurrent(nullptr, nullptr);
            wglDeleteContext(m_hglrc);
        }
        if (m_hdc && m_hwnd)
            ReleaseDC(m_hwnd, m_hdc);
        if (m_hwnd)
            DestroyWindow(m_hwnd);
    }

    static void* load_proc(char const* name)
    {
        // GL >1.1 entry points come from the ICD via wglGetProcAddress; GL 1.1 ones fall back to opengl32.dll.
        void* p = reinterpret_cast<void*>(wglGetProcAddress(name));
        // wglGetProcAddress may return small sentinel values on some drivers.
        auto const bad = (p == nullptr) || (p == reinterpret_cast<void*>(1)) || (p == reinterpret_cast<void*>(2))
                      || (p == reinterpret_cast<void*>(3)) || (p == reinterpret_cast<void*>(-1));
        if (bad)
        {
            static HMODULE const gl = LoadLibraryA("opengl32.dll");
            p = gl ? reinterpret_cast<void*>(GetProcAddress(gl, name)) : nullptr;
        }
        return p;
    }

#elif defined(ANTIPODAL_GL_CTX_EGL)
    EGLDisplay m_display{EGL_NO_DISPLAY};
    EGLContext m_context{EGL_NO_CONTEXT};
    EGLSurface m_surface{EGL_NO_SURFACE};

    bool init()
    {
        m_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (m_display == EGL_NO_DISPLAY)
            return false;
        if (!eglInitialize(m_display, nullptr, nullptr))
            return false;
        if (!eglBindAPI(EGL_OPENGL_API))
            return false;

        EGLint const cfg_attr[] = {
            EGL_SURFACE_TYPE,
            EGL_PBUFFER_BIT, //
            EGL_RENDERABLE_TYPE,
            EGL_OPENGL_BIT, //
            EGL_RED_SIZE,
            8,
            EGL_GREEN_SIZE,
            8,
            EGL_BLUE_SIZE,
            8,        //
            EGL_NONE, //
        };
        EGLConfig config{};
        EGLint num_config = 0;
        if (!eglChooseConfig(m_display, cfg_attr, &config, 1, &num_config) || num_config < 1)
            return false;

        EGLint const ctx_attr[] = {
            EGL_CONTEXT_MAJOR_VERSION,
            4, //
            EGL_CONTEXT_MINOR_VERSION,
            3,        //
            EGL_NONE, //
        };
        m_context = eglCreateContext(m_display, config, EGL_NO_CONTEXT, ctx_attr);
        if (m_context == EGL_NO_CONTEXT)
            return false;

        EGLint const pbuf_attr[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
        m_surface = eglCreatePbufferSurface(m_display, config, pbuf_attr);
        if (m_surface == EGL_NO_SURFACE)
            return false;

        return eglMakeCurrent(m_display, m_surface, m_surface, m_context) == EGL_TRUE;
    }

    void destroy()
    {
        if (m_display != EGL_NO_DISPLAY)
        {
            eglMakeCurrent(m_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
            if (m_surface != EGL_NO_SURFACE)
                eglDestroySurface(m_display, m_surface);
            if (m_context != EGL_NO_CONTEXT)
                eglDestroyContext(m_display, m_context);
            eglTerminate(m_display);
        }
    }

    static void* load_proc(char const* name)
    {
        auto const fn = eglGetProcAddress(name);
        void* p = nullptr;
        std::memcpy(&p, &fn, sizeof(p));
        return p;
    }

#else // no headless backend for this platform
    bool init() { return false; }
    void destroy() {}
    static void* load_proc(char const*) { return nullptr; }
#endif
};
} // namespace antipodal

#endif
