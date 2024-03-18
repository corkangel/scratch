
set(CMAKE_SUPPRESS_REGENERATION true)
set(CMAKE_CXX_STANDARD 17) 
cmake_minimum_required(VERSION 3.1)

project(scratch)

include_directories(imgui imgui/backends)

add_executable(
    scratch 
    main.cpp
    tests.cpp
    stensor.cpp 
    stensor.h)

add_library(imgui STATIC
    imguidemo.cpp
    imgui/imgui.h
    imgui/imgui.cpp
    imgui/imgui_internal.h
    imgui/imgui_draw.cpp
    imgui/imgui_tables.cpp
    imgui/imgui_widgets.cpp
    imgui/imgui_demo.cpp
    imgui/backends/imgui_impl_win32.cpp
    imgui/backends/imgui_impl_win32.h
    imgui/backends/imgui_impl_dx11.cpp
    imgui/backends/imgui_impl_dx11.h
    imgui/misc/debuggers/imgui.natvis
    imgui/misc/debuggers/imgui.natstepfilter
    imgui/misc/cpp/imgui_stdlib.cpp
    imgui/misc/cpp/imgui_stdlib.h
)

add_executable(imguidemo WIN32 imguidemo.cpp)

target_link_libraries(imguidemo imgui d3d11.lib)