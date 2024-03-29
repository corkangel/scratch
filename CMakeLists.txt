
cmake_minimum_required(VERSION 3.18)

set(CMAKE_SUPPRESS_REGENERATION true)
set(CMAKE_CXX_STANDARD 17) 
cmake_minimum_required(VERSION 3.1)

project(scratch)

find_package(CUDA REQUIRED)
enable_language(CUDA)

include_directories(imgui imgui/backends implot)

add_library(scratchlib STATIC
    matmul.cu
    matmul.ch
    meanshift.cpp
    tests.cpp
    stensor.cpp 
    stensor.h)

add_library(implot STATIC
    implot/implot.cpp
    implot/implot.h
    implot/implot_internal.h
    implot/implot_items.cpp
    implot/implot_demo.cpp)

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

add_library(imguidemo STATIC
    imguidemo.cpp
    imguidemo.h)

add_executable(scratchapp WIN32 imguiapp.cpp)

target_link_libraries(scratchapp scratchlib imgui imguidemo implot d3d11.lib cudart.lib cudadevrt.lib)