
cmake_minimum_required(VERSION 3.18)

set(CMAKE_SUPPRESS_REGENERATION true)
set(CMAKE_CXX_STANDARD 17) 
cmake_minimum_required(VERSION 3.1)

project(scratch)

include_directories(
    imgui 
    imgui/backends 
    implot 
    imguiapp 
    ${Stb_INCLUDE_DIR}
    #build/packages/Microsoft.AI.DirectML.1.13.1/include
    #build/packages/Microsoft.Windows.ImplementationLibrary.1.0.240122.1/include
    .)

# vanilla ImPlot - no dependencies
add_library(implot STATIC
    implot/implot.cpp
    implot/implot.h
    implot/implot_internal.h
    implot/implot_items.cpp
    implot/implot_demo.cpp)

# vanilla ImGui - no dependencies
add_library(imgui STATIC
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

# a base ImGui application for d3d device, no other depends
add_library(imguiapp STATIC
    imguiapp/imguiapp.cpp
    imguiapp/imguiapp.h)

# from scratch AI lib
add_library(scratchlib STATIC
    scratch.natvis
    #scratch/matmul.cu
    #scratch/matmul.ch
    scratch/dml.cpp 
    scratch/stensor.cpp 
    scratch/stensor.h
    scratch/slearner.h
    scratch/smodel.cpp
    scratch/smodel.h
    scratch/slog.cpp
    scratch/minst.h
    scratch/minst.cpp
    scratch/ptr.h
    scratch/utils.h)

# imgui utils for scratch lib
add_library(scratchgui STATIC
    scratchgui/tools.h
    scratchgui/tools.cpp
)

add_executable(meanshift WIN32 meanshift/app.cpp meanshift/meanshift.cpp)
target_link_libraries(meanshift scratchlib scratchgui imgui imguiapp implot d3d11.lib WindowsApp.lib dxgi.lib d3d12.lib)

add_executable(tests
     tests/app.cpp 
     tests/tests.h
     tests/test_dml.cpp
     tests/test_tensors.cpp
     tests/test_model.cpp
     tests/test_utils.cpp)

set_target_properties(tests PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

target_link_libraries(tests scratchlib WindowsApp.lib dxgi.lib d3d12.lib)

add_executable(sgd WIN32 sgd/app.cpp sgd/sgd.cpp)
target_link_libraries(sgd scratchlib scratchgui imgui imguiapp implot d3d11.lib WindowsApp.lib dxgi.lib d3d12.lib)

set_target_properties(sgd PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

add_executable(cnn WIN32 cnn/app.cpp cnn/cnn.cpp)
target_link_libraries(cnn scratchlib scratchgui imgui imguiapp implot d3d11.lib WindowsApp.lib dxgi.lib d3d12.lib)

set_target_properties(cnn PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")