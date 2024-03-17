
set(CMAKE_CXX_STANDARD 17) 
cmake_minimum_required(VERSION 3.1)

project(scratch)

add_executable(
    scratch 
    main.cpp
    tests.cpp
    stensor.cpp 
    stensor.h)
