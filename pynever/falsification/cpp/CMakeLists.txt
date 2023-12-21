## You can just modify your project name, in line 5.

cmake_minimum_required(VERSION 3.25)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})
set(MY_PROJECT_NAME cpp_verify)
project(${MY_PROJECT_NAME})

######## Cplex Dependency
find_package(CPLEX REQUIRED)
if(CPLEX_FOUND)
    include_directories(${CPLEX_INCLUDE_DIR})
    link_directories(${CPLEX_LIBRARY})
endif()

if(CPLEX_CONCERT_FOUND)
    include_directories(${CPLEX_CONCERT_INCLUDE_DIR})
    link_directories(${CPLEX_CONCERT_LIBRARY})
endif()

if(CPLEX_ILOCPLEX_FOUND)
    include_directories(${CPLEX_ILOCPLEX_INCLUDE_DIR})
    link_directories(${CPLEX_ILOCPLEX_LIBRARY})
endif()

if(CPLEX_CP_FOUND)
    include_directories(${CPLEX_CP_INCLUDE_DIR})
    link_directories(${CPLEX_CP_LIBRARY})
endif()


######## Complier message
set(CMAKE_BUILD_TYPE Release)
message(STATUS "System: ${CMAKE_SYSTEM}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler version: ${CMAKE_CXX_COMPILER_VERSION}")

######## Set output directories.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build/lib)

######## Set C++ 11 standard
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++11" COMPILER_SUPPORTS_CXX11)
if (COMPILER_SUPPORTS_CXX11)
    if (CMAKE_COMPILER_IS_GNUCXX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
    else ()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    endif ()
else ()
    message(STATUS "Fatal Error: Your compiler doesn't support c++11, please install it!")
endif ()

######## find all the source files.
file(GLOB HEADER_FILES LEConstraintI.h CPVerificationSolver.h FullyConnectedLayer.h FullyConnectedNetwork.h PropertyBounds.h ReLUConstraintI.h UnReLUConstraintI.h)
file(GLOB SOURCE_FILES LEConstraintI.cpp CPVerificationSolver.cpp FullyConnectedLayer.cpp FullyConnectedNetwork.cpp PropertyBounds.cpp ReLUConstraintI.cpp UnReLUConstraintI.cpp bindings.cpp)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED HINTS "${Python3_SITELIB}")

######## Set executable file name, and add the source files for it.
pybind11_add_module(${MY_PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})
######## Add Dependency Library
target_link_libraries(${MY_PROJECT_NAME} PRIVATE cplex-library cplex-concert ilocplex cplex-cp)
