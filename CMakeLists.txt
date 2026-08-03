cmake_minimum_required(VERSION 3.16)
project(clustering-engine VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -Wall -Wextra -fPIC)
    # Check AVX2/FMA support at compile time
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx2" HAS_AVX2)
    check_cxx_compiler_flag("-mfma" HAS_FMA)
    if(HAS_AVX2)
        add_compile_options(-mavx2)
    endif()
    if(HAS_FMA)
        add_compile_options(-mfma)
    endif()
elseif(MSVC)
    add_compile_options(/W4 /O2)
endif()

# Debug logging
option(CLUSTERING_DEBUG "Enable detailed debug logging" OFF)
if(CLUSTERING_DEBUG)
    add_compile_definitions(CLUSTERING_DEBUG)
    message(STATUS "Clustering debug logging: ENABLED")
endif()

# Find dependencies
find_package(Eigen3 QUIET)
find_package(GTest QUIET)
find_package(glfw3 3.3 QUIET)
find_package(OpenGL REQUIRED)
find_package(GLEW QUIET)

# Fallback for GLFW on Windows (vcpkg or manual)
if(NOT glfw3_FOUND)
    find_library(GLFW_LIB glfw3 PATHS
        $ENV{VCPKG_ROOT}/installed/x64-windows/lib
        /usr/local/lib /usr/lib
    )
    find_path(GLFW_INCLUDE glfw3.h PATHS
        $ENV{VCPKG_ROOT}/installed/x64-windows/include
        /usr/local/include /usr/include
    )
    if(GLFW_LIB AND GLFW_INCLUDE)
        add_library(glfw UNKNOWN IMPORTED)
        set_target_properties(glfw PROPERTIES
            IMPORTED_LOCATION "${GLFW_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${GLFW_INCLUDE}"
        )
        set(glfw3_FOUND TRUE)
        message(STATUS "Found GLFW: ${GLFW_LIB}")
    else()
        message(FATAL_ERROR "GLFW not found. Install: sudo apt install libglfw3-dev (Linux) or vcpkg install glfw3 (Windows)")
    endif()
endif()

# Fallback for GLEW
if(NOT GLEW_FOUND)
    find_library(GLEW_LIB GLEW PATHS
        $ENV{VCPKG_ROOT}/installed/x64-windows/lib
        /usr/local/lib /usr/lib
    )
    find_path(GLEW_INCLUDE GL/glew.h PATHS
        $ENV{VCPKG_ROOT}/installed/x64-windows/include
        /usr/local/include /usr/include
    )
    if(GLEW_LIB AND GLEW_INCLUDE)
        add_library(GLEW::GLEW UNKNOWN IMPORTED)
        set_target_properties(GLEW::GLEW PROPERTIES
            IMPORTED_LOCATION "${GLEW_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${GLEW_INCLUDE}"
        )
        set(GLEW_FOUND TRUE)
        message(STATUS "Found GLEW: ${GLEW_LIB}")
    endif()
endif()

# Main library
set(CLUSTERING_SOURCES
    src/core/validate.cpp
    src/core/kmeans.cpp
    src/core/mini_batch.cpp
    src/core/online.cpp
    src/core/dbscan.cpp
    src/dimensionality/pca.cpp
    src/dimensionality/tsne.cpp
    src/operational/drift.cpp
    src/operational/versioning.cpp
    src/operational/rollback.cpp
    src/operational/feature_store.cpp
    src/operational/evaluation.cpp
    src/visualization/renderer.cpp
    src/visualization/mat4.cpp
    src/visualization/shaders.cpp
    src/visualization/text.cpp
    src/compute/distance_avx2.cpp
    src/compute/thread_pool.cpp
    src/gui/data_table.cpp
    src/gui/column_stats.cpp
    src/gui/csv_importer.cpp
    src/gui/preprocess_pipeline.cpp
    src/gui/missing_handler.cpp
)

add_library(clustering ${CLUSTERING_SOURCES})

target_include_directories(clustering PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/nanoflann
)

# Platform-specific linking
find_package(Threads REQUIRED)
target_link_libraries(clustering PRIVATE Threads::Threads)

if(WIN32)
    target_link_libraries(clustering PRIVATE
        glfw OpenGL::GL GLEW::GLEW
        opengl32 glu32
    )
else()
    target_link_libraries(clustering PRIVATE
        glfw OpenGL::GL GLEW::GLEW
    )
    if(CMAKE_DL_LIBS)
        target_link_libraries(clustering PRIVATE ${CMAKE_DL_LIBS})
    endif()
endif()

if(Eigen3_FOUND)
    target_link_libraries(clustering PUBLIC Eigen3::Eigen)
    find_package(BLAS QUIET)
    if(BLAS_FOUND)
        target_link_libraries(clustering PUBLIC ${BLAS_LIBRARIES})
        target_compile_definitions(clustering PUBLIC EIGEN_USE_BLAS=1)
    endif()
endif()

# Examples
add_executable(basic examples/basic_clustering.cpp)
target_link_libraries(basic clustering)

add_executable(streaming examples/streaming.cpp)
target_link_libraries(streaming clustering)

add_executable(viz examples/visualization.cpp)
target_link_libraries(viz clustering)

# ImGui interactive tool
add_executable(imgui_bench
    src/app/main.cpp
    extern/imgui/imgui.cpp
    extern/imgui/imgui_draw.cpp
    extern/imgui/imgui_tables.cpp
    extern/imgui/imgui_widgets.cpp
    extern/imgui/imgui_impl_glfw.cpp
    extern/imgui/imgui_impl_opengl3.cpp
    extern/tinyfiledialogs.cpp
)
target_include_directories(imgui_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/extern
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(imgui_bench PRIVATE -w)
elseif(MSVC)
    target_compile_options(imgui_bench PRIVATE /w)
endif()

target_link_libraries(imgui_bench clustering)

# Benchmark
add_executable(benchmark benchmarks/benchmark.cpp)
target_link_libraries(benchmark clustering)

# Tests
if(GTest_FOUND)
    enable_testing()
    add_executable(tests
        tests/test_kmeans.cpp
        tests/test_threading.cpp
        tests/test_operations.cpp
        tests/test_comprehensive.cpp
        tests/test_matrix.cpp
        tests/test_dbscan.cpp
        tests/test_evaluation.cpp
    )
    target_link_libraries(tests clustering GTest::gtest_main)
    add_test(NAME clustering_tests COMMAND tests)
endif()

# Install
install(TARGETS clustering
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/clustering DESTINATION include)

# Python bindings
option(BUILD_PYTHON "Build Python bindings" OFF)
if(BUILD_PYTHON)
    find_package(pybind11 REQUIRED)
    pybind11_add_module(_clustering python/bindings.cpp)
    target_link_libraries(_clustering PRIVATE clustering)
    target_include_directories(_clustering PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    install(TARGETS _clustering LIBRARY DESTINATION clustering)
endif()
