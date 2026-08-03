cmake_minimum_required(VERSION 3.16)
project(clustering-engine VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ==============================================================================
# Options
# ==============================================================================
option(BUILD_EXAMPLES    "Build example executables"           ON)
option(BUILD_BENCHMARK   "Build benchmark executable"          ON)
option(BUILD_GUI         "Build interactive GUI (requires GLFW/GLEW/OpenGL)" ON)
option(BUILD_TESTS       "Build test suite (requires GTest)"   ON)
option(BUILD_PYTHON      "Build Python bindings (requires pybind11)" OFF)
option(CLUSTERING_DEBUG  "Enable detailed debug logging"       OFF)

# ==============================================================================
# Compiler flags
# ==============================================================================
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -Wall -Wextra -fPIC)
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
    add_compile_options(/W4 /O2 /bigobj)
    # MSVC doesn't support -fPIC, -mavx2, -mfma as flags
    # AVX2 is enabled via /arch:AVX2 for specific files if needed
endif()

if(CLUSTERING_DEBUG)
    add_compile_definitions(CLUSTERING_DEBUG)
    message(STATUS "Clustering debug logging: ENABLED")
endif()

# ==============================================================================
# Dependencies
# ==============================================================================
find_package(Eigen3 QUIET)
find_package(Threads REQUIRED)
find_package(OpenGL REQUIRED)

# GLFW — find_package first, then vcpkg fallback on Windows
find_package(glfw3 3.3 QUIET)
if(NOT glfw3_FOUND AND WIN32)
    find_library(GLFW_LIB glfw3
        PATHS $ENV{VCPKG_ROOT}/installed/x64-windows/lib)
    find_path(GLFW_INCLUDE glfw3.h
        PATHS $ENV{VCPKG_ROOT}/installed/x64-windows/include)
    if(GLFW_LIB AND GLFW_INCLUDE)
        add_library(glfw UNKNOWN IMPORTED)
        set_target_properties(glfw PROPERTIES
            IMPORTED_LOCATION "${GLFW_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${GLFW_INCLUDE}")
        set(glfw3_FOUND TRUE)
        message(STATUS "Found GLFW: ${GLFW_LIB}")
    endif()
endif()

# GLEW — find_package first, then vcpkg fallback on Windows
find_package(GLEW QUIET)
if(NOT GLEW_FOUND AND WIN32)
    find_library(GLEW_LIB GLEW
        PATHS $ENV{VCPKG_ROOT}/installed/x64-windows/lib)
    find_path(GLEW_INCLUDE GL/glew.h
        PATHS $ENV{VCPKG_ROOT}/installed/x64-windows/include)
    if(GLEW_LIB AND GLEW_INCLUDE)
        add_library(GLEW::GLEW UNKNOWN IMPORTED)
        set_target_properties(GLEW::GLEW PROPERTIES
            IMPORTED_LOCATION "${GLEW_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${GLEW_INCLUDE}")
        set(GLEW_FOUND TRUE)
        message(STATUS "Found GLEW: ${GLEW_LIB}")
    endif()
endif()

# GTest (optional)
find_package(GTest QUIET)

# ==============================================================================
# Core library (standalone, no GUI dependencies)
# ==============================================================================
set(CLUSTERING_CORE_SOURCES
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
    src/compute/distance_avx2.cpp
    src/compute/thread_pool.cpp
)

add_library(clustering ${CLUSTERING_CORE_SOURCES})

target_include_directories(clustering PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(clustering PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/nanoflann
)

target_link_libraries(clustering PUBLIC Threads::Threads)

if(WIN32)
    target_link_libraries(clustering PUBLIC opengl32)
endif()

# Optional Eigen + BLAS
if(Eigen3_FOUND)
    target_link_libraries(clustering PUBLIC Eigen3::Eigen)
    find_package(BLAS QUIET)
    if(BLAS_FOUND)
        target_link_libraries(clustering PUBLIC ${BLAS_LIBRARIES})
        target_compile_definitions(clustering PUBLIC EIGEN_USE_BLAS=1)
    endif()
endif()

# ==============================================================================
# GUI library (optional, requires GLFW/GLEW/OpenGL)
# ==============================================================================
if(BUILD_GUI AND glfw3_FOUND AND GLEW_FOUND)
    set(CLUSTERING_GUI_SOURCES
        src/visualization/renderer.cpp
        src/visualization/mat4.cpp
        src/visualization/shaders.cpp
        src/visualization/text.cpp
        src/gui/data_table.cpp
        src/gui/column_stats.cpp
        src/gui/csv_importer.cpp
        src/gui/preprocess_pipeline.cpp
        src/gui/missing_handler.cpp
    )

    add_library(clustering_gui ${CLUSTERING_GUI_SOURCES})

    target_include_directories(clustering_gui PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    target_include_directories(clustering_gui PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/nanoflann
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    target_link_libraries(clustering_gui PUBLIC clustering glfw OpenGL::GL GLEW::GLEW)

    if(WIN32)
        target_link_libraries(clustering_gui PUBLIC opengl32 glu32)
    endif()

    message(STATUS "GUI target: ENABLED")
else()
    message(STATUS "GUI target: DISABLED (GLFW/GLEW not found, set BUILD_GUI=OFF to suppress)")
endif()

# ==============================================================================
# ImGui interactive application
# ==============================================================================
if(BUILD_GUI AND glfw3_FOUND AND GLEW_FOUND)
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
    target_link_libraries(imgui_bench clustering_gui)
endif()

# ==============================================================================
# Examples (optional)
# ==============================================================================
if(BUILD_EXAMPLES)
    add_executable(basic examples/basic_clustering.cpp)
    target_link_libraries(basic clustering)

    add_executable(streaming examples/streaming.cpp)
    target_link_libraries(streaming clustering)

    if(BUILD_GUI AND glfw3_FOUND AND GLEW_FOUND)
        add_executable(viz examples/visualization.cpp)
        target_link_libraries(viz clustering_gui)
    endif()
endif()

# ==============================================================================
# Benchmark (optional)
# ==============================================================================
if(BUILD_BENCHMARK)
    add_executable(benchmark_exe benchmarks/benchmark.cpp)
    target_link_libraries(benchmark_exe clustering)
    set_target_properties(benchmark_exe PROPERTIES OUTPUT_NAME "benchmark")
endif()

# ==============================================================================
# Tests (optional)
# ==============================================================================
if(BUILD_TESTS AND GTest_FOUND)
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

# ==============================================================================
# Install
# ==============================================================================
install(TARGETS clustering
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/clustering DESTINATION include)

# Install nanoflann header (needed by DBSCAN)
install(DIRECTORY extern/nanoflann/ DESTINATION include/nanoflann
    FILES_MATCHING PATTERN "*.hpp")

# Install GUI library if built
if(TARGET clustering_gui)
    install(TARGETS clustering_gui
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)
endif()

# Install executables
if(TARGET imgui_bench)
    install(TARGETS imgui_bench RUNTIME DESTINATION bin)
endif()
if(TARGET basic)
    install(TARGETS basic RUNTIME DESTINATION bin)
endif()
if(TARGET streaming)
    install(TARGETS streaming RUNTIME DESTINATION bin)
endif()
if(TARGET benchmark_exe)
    install(TARGETS benchmark_exe RUNTIME DESTINATION bin)
endif()

# ==============================================================================
# Python bindings (optional)
# ==============================================================================
if(BUILD_PYTHON)
    find_package(pybind11 REQUIRED)
    pybind11_add_module(_clustering python/bindings.cpp)
    target_link_libraries(_clustering PRIVATE clustering)
    target_include_directories(_clustering PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    install(TARGETS _clustering LIBRARY DESTINATION clustering)
endif()
