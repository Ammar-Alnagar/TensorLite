cmake_minimum_required(VERSION 3.20)
project(TensorLite VERSION 0.1.0 LANGUAGES CXX)

# Standard C++ configuration
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(TENSORLITE_ENABLE_CUDA "Enable native CUDA compilation" OFF)
option(TENSORLITE_BUILD_TESTS "Build TensorLite test suite" ON)

if(TENSORLITE_ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES "80;89;90")
    add_compile_definitions(TENSORLITE_USE_CUDA)
    message(STATUS "TensorLite: CUDA mode enabled (Architectures: ${CMAKE_CUDA_ARCHITECTURES})")
else()
    message(STATUS "TensorLite: CPU Simulation mode enabled (No GPU required)")
endif()

# Compiler warnings and optimization flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter -O3)
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Core sources
set(CORE_SOURCES
    src/core/types.cpp
    src/core/cuda_compat.cpp
    src/model/safetensors_loader.cpp
    src/kernel/attention.cpp
    src/model/model_runner.cpp
    src/scheduler/scheduler.cpp
    src/cache/prefix_cache.cpp
    src/engine/engine.cpp
)

add_library(tensorlite_core STATIC ${CORE_SOURCES})
target_include_directories(tensorlite_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(TENSORLITE_ENABLE_CUDA)
    target_link_libraries(tensorlite_core PUBLIC CUDA::cudart)
endif()

# Fetch external dependencies: nlohmann_json
include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)
target_link_libraries(tensorlite_core PUBLIC nlohmann_json::nlohmann_json)

# Standalone CLI Executable
add_executable(tensorlite_cli src/cli/main.cpp)
target_link_libraries(tensorlite_cli PRIVATE tensorlite_core)

if(TENSORLITE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
