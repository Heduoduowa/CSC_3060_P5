cmake_minimum_required(VERSION 3.12)

if(POLICY CMP0104)
    cmake_policy(SET CMP0104 NEW)
endif()

set(CMAKE_CUDA_ARCHITECTURES 75)

if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "CUDA compiler")
endif()

project(code_opt_bonus LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

set(REGULAR_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

# Ablation-selected bonus kernels. The remaining kernels reuse the regular/basic
# implementations to keep the bonus submission small and easier to justify.
set(BONUS_KERNEL_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/graph.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/image_proc.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/matmul.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/matmul_cuda.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/trace_replay.cpp"
)

set(REGULAR_FALLBACK_SOURCES
    "${REGULAR_ROOT}/src/kernel/bitwise.cpp"
    "${REGULAR_ROOT}/src/kernel/blackscholes.cpp"
    "${REGULAR_ROOT}/src/kernel/filter_gradient.cpp"
    "${REGULAR_ROOT}/src/kernel/grff.cpp"
    "${REGULAR_ROOT}/src/kernel/relu.cpp"
    "${REGULAR_ROOT}/src/kernel/sparse_spmm.cpp"
)

add_library(kernels STATIC
    ${BONUS_KERNEL_SOURCES}
    ${REGULAR_FALLBACK_SOURCES}
)

target_include_directories(kernels
    PUBLIC
        "${REGULAR_ROOT}/include"
)

set(BONUS_CXX_FLAGS
    -O3 -march=native -mtune=native -ffast-math -funroll-loops -DNDEBUG -Wall -fno-omit-frame-pointer
)
set(REGULAR_CXX_FLAGS
    -O2 -ffast-math -Wall -g -fno-omit-frame-pointer
)

set_source_files_properties(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/graph.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/image_proc.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/trace_replay.cpp"
    PROPERTIES COMPILE_OPTIONS "${BONUS_CXX_FLAGS}"
)

# The CUDA matmul wrapper is sensitive to reassociation around near-zero outputs;
# keep host-side repair in strict FP while the CUDA kernel uses round-to-nearest ops.
set_source_files_properties(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/matmul.cpp"
    PROPERTIES COMPILE_OPTIONS "-O2;-fno-fast-math;-Wall;-fno-omit-frame-pointer"
)

set_source_files_properties(
    ${REGULAR_FALLBACK_SOURCES}
    PROPERTIES COMPILE_OPTIONS "${REGULAR_CXX_FLAGS}"
)

target_compile_options(kernels
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-allow-unsupported-compiler>
)

target_link_libraries(kernels PUBLIC Threads::Threads)

add_executable(single_bench "${REGULAR_ROOT}/src/main/single_bench.cpp")
target_link_libraries(single_bench PRIVATE kernels)

add_executable(run_all "${REGULAR_ROOT}/src/main/run_all.cpp")
target_compile_definitions(run_all PRIVATE BONUS_NORMALIZED_GM=1)
target_link_libraries(run_all PRIVATE kernels)
