cmake_minimum_required(VERSION 3.18)

project(P1RV_CUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)

find_package(CUDAToolkit REQUIRED)

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/P1RV_CUDA")

# ===== CPU LIB ONLY =====
add_library(cuda_core STATIC
    ${SRC_DIR}/lsmc.cpp
    ${SRC_DIR}/gbm.cpp
    ${SRC_DIR}/fdm.cpp
)

target_include_directories(cuda_core PUBLIC
    ${SRC_DIR}
)

# ===== CUDA EXECUTABLE =====
add_executable(P1RV_CUDA
    ${SRC_DIR}/main.cu
    ${SRC_DIR}/gbm.cu
    ${SRC_DIR}/lsmc.cu
)

set_target_properties(P1RV_CUDA PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(P1RV_CUDA PRIVATE
    ${SRC_DIR}
)

find_package(OpenMP REQUIRED)

target_link_libraries(cuda_core PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(P1RV_CUDA PRIVATE
    cuda_core
    CUDA::cudart
    CUDA::curand
    OpenMP::OpenMP_CXX
)

if(MSVC)
    target_compile_options(P1RV_CUDA PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/openmp>)
endif()
