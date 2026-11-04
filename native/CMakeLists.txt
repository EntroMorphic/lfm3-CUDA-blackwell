cmake_minimum_required(VERSION 3.24)
set(CMAKE_SUPPRESS_REGENERATION ON)
project(liquid_native LANGUAGES CXX CUDA)
option(LIQUID_PROFILE_NVTX "Enable diagnostic host-side NVTX component ranges" OFF)
option(LIQUID_LFM3_OPS_TESTS "Build the isolated LFM3 operator test bridge" OFF)

find_package(CUDAToolkit REQUIRED)

add_executable(liquid-native src/main.cu src/safetensors.cpp src/tokenizer.cpp)
target_include_directories(liquid-native PRIVATE include)
if(LIQUID_PROFILE_NVTX)
  target_compile_definitions(liquid-native PRIVATE LFM_PROFILE_NVTX)
endif()
target_compile_features(liquid-native PRIVATE cxx_std_20)
target_compile_options(liquid-native PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --default-stream=per-thread>
)
target_link_libraries(liquid-native PRIVATE CUDA::cudart CUDA::cublas CUDA::cublasLt pcre2-8)
set_target_properties(liquid-native PROPERTIES
    CUDA_ARCHITECTURES 110
    CUDA_SEPARABLE_COMPILATION ON
)

if(LIQUID_LFM3_OPS_TESTS)
  add_library(lfm3_ops SHARED tests/lfm3_ops_bridge.cu)
  target_include_directories(lfm3_ops PRIVATE include)
  target_compile_features(lfm3_ops PRIVATE cuda_std_20)
  target_compile_options(lfm3_ops PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
  set_target_properties(lfm3_ops PROPERTIES CUDA_ARCHITECTURES 110 CUDA_RUNTIME_LIBRARY Static)
endif()
