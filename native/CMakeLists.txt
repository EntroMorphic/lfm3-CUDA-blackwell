cmake_minimum_required(VERSION 3.24)
set(CMAKE_SUPPRESS_REGENERATION ON)
project(liquid_native LANGUAGES CXX CUDA)
option(LIQUID_PROFILE_NVTX "Enable diagnostic host-side NVTX component ranges" OFF)
option(LIQUID_LFM3_OPS_TESTS "Build the isolated LFM3 operator test bridge" OFF)

find_package(CUDAToolkit REQUIRED)

add_executable(liquid-lfm3 src/lfm3_main.cu src/safetensors.cpp)
target_include_directories(liquid-lfm3 PRIVATE include)
target_compile_features(liquid-lfm3 PRIVATE cxx_std_20 cuda_std_20)
target_compile_options(liquid-lfm3 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=false>)
target_link_libraries(liquid-lfm3 PRIVATE CUDA::cudart CUDA::cublas)
set_target_properties(liquid-lfm3 PROPERTIES CUDA_ARCHITECTURES 110)

add_executable(liquid-lfm3-worker src/lfm3_worker.cu src/safetensors.cpp)
target_include_directories(liquid-lfm3-worker PRIVATE include)
target_compile_features(liquid-lfm3-worker PRIVATE cxx_std_20 cuda_std_20)
target_compile_options(liquid-lfm3-worker PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=false>)
target_link_libraries(liquid-lfm3-worker PRIVATE CUDA::cudart CUDA::cublas)
set_target_properties(liquid-lfm3-worker PROPERTIES CUDA_ARCHITECTURES 110)

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
  add_executable(lfm3-serving-ops-probe tests/lfm3_serving_ops_probe.cu src/safetensors.cpp)
  target_include_directories(lfm3-serving-ops-probe PRIVATE include)
  target_compile_features(lfm3-serving-ops-probe PRIVATE cxx_std_20 cuda_std_20)
  target_compile_options(lfm3-serving-ops-probe PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=false>)
  target_link_libraries(lfm3-serving-ops-probe PRIVATE CUDA::cudart CUDA::cublas)
  set_target_properties(lfm3-serving-ops-probe PROPERTIES CUDA_ARCHITECTURES 110)
  add_executable(text-sampling-oracle tests/text_sampling_oracle.cpp)
  target_include_directories(text-sampling-oracle PRIVATE include)
  target_compile_features(text-sampling-oracle PRIVATE cxx_std_17)
  add_executable(lfm3-projection-probe tests/lfm3_projection_probe.cu)
  target_link_libraries(lfm3-projection-probe PRIVATE CUDA::cudart CUDA::cublas)
  set_target_properties(lfm3-projection-probe PROPERTIES CUDA_ARCHITECTURES 110)
  add_library(lfm3_ops SHARED tests/lfm3_ops_bridge.cu)
  target_include_directories(lfm3_ops PRIVATE include)
  target_compile_features(lfm3_ops PRIVATE cuda_std_20)
  target_compile_options(lfm3_ops PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
  set_target_properties(lfm3_ops PROPERTIES CUDA_ARCHITECTURES 110 CUDA_RUNTIME_LIBRARY Static)
endif()
