# =============================================================================
# Copyright (c) 2018-2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

find_package(Threads REQUIRED)

# ##################################################################################################
# * compiler function -----------------------------------------------------------------------------

# Use an OBJECT library so we only compile these helper source files only once
# add_library(
#   cudf_benchmark_common OBJECT "${CUDF_SOURCE_DIR}/tests/utilities/base_fixture.cpp"
#                                synchronization/synchronization.cpp io/cuio_common.cpp
# )
# target_link_libraries(cudf_benchmark_common PRIVATE cudf_datagen $<TARGET_NAME_IF_EXISTS:conda_env>)
add_custom_command(
  OUTPUT UCXX_BENCHMARKS
  COMMAND echo Running benchmarks
  COMMAND mkdir -p results
  VERBATIM
  COMMENT "Running ucxx benchmarks."
  USES_TERMINAL
)

# This function takes in a benchmark name and benchmark source and handles setting all of the
# associated properties and linking to build the benchmark
function(ConfigureBench CMAKE_BENCH_NAME)
  add_executable(${CMAKE_BENCH_NAME} ${ARGN})
  set_target_properties(
    ${CMAKE_BENCH_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${UCXX_BINARY_DIR}/benchmarks>"
               INSTALL_RPATH "\$ORIGIN/../../../lib"
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               # For std:: support of __int128_t. Can be removed once using cuda::std
               CXX_EXTENSIONS ON
               CUDA_STANDARD 17
               CUDA_STANDARD_REQUIRED ON
  )
  target_link_libraries(
    ${CMAKE_BENCH_NAME} PRIVATE ucxx
                                $<TARGET_NAME_IF_EXISTS:conda_env>
  )
  add_custom_command(
    OUTPUT UCXX_BENCHMARKS
    # COMMAND ${CMAKE_BENCH_NAME} --benchmark_out_format=json
    #         --benchmark_out=results/${CMAKE_BENCH_NAME}.json
    COMMAND ${CMAKE_BENCH_NAME}
    APPEND
    COMMENT "Adding ${CMAKE_BENCH_NAME}"
  )

  install(
    TARGETS ${CMAKE_BENCH_NAME}
    COMPONENT testing
    DESTINATION bin/benchmarks/ucxx
    EXCLUDE_FROM_ALL
  )
endfunction()

# # This function takes in a benchmark name and benchmark source for nvbench benchmarks and handles
# # setting all of the associated properties and linking to build the benchmark
# function(ConfigureNVBench CMAKE_BENCH_NAME)
#   add_executable(${CMAKE_BENCH_NAME} ${ARGN})
#   set_target_properties(
#     ${CMAKE_BENCH_NAME}
#     PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${UCXX_BINARY_DIR}/benchmarks>"
#                INSTALL_RPATH "\$ORIGIN/../../../lib"
#   )
#   target_link_libraries(
#     ${CMAKE_BENCH_NAME} PRIVATE cudf_benchmark_common cudf_datagen nvbench::main
#                                 $<TARGET_NAME_IF_EXISTS:conda_env>
#   )
#   install(
#     TARGETS ${CMAKE_BENCH_NAME}
#     COMPONENT testing
#     DESTINATION bin/benchmarks/libcudf
#     EXCLUDE_FROM_ALL
#   )
# endfunction()

# ##################################################################################################
# * perftest benchmarks ----------------------------------------------------------------------------
ConfigureBench(ucxx_perftest perftest.cpp)

add_custom_target(
  run_benchmarks
  DEPENDS UCXX_BENCHMARKS
  COMMENT "Custom command for running ucxx benchmarks."
)