# ======================================================================================================
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights
# reserved. SPDX-License-Identifier: BSD 3-Clause License
# ======================================================================================================

find_package(Threads REQUIRED)

# Add CUDA support option for benchmarks
option(UCXX_BENCHMARKS_ENABLE_CUDA "Enable CUDA support in benchmarks" OFF)
message(VERBOSE "UCXX: Enable CUDA support in benchmarks: ${UCXX_BENCHMARKS_ENABLE_CUDA}")

# ##################################################################################################
# * compiler function -----------------------------------------------------------------------------

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
  )
  target_link_libraries(${CMAKE_BENCH_NAME} PRIVATE ucxx $<TARGET_NAME_IF_EXISTS:conda_env>)

  # Add CUDA support if enabled
  if(UCXX_BENCHMARKS_ENABLE_CUDA)
    target_compile_definitions(${CMAKE_BENCH_NAME} PRIVATE UCXX_BENCHMARKS_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(${CMAKE_BENCH_NAME} PRIVATE CUDA::cudart)
  endif()

  add_custom_command(
    OUTPUT UCXX_BENCHMARKS
    # COMMAND ${CMAKE_BENCH_NAME} --benchmark_out_format=json
    # --benchmark_out=results/${CMAKE_BENCH_NAME}.json
    COMMAND ${CMAKE_BENCH_NAME}
    APPEND
    COMMENT "Adding ${CMAKE_BENCH_NAME}"
  )

  install(
    TARGETS ${CMAKE_BENCH_NAME}
    COMPONENT benchmarks
    DESTINATION bin/benchmarks/libucxx
    EXCLUDE_FROM_ALL
  )
endfunction()

# ##################################################################################################
# * perftest benchmarks ----------------------------------------------------------------------------
ConfigureBench(ucxx_perftest perftest.cpp)

add_custom_target(
  run_benchmarks
  DEPENDS UCXX_BENCHMARKS
  COMMENT "Custom command for running ucxx benchmarks."
)
