# =============================================================================
# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

# ##################################################################################################
# * compiler function -----------------------------------------------------------------------------

# This function takes in a test name and test source and handles setting all of the associated
# properties and linking to build the test
function(ConfigureTest CMAKE_TEST_NAME)
  add_executable(${CMAKE_TEST_NAME} ${ARGN})
  set_target_properties(
    ${CMAKE_TEST_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${UCXX_BINARY_DIR}/gtests>"
               INSTALL_RPATH "\$ORIGIN/../../../lib"
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
  )

  target_include_directories(
      ${CMAKE_TEST_NAME} PUBLIC "$<BUILD_INTERFACE:${UCXX_SOURCE_DIR}>"
                                "$<BUILD_INTERFACE:${UCXX_SOURCE_DIR}/src>"
  )

  target_link_libraries(
    ${CMAKE_TEST_NAME} PRIVATE ucxx GTest::gmock_main GTest::gtest_main
                               $<TARGET_NAME_IF_EXISTS:conda_env>
  )
  add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})
  install(
    TARGETS ${CMAKE_TEST_NAME}
    COMPONENT testing
    DESTINATION bin/gtests/libucxx
    EXCLUDE_FROM_ALL
  )
endfunction()

# ##################################################################################################
# test sources ##################################################################################
# ##################################################################################################

# ##################################################################################################
# * ucxx tests ------------------------------------------------------------------------------------
ConfigureTest(
  UCXX_TEST
  buffer.cpp
  config.cpp
  context.cpp
  endpoint.cpp
  header.cpp
  listener.cpp
  request.cpp
  utils.cpp
  worker.cpp
)

# ##################################################################################################
# enable testing ################################################################################
# ##################################################################################################

enable_testing()
