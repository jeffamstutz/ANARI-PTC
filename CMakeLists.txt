# ======================================================================== #
# Copyright 2024++ Ingo Wald                                               #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(ANARI_PTC VERSION 1.1.0 LANGUAGES C CXX)

# ------------------------------------------------------------------
option(PTC_BUILD_TESTING "Build the MPI smoke tests (run with ctest)" off)

# ------------------------------------------------------------------
find_package(anari 0.15.0 REQUIRED)
# ------------------------------------------------------------------
find_package(MPI REQUIRED)
# ------------------------------------------------------------------

add_subdirectory(cpu_compositing)

if ((${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR}))
  SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
endif()

add_subdirectory(ptc)

if (PTC_BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
