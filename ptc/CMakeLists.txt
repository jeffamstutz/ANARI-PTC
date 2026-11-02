# ======================================================================== #
# Copyright 2024__ Ingo Wald                                               #
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

## PTC device target ##

project(anari_library_ptc LANGUAGES C CXX)

project_add_library(MODULE
  MpiExtension.cpp
  PTCDevice.cpp
  PTCLibrary.cpp
  WrappedLibrary.cpp
  FrameWrapper.cpp
)
project_link_libraries(
  cpu_compositing
  anari::helium
  anari::anari
)

# ptc reports its own version as the device's 'version' property
target_compile_definitions(${PROJECT_NAME} PRIVATE
  PTC_VERSION_MAJOR=${ANARI_PTC_VERSION_MAJOR}
  PTC_VERSION_MINOR=${ANARI_PTC_VERSION_MINOR}
  PTC_VERSION_PATCH=${ANARI_PTC_VERSION_PATCH}
  PTC_VERSION_STRING="${ANARI_PTC_VERSION}")

include(GenerateExportHeader)
generate_export_header(${PROJECT_NAME} EXPORT_MACRO_NAME "PTC_EXPORT")
project_include_directories(PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

## Installation ##

include(GNUInstallDirs)

install(TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
