# SPDX-License-Identifier: Apache-2.0

## MPI smoke test for the ptc device (wrapping helide) ##
#
# Each test runs ptcSmoke under mpiexec. Every rank renders one sphere through
# ptc; rank 0 checks the composited image and all ranks exit non-zero if any
# check failed. The tests set LD_LIBRARY_PATH themselves, so no manual
# environment setup is needed.

if (NOT MPIEXEC_EXECUTABLE)
  message(WARNING "#ptc: mpiexec not found, skipping tests")
  return()
endif()

include(GNUInstallDirs)

# helide is not an exported CMake target; look for it next to the anari
# install that find_package(anari) found.
get_filename_component(PTC_ANARI_PREFIX "${anari_DIR}/../../.." ABSOLUTE)
find_library(PTC_HELIDE_LIBRARY
  NAMES anari_library_helide
  HINTS "${PTC_ANARI_PREFIX}/${CMAKE_INSTALL_LIBDIR}" "${PTC_ANARI_PREFIX}/lib"
)
if (NOT PTC_HELIDE_LIBRARY)
  message(WARNING "#ptc: helide (anari_library_helide) not found, skipping tests")
  return()
endif()
get_filename_component(PTC_HELIDE_DIR "${PTC_HELIDE_LIBRARY}" DIRECTORY)
message("#ptc: testing with mpiexec ${MPIEXEC_EXECUTABLE} and ${PTC_HELIDE_LIBRARY}")

add_executable(ptcSmoke ptcSmoke.cpp)
target_link_libraries(ptcSmoke PRIVATE anari::anari MPI::MPI_CXX)
# ptc is loaded at runtime by anari::loadLibrary, so make sure it is built
add_dependencies(ptcSmoke anari_library_ptc)

set(PTC_TEST_RANKS 1 2 3)
set(PTC_TEST_COLORS ufixed8 float32)

# Run ptcSmoke on <ranks> ranks with the given extra arguments.
function(ptc_smoke_command out ranks)
  set(${out}
    ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:ptcSmoke> ${MPIEXEC_POSTFLAGS}
    --expect-library $<TARGET_FILE:anari_library_ptc> ${ARGN}
    PARENT_SCOPE)
endfunction()

# The freshly built ptc must come first on the search path: an installed
# libanari_library_ptc.so may sit next to helide. ptcSmoke also checks which
# ptc library it actually loaded (--expect-library).
function(ptc_set_test_environment test ranks)
  set(libdirs "$<TARGET_FILE_DIR:anari_library_ptc>" "${PTC_HELIDE_DIR}")
  set_tests_properties(${test} PROPERTIES
    TIMEOUT 120
    PROCESSORS ${ranks}
    ENVIRONMENT "PTC_ANARI_LIBRARY=helide")
  if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.22)
    # each prepend goes in front of the previous one, so add in reverse
    set(mods "")
    list(REVERSE libdirs)
    foreach(dir ${libdirs})
      list(APPEND mods "LD_LIBRARY_PATH=path_list_prepend:${dir}")
    endforeach()
    set_tests_properties(${test} PROPERTIES ENVIRONMENT_MODIFICATION "${mods}")
  else()
    # Note: this captures LD_LIBRARY_PATH at configure time.
    string(REPLACE ";" ":" path "${libdirs}")
    if (DEFINED ENV{LD_LIBRARY_PATH} AND NOT "$ENV{LD_LIBRARY_PATH}" STREQUAL "")
      string(APPEND path ":$ENV{LD_LIBRARY_PATH}")
    endif()
    set_property(TEST ${test} APPEND PROPERTY ENVIRONMENT
      "LD_LIBRARY_PATH=${path}")
  endif()
endfunction()

# Add ptc_smoke_<name>_<ranks>ranks for each rank count. If any color the test
# renders is float32, the CPU build expects it to fail: the CPU compositor does
# not support FLOAT32_VEC4 yet (ticket 08). Only that specific error counts as
# a pass, so unrelated failures still show up.
function(ptc_add_smoke_tests name)
  foreach(ranks ${PTC_TEST_RANKS})
    set(test ptc_smoke_${name}_${ranks}ranks)
    ptc_smoke_command(cmd ${ranks} ${ARGN})
    add_test(NAME ${test} COMMAND ${cmd})
    ptc_set_test_environment(${test} ${ranks})
    if (NOT USE_CUDA AND "float32" IN_LIST ARGN)
      set_tests_properties(${test} PROPERTIES PASS_REGULAR_EXPRESSION
        "support for FLOAT32_VEC4 color channel not implemented")
    endif()
  endforeach()
endfunction()

# Each color runs twice: without channel.depth (mapping depth must be refused)
# and with it (--depth: composited depth must be correct).
foreach(color ${PTC_TEST_COLORS})
  ptc_add_smoke_tests(${color} --color ${color})
  ptc_add_smoke_tests(${color}_depth --color ${color} --depth)
endforeach()

# Frame parameters changing between two frames (ticket 05). Each also flips
# channel.depth between the frames.
foreach(color ${PTC_TEST_COLORS})
  ptc_add_smoke_tests(${color}_resize --color ${color} --resize)
endforeach()
ptc_add_smoke_tests(retype_float32_to_ufixed8
  --first-color float32 --color ufixed8 --depth)
ptc_add_smoke_tests(retype_ufixed8_to_float32
  --first-color ufixed8 --color float32 --depth)
# same pixel size, so this is the color type change the CPU path can run
ptc_add_smoke_tests(retype_ufixed8_linear_to_ufixed8
  --first-color ufixed8-linear --color ufixed8)
# no channel.color: nothing to map, but depth still composites
ptc_add_smoke_tests(nocolor_depth --color none --depth)
ptc_add_smoke_tests(unset_color --first-color ufixed8 --color none --depth)
ptc_add_smoke_tests(unset_all
  --unset-all --resize --first-color ufixed8-linear --color ufixed8)
ptc_add_smoke_tests(unset_all_depth --unset-all --color ufixed8 --depth)
foreach(color ${PTC_TEST_COLORS})
  ptc_add_smoke_tests(${color}_uncommitted --color ${color} --uncommitted)
endforeach()

# Frame completion callback (ticket 06): called once per render with the ptc
# handles, after compositing; not called once unset.
ptc_add_smoke_tests(callback --color ufixed8 --callback)
ptc_add_smoke_tests(callback_resize --color ufixed8 --resize --depth --callback)

# Failure path: with inverted expectations the pixel check must fail and
# every rank must exit non-zero.
set(ranks 2)
set(test ptc_smoke_detects_failure_${ranks}ranks)
ptc_smoke_command(cmd ${ranks} --invert-check)
string(REPLACE ";" "$<SEMICOLON>" cmd "${cmd}")
add_test(NAME ${test}
  COMMAND ${CMAKE_COMMAND} -DRANKS=${ranks} -DCOMMAND=${cmd}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/ExpectFailure.cmake)
ptc_set_test_environment(${test} ${ranks})
