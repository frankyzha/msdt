cmake_minimum_required(VERSION 3.15...3.27)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

include(CheckIPOSupported)

set(CMAKE_CXX_STANDARD 20)

option(SPLIT_ENABLE_PGO_GENERATE "Build with profile generation enabled." OFF)
option(SPLIT_ENABLE_PGO_USE "Build with profile use enabled." OFF)
set(SPLIT_PGO_DIR "" CACHE PATH "Directory for compiler profile data.")

if (SPLIT_ENABLE_PGO_GENERATE AND SPLIT_ENABLE_PGO_USE)
  message(FATAL_ERROR "SPLIT_ENABLE_PGO_GENERATE and SPLIT_ENABLE_PGO_USE cannot both be ON.")
endif()

if (SPLIT_PGO_DIR STREQUAL "")
  set(SPLIT_PGO_DIR "${CMAKE_BINARY_DIR}/pgo-data" CACHE PATH "Directory for compiler profile data." FORCE)
endif()

set(TARGET "_libgosdt")
set(CLI_TARGET "gosdt_cli")
set(
  MSPLIT_INLINE_SOURCE_REGEX
  "src/libgosdt/src/(msplit\\.cpp|msplit_(debug|atomized|atomized_support|atomized_heuristic|linear|nonlinear)\\.cpp)$")

function(configure_local_release_optimizations target_name)
  if (CMAKE_BUILD_TYPE STREQUAL "Release")
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
      target_compile_options(${target_name} PRIVATE -march=native -mtune=native)
      set(apply_pgo FALSE)
      if (SKBUILD)
        if ("${target_name}" STREQUAL "${TARGET}")
          set(apply_pgo TRUE)
        endif()
      elseif ("${target_name}" STREQUAL "${CLI_TARGET}")
        set(apply_pgo TRUE)
      endif()
      if (apply_pgo)
        if (SPLIT_ENABLE_PGO_GENERATE)
          target_compile_options(${target_name} PRIVATE -fprofile-generate=${SPLIT_PGO_DIR})
          target_link_options(${target_name} PRIVATE -fprofile-generate=${SPLIT_PGO_DIR})
        elseif (SPLIT_ENABLE_PGO_USE)
          target_compile_options(
            ${target_name}
            PRIVATE
              -fprofile-use=${SPLIT_PGO_DIR}
              -fprofile-correction
              -Wno-error=coverage-mismatch)
          target_link_options(${target_name} PRIVATE -fprofile-use=${SPLIT_PGO_DIR})
        endif()
      endif()
    endif()
    check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
    if (ipo_supported)
      set_property(TARGET ${target_name} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
      message(STATUS "IPO/LTO not enabled for ${target_name}: ${ipo_error}")
    endif()
  endif()
endfunction()

message(STATUS "PATH environment variable: $ENV{PATH}")
message(STATUS "LD_LIBRARY_PATH environment variable: $ENV{LD_LIBRARY_PATH}")

# Python Target
if (SKBUILD)
  # Find the module development requirements (requires FindPython from 3.17 or
  # scikit-build-core's built-in backport)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)

  # Python Target
  file(GLOB_RECURSE SOURCE_FILES src/libgosdt/src/*.cpp src/libgosdt.cpp)
  list(FILTER SOURCE_FILES EXCLUDE REGEX "${MSPLIT_INLINE_SOURCE_REGEX}")
  python_add_library(${TARGET} MODULE ${SOURCE_FILES} WITH_SOABI)
  target_link_libraries(${TARGET} PRIVATE pybind11::module)
  target_include_directories(${TARGET} PUBLIC src/libgosdt/include/)
  set_target_properties(${TARGET} PROPERTIES INSTALL_RPATH "@loader_path")
  set_target_properties(${TARGET} PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
  configure_local_release_optimizations(${TARGET})
  if (APPLE AND DEFINED ENV{SPLIT_LOCALDEPS_LIBDIR})
    set_property(TARGET ${TARGET} APPEND PROPERTY INSTALL_RPATH "$ENV{SPLIT_LOCALDEPS_LIBDIR}")
  endif()
  install(TARGETS ${TARGET} DESTINATION ${SKBUILD_PROJECT_NAME})

  # In the case where wer're running on Windows with packages installed via vcpkg, we include
  # the vcpkg.cmake file to find the packages.
  if (WIN32)
    include($ENV{VCPKG_INSTALLATION_ROOT}/scripts/buildsystems/vcpkg.cmake)
  endif()

endif()

# CLI Target
file(GLOB_RECURSE SOURCE_FILES src/libgosdt/src/*.cpp src/cli.cpp)
list(FILTER SOURCE_FILES EXCLUDE REGEX "${MSPLIT_INLINE_SOURCE_REGEX}")
add_executable(${CLI_TARGET} ${SOURCE_FILES})
target_include_directories(${CLI_TARGET} PUBLIC src/libgosdt/include/)
configure_local_release_optimizations(${CLI_TARGET})

#
# Link all needed dependencies to this project:
#
find_package(PkgConfig REQUIRED)
include(FetchContent)

# TBB:
find_package(TBB REQUIRED)
if (SKBUILD)
  target_link_libraries(${TARGET} PRIVATE TBB::tbb)
  target_link_libraries(${TARGET} PRIVATE TBB::tbbmalloc)
endif()
target_link_libraries(${CLI_TARGET} PRIVATE TBB::tbb TBB::tbbmalloc)

# GMP:
pkg_check_modules(GMP REQUIRED IMPORTED_TARGET gmp)
if (SKBUILD)
  target_link_libraries(${TARGET} PRIVATE PkgConfig::GMP)
  target_include_directories(${TARGET} PRIVATE ${GMP_INCLUDE_DIRS})
endif()
target_link_libraries(${CLI_TARGET} PRIVATE PkgConfig::GMP)
target_include_directories(${CLI_TARGET} PRIVATE ${GMP_INCLUDE_DIRS})

# GLPK:
find_library(GLPK_LIBRARY NAMES glpk REQUIRED)
if (SKBUILD)
  target_link_libraries(${TARGET} PRIVATE ${GLPK_LIBRARY})
endif()
target_link_libraries(${CLI_TARGET} PRIVATE ${GLPK_LIBRARY})

# nlohmann::json:
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)
if (SKBUILD)
  target_link_libraries(${TARGET} PRIVATE nlohmann_json::nlohmann_json)
endif()
target_link_libraries(${CLI_TARGET} PRIVATE nlohmann_json::nlohmann_json)

# Generate compile_commands.json for vscode
set(CMAKE_EXPORT_COMPILE_COMMANDS
    ON
    CACHE INTERNAL "")
