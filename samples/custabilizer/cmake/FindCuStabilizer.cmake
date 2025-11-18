# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
#
# SPDX-License-Identifier: BSD-3-Clause

#[=======================================================================[.rst:
FindCuStabilizer
----------------

Finds the NVIDIA cuStabilizer library.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following imported targets, if found:

``cuStabilizer::cuStabilizer``
  The cuStabilizer library

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``CuStabilizer_FOUND``
  True if the system has the cuStabilizer library.
``CUSTABILIZER_INCLUDE_DIR``
  Include directory needed to use cuStabilizer.
``CUSTABILIZER_LIBRARY``
  Library needed to link against cuStabilizer.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``CUSTABILIZER_ROOT``
  The root directory of the cuStabilizer installation (optional hint).

#]=======================================================================]

set(CUSTABILIZER_ROOT "" CACHE PATH "Path to cuStabilizer installation/prefix")

find_path(CUSTABILIZER_INCLUDE_DIR
  NAMES custabilizer.hpp
  HINTS 
    ${CUSTABILIZER_ROOT}
    ${CMAKE_CURRENT_LIST_DIR}/../..
    ENV CUSTABILIZER_ROOT
    ENV CPATH
    ENV C_INCLUDE_PATH
    ENV CPLUS_INCLUDE_PATH
  PATHS
    ENV CMAKE_PREFIX_PATH
    /usr/local
    /usr
  PATH_SUFFIXES include include/custabilizer
  DOC "cuStabilizer include directory"
)

find_library(CUSTABILIZER_LIBRARY
  NAMES custabilizer
  HINTS 
    ${CUSTABILIZER_ROOT}
    ${CMAKE_CURRENT_LIST_DIR}/../..
    ENV CUSTABILIZER_ROOT
    ENV LD_LIBRARY_PATH
    ENV LIBRARY_PATH
  PATHS
    ENV CMAKE_PREFIX_PATH
    /usr/local
    /usr
  PATH_SUFFIXES lib lib64 build/lib
  DOC "cuStabilizer library"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CuStabilizer
  FOUND_VAR CuStabilizer_FOUND
  REQUIRED_VARS
    CUSTABILIZER_LIBRARY
    CUSTABILIZER_INCLUDE_DIR
  VERSION_VAR CuStabilizer_VERSION
)

if(CuStabilizer_FOUND)
  if(NOT TARGET cuStabilizer::cuStabilizer)
    add_library(cuStabilizer::cuStabilizer UNKNOWN IMPORTED)
    set_target_properties(cuStabilizer::cuStabilizer PROPERTIES
      IMPORTED_LOCATION "${CUSTABILIZER_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${CUSTABILIZER_INCLUDE_DIR}"
    )
  endif()
endif()

mark_as_advanced(
  CUSTABILIZER_ROOT
  CUSTABILIZER_INCLUDE_DIR
  CUSTABILIZER_LIBRARY
)

