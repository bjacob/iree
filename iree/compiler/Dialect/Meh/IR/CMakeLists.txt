# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

iree_add_all_subdirs()

file(GLOB _GLOB_X_TD LIST_DIRECTORIES false RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS *.td)
iree_cc_library(
  NAME
    IR
  HDRS
    "MehOps.h"
  TEXTUAL_HDRS
    "MehOpInterface.h.inc"
    "MehOps.cpp.inc"
    "MehOps.h.inc"
  SRCS
    "MehOps.cpp"
  DEPS
    LLVMSupport
    MLIRIR
    MLIRParser
    MLIRSideEffectInterfaces
    MLIRStandard
    MLIRSupport
    MLIRTransformUtils
  PUBLIC
)

iree_cc_library(
  NAME
    MehDialect
  HDRS
    "MehDialect.h"
  SRCS
    "MehDialect.cpp"
  DEPS
    ::IR
  PUBLIC
)

iree_tablegen_library(
  NAME
    MehInterfaceGen
  TD_FILE
    "MehBase.td"
  OUTS
    -gen-op-interface-decls MehOpInterface.h.inc
    -gen-op-interface-defs MehOpInterface.cpp.inc
)

iree_tablegen_library(
  NAME
    MehOpsGen
  TD_FILE
    "MehOps.td"
  OUTS
    -gen-op-decls MehOps.h.inc
    -gen-op-defs MehOps.cpp.inc
)
