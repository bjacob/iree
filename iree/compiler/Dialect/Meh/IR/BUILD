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

load("//build_tools/bazel:iree_tablegen_doc.bzl", "iree_tablegen_doc")
load("//build_tools/bazel:tblgen.bzl", "gentbl")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["MehBase.td"])

filegroup(
    name = "td_files",
    srcs = glob(["*.td"]),
)

cc_library(
    name = "IR",
    srcs = [
        "MehOps.cpp",
    ],
    hdrs = [
        "MehOps.h",
    ],
    textual_hdrs = [
        "MehOpInterface.h.inc",
        "MehOps.h.inc",
        "MehOps.cpp.inc",
    ],
    deps = [
        ":MehOpsGen",
        ":MehInterfaceGen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "MehDialect",
    srcs = ["MehDialect.cpp"],
    hdrs = ["MehDialect.h"],
    deps = [
        ":IR",
    ],
)

gentbl(
    name = "MehInterfaceGen",
    tbl_outs = [
        ("-gen-op-interface-decls", "MehOpInterface.h.inc"),
        ("-gen-op-interface-defs", "MehOpInterface.cpp.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "MehBase.td",
    td_srcs = [
        ":td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)

gentbl(
    name = "MehOpsGen",
    tbl_outs = [
        ("-gen-op-decls", "MehOps.h.inc"),
        ("-gen-op-defs", "MehOps.cpp.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "MehOps.td",
    td_srcs = [
        ":td_files",
        "@llvm-project//mlir:include/mlir/IR/SymbolInterfaces.td",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)
