// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Modules/Check/IR/CheckDialect.h"
#include "iree/compiler/Dialect/Modules/Check/IR/CheckOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

// Clones each exported functions (including those just created) with
// placeholder constant inputs instead of arguments and removes the exported
// attribute from the old functions.
// The input are provided using util.globals.
class ExportMatmulTestFuncsPass
    : public ExportMatmulTestFuncsBase<ExportMatmulTestFuncsPass> {
 public:
  struct FuncOps {
    mlir::FuncOp actual;
    mlir::FuncOp expected;
    mlir::FuncOp zeroMatrix;
    mlir::FuncOp pseudorandomMatrix;
    mlir::FuncOp identityMatrix;
  };

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<IREE::Util::UtilDialect>();
    registry.insert<IREE::Check::CheckDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    auto findFuncOp = [=](const char* sym_name) mutable {
      for (auto funcOp : moduleOp.getOps<mlir::FuncOp>()) {
        if (funcOp.sym_name() == sym_name) {
          return funcOp;
        }
      }
      llvm::errs() << "No function named `" << sym_name << "` in module.\n";
      signalPassFailure();
      return mlir::FuncOp();
    };

    FuncOps funcOps;
    funcOps.actual = findFuncOp("actual_matmul");
    funcOps.expected = findFuncOp("expected_matmul");
    funcOps.zeroMatrix = findFuncOp("zero_matrix");
    funcOps.pseudorandomMatrix = findFuncOp("pseudorandom_matrix");
    funcOps.identityMatrix = findFuncOp("identity_matrix");

    if (failed(createEntryPointMatmulTestFunc(moduleOp, funcOps))) {
      signalPassFailure();
    }
  }

 private:
  Value createMatrix(Location loc, Value rows, Value cols, FuncOp funcOp,
                     OpBuilder& builder) {
    SmallVector<Value> args({rows, cols});
    auto callOp = builder.create<mlir::CallOp>(loc, funcOp, args);
    return callOp.getResults()[0];
  }

  LogicalResult createEntryPointMatmulTestFunc(mlir::ModuleOp moduleOp,
                                               FuncOps& funcOps) {
    OpBuilder moduleBuilder(&getContext());
    Location loc = funcOps.expected.getLoc();
    moduleBuilder.setInsertionPointAfter(funcOps.expected);

    // Create a `() -> ()` entry point op the MatmulTest tool can run.
    std::string funcName = "matmul_test";
    auto funcOp = moduleBuilder.create<mlir::FuncOp>(
        loc, funcName, moduleBuilder.getFunctionType({}, {}));
    funcOp.setPublic();
    funcOp->setAttr("iree.abi.stub", moduleBuilder.getUnitAttr());
    SmallVector<NamedAttribute> reflectionAttrs = {
        moduleBuilder.getNamedAttr("MatmulTest",
                                   moduleBuilder.getStringAttr("entry")),
    };
    funcOp->setAttr("iree.reflection",
                    moduleBuilder.getDictionaryAttr(reflectionAttrs));
    Block* block = funcOp.addEntryBlock();

    auto blockBuilder = OpBuilder::atBlockBegin(block);
    int m = 10;
    int k = 10;
    int n = 10;
    struct InputFuncOps {
      FuncOp lhs;
      FuncOp rhs;
      FuncOp acc;
    };
    InputFuncOps inputFuncOps;
    inputFuncOps.lhs = funcOps.identityMatrix;
    inputFuncOps.rhs = funcOps.identityMatrix;
    inputFuncOps.acc = funcOps.zeroMatrix;
    Value mConstOp =
        blockBuilder.create<ConstantOp>(loc, blockBuilder.getIndexAttr(m));
    Value kConstOp =
        blockBuilder.create<ConstantOp>(loc, blockBuilder.getIndexAttr(k));
    Value nConstOp =
        blockBuilder.create<ConstantOp>(loc, blockBuilder.getIndexAttr(n));

    Value lhs =
        createMatrix(loc, mConstOp, kConstOp, inputFuncOps.lhs, blockBuilder);
    Value rhs =
        createMatrix(loc, kConstOp, nConstOp, inputFuncOps.rhs, blockBuilder);
    Value acc =
        createMatrix(loc, mConstOp, nConstOp, inputFuncOps.acc, blockBuilder);

    SmallVector<Value> args{lhs, rhs, acc};
    auto actualCallOp =
        blockBuilder.create<mlir::CallOp>(loc, funcOps.actual, args);
    Value actualResult = actualCallOp.getResults()[0];

    auto expectedCallOp =
        blockBuilder.create<mlir::CallOp>(loc, funcOps.expected, args);
    Value expectedResult = expectedCallOp.getResults()[0];

    blockBuilder.create<IREE::Check::ExpectAlmostEqOp>(loc, actualResult,
                                                       expectedResult);

    blockBuilder.create<mlir::ReturnOp>(loc);

    return success();
  }
};

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createExportMatmulTestFuncsPass() {
  return std::make_unique<ExportMatmulTestFuncsPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
