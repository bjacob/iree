// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This is an experimental code for now in experimental branch to play with
// different lowering of linalg.matmul

#include "iree/compiler/Dialect/Meh/Conversion/MehToSCF/ConvertMehToSCF.h"

#include "iree/compiler/Dialect/Meh/IR/MehDialect.h"
#include "iree/compiler/Dialect/Meh/IR/MehOps.h"
#include "mlir/Dialect/SCF/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace meh {
namespace {

class MehMatmulToSCFPattern : public OpRewritePattern<meh::MatmulOp> {
 public:
  using OpRewritePattern<meh::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(meh::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    auto lhsVal = op.lhs();
    auto rhsVal = op.rhs();
    auto dstVal = op.dst();
    auto dstValType = dstVal.getType().cast<MemRefType>();
    auto dstShape = dstValType.getShape();  // ArrayRef<int64_t>
    auto M = dstShape[0];
    auto N = dstShape[1];
    auto rhsValType = dstVal.getType().cast<MemRefType>();
    auto rhsShape = rhsValType.getShape();

    edsc::ScopedContext scope(rewriter, op.getLoc());

    auto K = rhsShape[0];

    Value zero = edsc::intrinsics::std_constant_index(0);
    Value step = edsc::intrinsics::std_constant_index(1);

    Value boundM = edsc::intrinsics::std_constant_index(M);
    Value boundN = edsc::intrinsics::std_constant_index(N);
    Value boundK = edsc::intrinsics::std_constant_index(K);

    edsc::loopNestBuilder(zero, boundM, step, [&](Value m) {
      edsc::loopNestBuilder(zero, boundN, step, [&](Value n) {
        edsc::loopNestBuilder(zero, boundK, step, [&](Value k) {
          Value lhs_val =
              edsc::intrinsics::std_load(lhsVal, ArrayRef<Value>{m, k});
          Value rhs_val =
              edsc::intrinsics::std_load(rhsVal, ArrayRef<Value>{k, n});
          Value dst_val =
              edsc::intrinsics::std_load(dstVal, ArrayRef<Value>{m, n});
          Value mul_f = edsc::intrinsics::std_mulf(lhs_val, rhs_val);
          Value res = edsc::intrinsics::std_addf(dst_val, mul_f);
          edsc::intrinsics::std_store(res, dstVal, ArrayRef<Value>{m, n});
        });
      });
    });

    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertMehMatmulToSCFPass
    : public PassWrapper<ConvertMehMatmulToSCFPass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<meh::MehDialect, scf::SCFDialect>();
  }
  void runOnFunction() override;
};

}  // namespace

void ConvertMehMatmulToSCFPass::runOnFunction() {
  auto funcOp = getOperation();
  MLIRContext *context = &getContext();

  OwningRewritePatternList toSCFConversionPatterns;
  toSCFConversionPatterns.insert<MehMatmulToSCFPattern>(context);
  applyPatternsAndFoldGreedily(
      funcOp,
      std::move(toSCFConversionPatterns));
}

std::unique_ptr<FunctionPass> createConvertMehMatmulToSCFPass() {
  return std::make_unique<ConvertMehMatmulToSCFPass>();
}

void registerConvertMehMatmulToSCFPass() {
  PassRegistration<ConvertMehMatmulToSCFPass> registration( "convert-meh-matmul-to-scf",
    "Convert meh matmul to scf",
    [] { return createConvertMehMatmulToSCFPass(); });
}

}  // namespace meh
}  // namespace iree_compiler
}  // namespace mlir
