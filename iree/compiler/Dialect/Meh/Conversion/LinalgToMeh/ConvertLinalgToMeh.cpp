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

#include "iree/compiler/Dialect/Meh/Conversion/LinalgToMeh/ConvertLinalgToMeh.h"

#include "iree/compiler/Dialect/Meh/IR/MehDialect.h"
#include "iree/compiler/Dialect/Meh/IR/MehOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace meh {
namespace {

class MatmulOpToPaddedMatmulPattern
    : public OpRewritePattern<linalg::MatmulOp> {
 public:
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<meh::PaddedMatmulOp>(op.getLoc(), op.getOperand(0),
                                         op.getOperand(1), op.getOperand(2));
    rewriter.eraseOp(op);
    return success();
  }
};

/*

lina.ops ----> meh.ops
(linalg ops illegal)


*/

struct ConvertMatmulToPaddedMatmulPass
    : public PassWrapper<ConvertMatmulToPaddedMatmulPass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, meh::MehDialect>();
  }
  void runOnFunction() override;
};

}  // namespace

void ConvertMatmulToPaddedMatmulPass::runOnFunction() {
  auto funcOp = getOperation();
  MLIRContext *context = &getContext();

  OwningRewritePatternList conversionPatterns;

  conversionPatterns.insert<MatmulOpToPaddedMatmulPattern>(context);

  applyPatternsAndFoldGreedily(funcOp, std::move(conversionPatterns));
}

std::unique_ptr<FunctionPass> createConvertMatmulToPaddedMatmulPass() {
  return std::make_unique<ConvertMatmulToPaddedMatmulPass>();
}

void registerConvertMatmulToPaddedMatmulPass() {
  PassRegistration<ConvertMatmulToPaddedMatmulPass> registration("convert-linalg-matmul-to-meh-padded-matmul",
    "Convert matmul to meh padded matmul",
    [] { return createConvertMatmulToPaddedMatmulPass(); });
}

}  // namespace meh
}  // namespace iree_compiler
}  // namespace mlir
