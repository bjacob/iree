#include "iree/compiler/Dialect/Meh/IR/MehDialect.h"

#include "iree/compiler/Dialect/Meh/IR/MehOps.h"

namespace mlir {
namespace iree_compiler {
namespace meh {
MehDialect::MehDialect(MLIRContext* context)
    : Dialect(getDialectNamespace(), context, TypeID::get<MehDialect>()) {
  // context->loadDialect<MehDialect>();
#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Dialect/Meh/IR/MehOps.cpp.inc"
      >();
}
}  // namespace meh
}  // namespace iree_compiler
}  // namespace mlir
