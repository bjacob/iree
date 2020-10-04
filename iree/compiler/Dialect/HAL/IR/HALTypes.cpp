// Copyright 2019 Google LLC
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

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"

#include "llvm/ADT/StringExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

// Order matters:
#include "iree/compiler/Dialect/HAL/IR/HALEnums.cpp.inc"
#include "iree/compiler/Dialect/HAL/IR/HALStructs.cpp.inc"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

//===----------------------------------------------------------------------===//
// Enum utilities
//===----------------------------------------------------------------------===//

template <typename T>
static LogicalResult parseEnumAttr(DialectAsmParser &parser, StringRef attrName,
                                   Attribute &attr) {
  Attribute genericAttr;
  auto loc = parser.getCurrentLocation();
  if (failed(parser.parseAttribute(genericAttr,
                                   parser.getBuilder().getNoneType()))) {
    return parser.emitError(loc)
           << "failed to parse '" << attrName << "' enum string value";
  }
  auto stringAttr = genericAttr.dyn_cast<StringAttr>();
  if (!stringAttr) {
    return parser.emitError(loc)
           << "expected " << attrName << " attribute specified as string";
  }
  auto symbolized = symbolizeEnum<T>(stringAttr.getValue());
  if (!symbolized.hasValue()) {
    return parser.emitError(loc)
           << "failed to parse '" << attrName << "' enum value";
  }
  attr = parser.getBuilder().getI32IntegerAttr(
      static_cast<int32_t>(symbolized.getValue()));
  return success();
}

// LINT.IfChange(element_type)
namespace {
enum class NumericalType : uint32_t {
  kUnknown = 0x00,
  kIntegerSigned = 0x01,
  kIntegerUnsigned = 0x02,
  // TODO(benvanik): specialize with semantics from APFloat.
  kFloatIEEE = 0x03,
};
constexpr inline int32_t makeElementTypeValue(NumericalType numericalType,
                                              int32_t bitCount) {
  return (static_cast<uint32_t>(numericalType) << 24) | bitCount;
}
}  // namespace
llvm::Optional<int32_t> getElementTypeValue(Type type) {
  if (auto intType = type.dyn_cast_or_null<IntegerType>()) {
    // TODO(benvanik): add signed/unsigned check when landed in MLIR.
    return makeElementTypeValue(NumericalType::kIntegerSigned,
                                intType.getWidth());
  } else if (auto floatType = type.dyn_cast_or_null<FloatType>()) {
    switch (APFloat::SemanticsToEnum(floatType.getFloatSemantics())) {
      case APFloat::S_IEEEhalf:
      case APFloat::S_IEEEsingle:
      case APFloat::S_IEEEdouble:
      case APFloat::S_IEEEquad:
        return makeElementTypeValue(NumericalType::kFloatIEEE,
                                    floatType.getWidth());
      default:
        return llvm::None;
    }
  }
  return llvm::None;
}
// LINT.ThenChange(https://github.com/google/iree/tree/main/iree/hal/api.h:element_type)

IntegerAttr getElementTypeAttr(Type type) {
  auto elementType = getElementTypeValue(type);
  if (!elementType) return {};
  return IntegerAttr::get(IntegerType::get(32, type.getContext()),
                          elementType.getValue());
}

//===----------------------------------------------------------------------===//
// Attribute printing and parsing
//===----------------------------------------------------------------------===//

// static
Attribute ByteRangeAttr::parse(DialectAsmParser &p) {
  auto b = p.getBuilder();
  if (failed(p.parseLess())) return {};

  // TODO(benvanik): support the range syntax; the dialect asm parser fights
  // with it though by checking for proper []/() nesting.

  // Try first the range style: byte_range<[start..end)>
  bool startInclusive;
  if (succeeded(p.parseOptionalLSquare())) {  // [...
    startInclusive = true;
  } else if (succeeded(p.parseOptionalLParen())) {  // (...
    startInclusive = false;
  } else {
    // byte_range<offset, length>
    IntegerAttr offsetAttr;
    IntegerAttr lengthAttr;
    if (failed(p.parseAttribute(offsetAttr, b.getIndexType())) ||
        failed(p.parseComma()) ||
        failed(p.parseAttribute(lengthAttr, b.getIndexType())) ||
        failed(p.parseGreater())) {
      return {};
    }
    return get(offsetAttr, lengthAttr);
  }

  IntegerAttr startAttr;
  IntegerAttr endAttr;
  if (failed(p.parseAttribute(startAttr, b.getIndexType())) ||
      failed(p.parseKeyword("to")) ||
      failed(p.parseAttribute(endAttr, b.getIndexType()))) {
    return {};
  }

  bool endInclusive;
  if (succeeded(p.parseOptionalRSquare())) {  // ...]
    endInclusive = true;
  } else if (succeeded(p.parseOptionalRParen())) {  // ...)
    endInclusive = false;
  } else {
    p.emitError(p.getCurrentLocation()) << "expected ] or ) to end range";
    return {};
  }

  if (failed(p.parseGreater())) return {};

  startAttr = startInclusive
                  ? startAttr
                  : b.getIndexAttr((startAttr.getValue() + 1).getSExtValue());
  endAttr = endInclusive
                ? endAttr
                : b.getIndexAttr((endAttr.getValue() - 1).getSExtValue());

  IntegerAttr offsetAttr = startAttr;
  IntegerAttr lengthAttr = b.getIndexAttr(
      (endAttr.getValue() - startAttr.getValue()).getSExtValue());
  return get(offsetAttr, lengthAttr);
}

void ByteRangeAttr::print(DialectAsmPrinter &p) const {
  auto &os = p.getStream();
  os << getKindName() << "<";
  os << offset();
  os << ", ";
  os << length();
  os << ">";
}

// static
Attribute DescriptorSetLayoutBindingAttr::parse(DialectAsmParser &p) {
  auto b = p.getBuilder();
  IntegerAttr bindingAttr;
  IntegerAttr typeAttr;
  IntegerAttr accessAttr;
  if (failed(p.parseLess()) ||
      failed(p.parseAttribute(bindingAttr, b.getIntegerType(32))) ||
      failed(p.parseComma()) ||
      failed(parseEnumAttr<DescriptorType>(p, "type", typeAttr)) ||
      failed(p.parseComma()) ||
      failed(parseEnumAttr<MemoryAccessBitfield>(p, "access", accessAttr)) ||
      failed(p.parseGreater())) {
    return {};
  }
  return get(bindingAttr, typeAttr, accessAttr);
}

void DescriptorSetLayoutBindingAttr::print(DialectAsmPrinter &p) const {
  auto &os = p.getStream();
  os << getKindName() << "<";
  os << binding() << ", ";
  os << "\"" << stringifyDescriptorType(type()) << "\", ";
  os << "\"" << stringifyMemoryAccessBitfield(access()) << "\"";
  os << ">";
}

// static
Attribute MatchAlwaysAttr::parse(DialectAsmParser &p) {
  return get(p.getBuilder().getContext());
}

void MatchAlwaysAttr::print(DialectAsmPrinter &p) const {
  auto &os = p.getStream();
  os << getKindName();
}

static ArrayAttr parseMultiMatchAttrArray(DialectAsmParser &p) {
  auto b = p.getBuilder();
  SmallVector<Attribute, 4> conditionAttrs;
  if (failed(p.parseLess()) || failed(p.parseLSquare())) {
    return {};
  }
  do {
    Attribute conditionAttr;
    if (failed(p.parseAttribute(conditionAttr))) {
      return {};
    }
    conditionAttrs.push_back(conditionAttr);
  } while (succeeded(p.parseOptionalComma()));
  if (failed(p.parseRSquare()) || failed(p.parseGreater())) {
    return {};
  }
  return b.getArrayAttr(conditionAttrs);
}

static void printMultiMatchAttrList(ArrayAttr conditionAttrs,
                                    DialectAsmPrinter &p) {
  auto &os = p.getStream();
  os << "<[";
  interleaveComma(conditionAttrs, os,
                  [&](Attribute condition) { os << condition; });
  os << "]>";
}

// static
Attribute MatchAnyAttr::parse(DialectAsmParser &p) {
  return get(parseMultiMatchAttrArray(p));
}

void MatchAnyAttr::print(DialectAsmPrinter &p) const {
  p << getKindName();
  printMultiMatchAttrList(conditions().cast<ArrayAttr>(), p);
}

// static
Attribute MatchAllAttr::parse(DialectAsmParser &p) {
  return get(parseMultiMatchAttrArray(p));
}

void MatchAllAttr::print(DialectAsmPrinter &p) const {
  p << getKindName();
  printMultiMatchAttrList(conditions().cast<ArrayAttr>(), p);
}

// static
Attribute DeviceMatchIDAttr::parse(DialectAsmParser &p) {
  StringAttr patternAttr;
  if (failed(p.parseLess()) || failed(p.parseAttribute(patternAttr)) ||
      failed(p.parseGreater())) {
    return {};
  }
  return get(patternAttr);
}

void DeviceMatchIDAttr::print(DialectAsmPrinter &p) const {
  auto &os = p.getStream();
  os << getKindName() << "<\"";
  os << pattern();
  os << "\">";
}

#include "iree/compiler/Dialect/HAL/IR/HALOpInterface.cpp.inc"

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
