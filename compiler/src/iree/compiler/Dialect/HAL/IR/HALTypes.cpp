// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Utils/StringUtils.h"
#include "llvm/ADT/StringExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

// clang-format off: must be included after all LLVM/MLIR headers.
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Dialect/HAL/IR/HALAttrs.cpp.inc"  // IWYU pragma: keep
#include "iree/compiler/Dialect/HAL/IR/HALEnums.cpp.inc"  // IWYU pragma: keep
// clang-format on

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

//===----------------------------------------------------------------------===//
// Enum utilities
//===----------------------------------------------------------------------===//

template <typename AttrType>
static LogicalResult parseEnumAttr(AsmParser &parser, StringRef attrName,
                                   AttrType &attr) {
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
  auto symbolized =
      symbolizeEnum<typename AttrType::ValueType>(stringAttr.getValue());
  if (!symbolized.hasValue()) {
    return parser.emitError(loc)
           << "failed to parse '" << attrName << "' enum value";
  }
  attr = AttrType::get(parser.getBuilder().getContext(), symbolized.getValue());
  return success();
}

template <typename AttrType>
static LogicalResult parseOptionalEnumAttr(AsmParser &parser,
                                           StringRef attrName, AttrType &attr) {
  if (succeeded(parser.parseOptionalQuestion())) {
    // Special case `?` to indicate any/none/undefined/etc.
    attr = AttrType::get(parser.getBuilder().getContext(), 0);
    return success();
  }
  return parseEnumAttr<AttrType>(parser, attrName, attr);
}

//===----------------------------------------------------------------------===//
// Element types
//===----------------------------------------------------------------------===//

// Keep these in sync with iree/hal/api.h
namespace {
enum class NumericalType : uint32_t {
  kUnknown = 0x00,
  kInteger = 0x10,
  kIntegerSigned = kInteger | 0x01,
  kIntegerUnsigned = kInteger | 0x02,
  kBoolean = kInteger | 0x03,
  kFloat = 0x20,
  kFloatIEEE = kFloat | 0x01,
  kFloatBrain = kFloat | 0x02,
  kFloatComplex = kFloat | 0x03,
};
constexpr inline int32_t makeElementTypeValue(NumericalType numericalType,
                                              int32_t bitCount) {
  return (static_cast<uint32_t>(numericalType) << 24) | bitCount;
}
}  // namespace

llvm::Optional<int32_t> getElementTypeValue(Type type) {
  if (auto intType = type.dyn_cast_or_null<IntegerType>()) {
    NumericalType numericalType;
    if (intType.isInteger(1)) {
      return makeElementTypeValue(NumericalType::kBoolean, 8);
    } else if (intType.isSigned()) {
      numericalType = NumericalType::kIntegerSigned;
    } else if (intType.isUnsigned()) {
      numericalType = NumericalType::kIntegerUnsigned;
    } else {
      // There's no such thing as a signless integer in machine types but we
      // need to be able to round-trip the format through the ABI. Exact
      // numerical type equality comparisons may fail if the frontend assumes
      // signed/unsigned but the compiler is propagating signless.
      numericalType = NumericalType::kInteger;
    }
    return makeElementTypeValue(numericalType, intType.getWidth());
  } else if (auto floatType = type.dyn_cast_or_null<FloatType>()) {
    switch (APFloat::SemanticsToEnum(floatType.getFloatSemantics())) {
      case APFloat::S_IEEEhalf:
      case APFloat::S_IEEEsingle:
      case APFloat::S_IEEEdouble:
      case APFloat::S_IEEEquad:
        return makeElementTypeValue(NumericalType::kFloatIEEE,
                                    floatType.getWidth());
      case APFloat::S_BFloat:
        return makeElementTypeValue(NumericalType::kFloatBrain,
                                    floatType.getWidth());
      default:
        return llvm::None;
    }
  } else if (auto complexType = type.dyn_cast_or_null<ComplexType>()) {
    return makeElementTypeValue(
        NumericalType::kFloatComplex,
        complexType.getElementType().getIntOrFloatBitWidth() * 2);
  }
  return llvm::None;
}

llvm::Optional<int32_t> getEncodingTypeValue(Attribute attr) {
  // TODO(#6762): encoding attribute handling/mapping to enums.
  assert(!attr && "encoding types other than default not yet supported");
  // Default to IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR for now.
  return 1;
}

//===----------------------------------------------------------------------===//
// Object types
//===----------------------------------------------------------------------===//

Value BufferType::inferSizeFromValue(Location loc, Value value,
                                     OpBuilder &builder) const {
  return builder.createOrFold<BufferLengthOp>(loc, builder.getIndexType(),
                                              value);
}

Value BufferViewType::inferSizeFromValue(Location loc, Value value,
                                         OpBuilder &builder) const {
  return builder.createOrFold<BufferLengthOp>(
      loc, builder.getIndexType(),
      builder.createOrFold<BufferViewBufferOp>(
          loc, builder.getType<IREE::HAL::BufferType>(), value));
}

//===----------------------------------------------------------------------===//
// #hal.device.target
//===----------------------------------------------------------------------===//

// static
DeviceTargetAttr DeviceTargetAttr::get(MLIRContext *context,
                                       StringRef deviceID) {
  // TODO(benvanik): query default configuration from the target backend.
  return get(context, StringAttr::get(context, deviceID),
             DictionaryAttr::get(context));
}

// static
Attribute DeviceTargetAttr::parse(AsmParser &p, Type type) {
  StringAttr deviceIDAttr;
  DictionaryAttr configAttr;
  // `<"device-id"`
  if (failed(p.parseLess()) || failed(p.parseAttribute(deviceIDAttr))) {
    return {};
  }
  // `, {config}`
  if (succeeded(p.parseOptionalComma()) &&
      failed(p.parseAttribute(configAttr))) {
    return {};
  }
  // `>`
  if (failed(p.parseGreater())) {
    return {};
  }
  return get(p.getContext(), deviceIDAttr, configAttr);
}

void DeviceTargetAttr::print(AsmPrinter &p) const {
  auto &os = p.getStream();
  os << "<";
  p.printAttribute(getDeviceID());
  auto configAttr = getConfiguration();
  if (configAttr && !configAttr.empty()) {
    os << ", ";
    p.printAttribute(configAttr);
  }
  os << ">";
}

std::string DeviceTargetAttr::getSymbolNameFragment() {
  return sanitizeSymbolName(getDeviceID().getValue().lower());
}

Attribute DeviceTargetAttr::getMatchExpression() {
  return DeviceMatchIDAttr::get(*this);
}

bool DeviceTargetAttr::hasConfigurationAttr(StringRef name) {
  auto configAttr = getConfiguration();
  return configAttr && configAttr.get(name);
}

SmallVector<ExecutableTargetAttr, 4> DeviceTargetAttr::getExecutableTargets() {
  SmallVector<ExecutableTargetAttr, 4> resultAttrs;
  auto configAttr = getConfiguration();
  if (configAttr) {
    auto targetsAttr = configAttr.getAs<ArrayAttr>("executable_targets");
    if (targetsAttr) {
      for (auto attr : targetsAttr.getValue()) {
        resultAttrs.push_back(attr.dyn_cast<ExecutableTargetAttr>());
      }
    }
  }
  return resultAttrs;
}

// static
SmallVector<IREE::HAL::DeviceTargetAttr, 4> DeviceTargetAttr::lookup(
    Operation *op) {
  auto attrId = mlir::StringAttr::get(op->getContext(), "hal.device.targets");
  while (op) {
    auto targetsAttr = op->getAttrOfType<ArrayAttr>(attrId);
    if (targetsAttr) {
      SmallVector<IREE::HAL::DeviceTargetAttr, 4> result;
      for (auto targetAttr : targetsAttr) {
        result.push_back(targetAttr.cast<IREE::HAL::DeviceTargetAttr>());
      }
      return result;
    }
    op = op->getParentOp();
  }
  return {};  // No devices found; let caller decide what to do.
}

// static
SmallVector<ExecutableTargetAttr, 4> DeviceTargetAttr::lookupExecutableTargets(
    Operation *op) {
  SmallVector<ExecutableTargetAttr, 4> resultAttrs;
  for (auto deviceTargetAttr : lookup(op)) {
    for (auto executableTargetAttr : deviceTargetAttr.getExecutableTargets()) {
      if (!llvm::is_contained(resultAttrs, executableTargetAttr)) {
        resultAttrs.push_back(executableTargetAttr);
      }
    }
  }
  return resultAttrs;
}

//===----------------------------------------------------------------------===//
// #hal.executable.target
//===----------------------------------------------------------------------===//

// static
ExecutableTargetAttr ExecutableTargetAttr::get(MLIRContext *context,
                                               StringRef backend,
                                               StringRef format) {
  return get(context, StringAttr::get(context, backend),
             StringAttr::get(context, format), DictionaryAttr::get(context));
}

// static
Attribute ExecutableTargetAttr::parse(AsmParser &p, Type type) {
  StringAttr backendAttr;
  StringAttr formatAttr;
  DictionaryAttr configurationAttr;
  // `<"backend", "format"`
  if (failed(p.parseLess()) || failed(p.parseAttribute(backendAttr)) ||
      failed(p.parseComma()) || failed(p.parseAttribute(formatAttr))) {
    return {};
  }
  // `, {config}`
  if (succeeded(p.parseOptionalComma()) &&
      failed(p.parseAttribute(configurationAttr))) {
    return {};
  }
  // `>`
  if (failed(p.parseGreater())) {
    return {};
  }
  return get(p.getContext(), backendAttr, formatAttr, configurationAttr);
}

void ExecutableTargetAttr::print(AsmPrinter &p) const {
  auto &os = p.getStream();
  os << "<";
  p.printAttribute(getBackend());
  os << ", ";
  p.printAttribute(getFormat());
  auto config = getConfiguration();
  if (config && !config.empty()) {
    os << ", ";
    p.printAttribute(config);
  }
  os << ">";
}

std::string ExecutableTargetAttr::getSymbolNameFragment() {
  std::string name = getFormat().getValue().lower();
  if (auto config = getConfiguration()) {
    if (auto targetTripleAttr = config.getAs<StringAttr>("target_triple")) {
      StringRef arch = targetTripleAttr.getValue().split('-').first;
      if (!(arch.empty() || arch == "unknown" ||
            StringRef(name).endswith(arch))) {
        name += '_';
        name += arch.lower();
      }
    }
    if (auto cpuFeaturesAttr = config.getAs<StringAttr>("cpu_feature")) {
      StringRef cpuFeatures = cpuFeaturesAttr.getValue();
      if (!cpuFeatures.empty()) {
        name += '_';
        name += cpuFeatures.lower();
      }
    }
  }
  return sanitizeSymbolName(name, /*coalesce_delims=*/true);
}

Attribute ExecutableTargetAttr::getMatchExpression() {
  return DeviceMatchExecutableFormatAttr::get(getContext(), getFormat());
}

// static
ExecutableTargetAttr ExecutableTargetAttr::lookup(Operation *op) {
  auto *context = op->getContext();
  auto attrId = StringAttr::get(context, "hal.executable.target");
  while (op) {
    // Take directly from the enclosing variant.
    if (auto variantOp = llvm::dyn_cast<ExecutableVariantOp>(op)) {
      return variantOp.getTarget();
    }
    // Use an override if specified.
    auto attr = op->getAttrOfType<ExecutableTargetAttr>(attrId);
    if (attr) return attr;
    // Continue walk.
    op = op->getParentOp();
  }
  // No target found during walk. No default to provide so fail and let the
  // caller decide what to do (assert/fallback/etc).
  return nullptr;
}

//===----------------------------------------------------------------------===//
// #hal.affinity.queue
//===----------------------------------------------------------------------===//

// static
Attribute AffinityQueueAttr::parse(AsmParser &p, Type type) {
  int64_t mask = 0;
  // `<`
  if (failed(p.parseLess())) return {};
  // `*` (any)
  if (succeeded(p.parseOptionalStar())) {
    mask = -1;
  } else {
    // `[`queue_bit[, ...] `]`
    if (failed(p.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&]() {
          int64_t i = 0;
          if (failed(p.parseInteger(i))) return failure();
          mask |= 1ll << i;
          return success();
        }))) {
      return {};
    }
  }
  // `>`
  if (failed(p.parseGreater())) return {};
  return get(p.getContext(), mask);
}

void AffinityQueueAttr::print(AsmPrinter &p) const {
  auto &os = p.getStream();
  os << "<";
  int64_t mask = getMask();
  if (mask == -1) {
    os << "*";
  } else {
    os << "[";
    for (int i = 0, j = 0; i < sizeof(mask) * 8; ++i) {
      if (mask & (1ll << i)) {
        if (j++ > 0) os << ", ";
        os << i;
      }
    }
    os << "]";
  }
  os << ">";
}

bool AffinityQueueAttr::isExecutableWith(
    IREE::Stream::AffinityAttr other) const {
  if (!other) return true;
  // Only compatible with other queue affinities today. When we extend the
  // attributes to specify device targets we'd want to check here.
  auto otherQueueAttr = other.dyn_cast_or_null<AffinityQueueAttr>();
  if (!otherQueueAttr) return false;
  // If this affinity is a subset of the target affinity then it can execute
  // with it.
  if ((getMask() & otherQueueAttr.getMask()) == getMask()) return true;
  // Otherwise not compatible.
  return false;
}

IREE::Stream::AffinityAttr AffinityQueueAttr::joinOR(
    IREE::Stream::AffinityAttr other) const {
  if (!other) return *this;
  if (!IREE::Stream::AffinityAttr::canExecuteTogether(*this, other)) {
    return nullptr;
  }
  auto otherQueueAttr = other.dyn_cast_or_null<AffinityQueueAttr>();
  return AffinityQueueAttr::get(getContext(),
                                getMask() | otherQueueAttr.getMask());
}

IREE::Stream::AffinityAttr AffinityQueueAttr::joinAND(
    IREE::Stream::AffinityAttr other) const {
  if (!other) return *this;
  if (!IREE::Stream::AffinityAttr::canExecuteTogether(*this, other)) {
    return nullptr;
  }
  auto otherQueueAttr = other.dyn_cast_or_null<AffinityQueueAttr>();
  return AffinityQueueAttr::get(getContext(),
                                getMask() & otherQueueAttr.getMask());
}

//===----------------------------------------------------------------------===//
// #hal.match.*
//===----------------------------------------------------------------------===//

Value MatchAlwaysAttr::buildConditionExpression(Location loc, Value value,
                                                OpBuilder builder) const {
  // #hal.match.always -> true
  return builder.createOrFold<arith::ConstantIntOp>(loc, /*value=*/1,
                                                    /*width=*/1);
}

static ArrayAttr parseMultiMatchAttrArray(AsmParser &p) {
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

static void printMultiMatchAttrList(ArrayAttr conditionAttrs, AsmPrinter &p) {
  auto &os = p.getStream();
  os << "<[";
  interleaveComma(conditionAttrs, os,
                  [&](Attribute condition) { os << condition; });
  os << "]>";
}

// static
Attribute MatchAnyAttr::parse(AsmParser &p, Type type) {
  return get(p.getContext(), parseMultiMatchAttrArray(p));
}

void MatchAnyAttr::print(AsmPrinter &p) const {
  printMultiMatchAttrList(getConditions(), p);
}

Value MatchAnyAttr::buildConditionExpression(Location loc, Value value,
                                             OpBuilder builder) const {
  // #hal.match.any<[a, b, c]> -> or(or(a, b), c)
  if (getConditions().empty()) {
    // Empty returns false (no conditions match).
    return builder.create<arith::ConstantIntOp>(loc, /*value=*/0, /*width=*/1);
  }
  auto conditionValues =
      llvm::map_range(getConditions(), [&](MatchAttrInterface attr) {
        return attr.buildConditionExpression(loc, value, builder);
      });
  Value resultValue;
  for (auto conditionValue : conditionValues) {
    resultValue = resultValue ? builder.createOrFold<arith::OrIOp>(
                                    loc, resultValue, conditionValue)
                              : conditionValue;
  }
  return resultValue;
}

// static
Attribute MatchAllAttr::parse(AsmParser &p, Type type) {
  return get(p.getContext(), parseMultiMatchAttrArray(p));
}

void MatchAllAttr::print(AsmPrinter &p) const {
  printMultiMatchAttrList(getConditions(), p);
}

Value MatchAllAttr::buildConditionExpression(Location loc, Value value,
                                             OpBuilder builder) const {
  // #hal.match.all<[a, b, c]> -> and(and(a, b), c)
  if (getConditions().empty()) {
    // Empty returns true (all 0 conditions match).
    return builder.create<arith::ConstantIntOp>(loc, /*value=*/1, /*width=*/1);
  }
  auto conditionValues =
      llvm::map_range(getConditions(), [&](MatchAttrInterface attr) {
        return attr.buildConditionExpression(loc, value, builder);
      });
  Value resultValue;
  for (auto conditionValue : conditionValues) {
    resultValue = resultValue ? builder.createOrFold<arith::AndIOp>(
                                    loc, resultValue, conditionValue)
                              : conditionValue;
  }
  return resultValue;
}

// static
Attribute DeviceMatchIDAttr::parse(AsmParser &p, Type type) {
  StringAttr patternAttr;
  if (failed(p.parseLess()) || failed(p.parseAttribute(patternAttr)) ||
      failed(p.parseGreater())) {
    return {};
  }
  return get(p.getContext(), patternAttr);
}

void DeviceMatchIDAttr::print(AsmPrinter &p) const {
  auto &os = p.getStream();
  os << "<";
  p.printAttribute(getPattern());
  os << ">";
}

Value DeviceMatchIDAttr::buildConditionExpression(Location loc, Value device,
                                                  OpBuilder builder) const {
  auto i1Type = builder.getI1Type();
  return builder
      .create<IREE::HAL::DeviceQueryOp>(
          loc, i1Type, i1Type, device, builder.getStringAttr("hal.device.id"),
          getPattern(), builder.getZeroAttr(i1Type))
      .getValue();
}

// static
Attribute DeviceMatchFeatureAttr::parse(AsmParser &p, Type type) {
  StringAttr patternAttr;
  if (failed(p.parseLess()) || failed(p.parseAttribute(patternAttr)) ||
      failed(p.parseGreater())) {
    return {};
  }
  return get(p.getContext(), patternAttr);
}

void DeviceMatchFeatureAttr::print(AsmPrinter &p) const {
  auto &os = p.getStream();
  os << "<";
  p.printAttribute(getPattern());
  os << ">";
}

Value DeviceMatchFeatureAttr::buildConditionExpression(
    Location loc, Value device, OpBuilder builder) const {
  auto i1Type = builder.getI1Type();
  return builder
      .create<IREE::HAL::DeviceQueryOp>(
          loc, i1Type, i1Type, device,
          builder.getStringAttr("hal.device.feature"), getPattern(),
          builder.getZeroAttr(i1Type))
      .getValue();
}

// static
Attribute DeviceMatchArchitectureAttr::parse(AsmParser &p, Type type) {
  StringAttr patternAttr;
  if (failed(p.parseLess()) || failed(p.parseAttribute(patternAttr)) ||
      failed(p.parseGreater())) {
    return {};
  }
  return get(p.getContext(), patternAttr);
}

void DeviceMatchArchitectureAttr::print(AsmPrinter &p) const {
  auto &os = p.getStream();
  os << "<";
  p.printAttribute(getPattern());
  os << ">";
}

Value DeviceMatchArchitectureAttr::buildConditionExpression(
    Location loc, Value device, OpBuilder builder) const {
  auto i1Type = builder.getI1Type();
  return builder
      .create<IREE::HAL::DeviceQueryOp>(
          loc, i1Type, i1Type, device,
          builder.getStringAttr("hal.device.architecture"), getPattern(),
          builder.getZeroAttr(i1Type))
      .getValue();
}

// static
Attribute DeviceMatchExecutableFormatAttr::parse(AsmParser &p, Type type) {
  StringAttr patternAttr;
  if (failed(p.parseLess()) || failed(p.parseAttribute(patternAttr)) ||
      failed(p.parseGreater())) {
    return {};
  }
  return get(p.getContext(), patternAttr);
}

void DeviceMatchExecutableFormatAttr::print(AsmPrinter &p) const {
  auto &os = p.getStream();
  os << "<";
  p.printAttribute(getPattern());
  os << ">";
}

Value DeviceMatchExecutableFormatAttr::buildConditionExpression(
    Location loc, Value device, OpBuilder builder) const {
  auto i1Type = builder.getI1Type();
  return builder
      .create<IREE::HAL::DeviceQueryOp>(
          loc, i1Type, i1Type, device,
          builder.getStringAttr("hal.executable.format"), getPattern(),
          builder.getZeroAttr(i1Type))
      .getValue();
}

//===----------------------------------------------------------------------===//
// Dialect registration
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/HAL/IR/HALAttrInterfaces.cpp.inc"
#include "iree/compiler/Dialect/HAL/IR/HALOpInterfaces.cpp.inc"
#include "iree/compiler/Dialect/HAL/IR/HALTypeInterfaces.cpp.inc"

void HALDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree/compiler/Dialect/HAL/IR/HALAttrs.cpp.inc"  // IWYU pragma: keep
      >();
}

void HALDialect::registerTypes() {
  addTypes<AllocatorType, BufferType, BufferViewType, CommandBufferType,
           DescriptorSetLayoutType, DeviceType, EventType, ExecutableType,
           PipelineLayoutType, FenceType, RingBufferType, SemaphoreType>();
}

//===----------------------------------------------------------------------===//
// Attribute printing and parsing
//===----------------------------------------------------------------------===//

Attribute HALDialect::parseAttribute(DialectAsmParser &parser,
                                     Type type) const {
  StringRef mnemonic;
  Attribute genAttr;
  OptionalParseResult parseResult =
      generatedAttributeParser(parser, &mnemonic, type, genAttr);
  if (parseResult.has_value()) return genAttr;
  parser.emitError(parser.getNameLoc())
      << "unknown HAL attribute: " << mnemonic;
  return {};
}

void HALDialect::printAttribute(Attribute attr, DialectAsmPrinter &p) const {
  TypeSwitch<Attribute>(attr).Default([&](Attribute) {
    if (failed(generatedAttributePrinter(attr, p))) {
      assert(false && "unhandled HAL attribute kind");
    }
  });
}

//===----------------------------------------------------------------------===//
// Type printing and parsing
//===----------------------------------------------------------------------===//

Type HALDialect::parseType(DialectAsmParser &parser) const {
  StringRef typeKind;
  if (parser.parseKeyword(&typeKind)) return {};
  auto type =
      llvm::StringSwitch<Type>(typeKind)
          .Case("allocator", AllocatorType::get(getContext()))
          .Case("buffer", BufferType::get(getContext()))
          .Case("buffer_view", BufferViewType::get(getContext()))
          .Case("command_buffer", CommandBufferType::get(getContext()))
          .Case("descriptor_set_layout",
                DescriptorSetLayoutType::get(getContext()))
          .Case("device", DeviceType::get(getContext()))
          .Case("event", EventType::get(getContext()))
          .Case("executable", ExecutableType::get(getContext()))
          .Case("pipeline_layout", PipelineLayoutType::get(getContext()))
          .Case("fence", FenceType::get(getContext()))
          .Case("ring_buffer", RingBufferType::get(getContext()))
          .Case("semaphore", SemaphoreType::get(getContext()))
          .Default(nullptr);
  if (!type) {
    parser.emitError(parser.getCurrentLocation())
        << "unknown HAL type: " << typeKind;
  }
  return type;
}

void HALDialect::printType(Type type, DialectAsmPrinter &p) const {
  if (type.isa<AllocatorType>()) {
    p << "allocator";
  } else if (type.isa<BufferType>()) {
    p << "buffer";
  } else if (type.isa<BufferViewType>()) {
    p << "buffer_view";
  } else if (type.isa<CommandBufferType>()) {
    p << "command_buffer";
  } else if (type.isa<DescriptorSetLayoutType>()) {
    p << "descriptor_set_layout";
  } else if (type.isa<DeviceType>()) {
    p << "device";
  } else if (type.isa<EventType>()) {
    p << "event";
  } else if (type.isa<ExecutableType>()) {
    p << "executable";
  } else if (type.isa<PipelineLayoutType>()) {
    p << "pipeline_layout";
  } else if (type.isa<FenceType>()) {
    p << "fence";
  } else if (type.isa<RingBufferType>()) {
    p << "ring_buffer";
  } else if (type.isa<SemaphoreType>()) {
    p << "semaphore";
  } else {
    assert(false && "unknown HAL type");
  }
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
