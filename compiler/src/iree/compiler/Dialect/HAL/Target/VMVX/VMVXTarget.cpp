// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/VMVX/VMVXTarget.h"

#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/VM/IR/VMDialect.h"
#include "iree/compiler/Dialect/VM/Target/Bytecode/BytecodeModuleTarget.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "iree/compiler/Dialect/VMVX/IR/VMVXDialect.h"
#include "iree/compiler/Dialect/VMVX/Transforms/Passes.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

static llvm::cl::opt<std::string> clVMVXTuneCPU(
    "iree-vmvx-tune-cpu",
    llvm::cl::desc(
        "Hint to make IR more likely to perform well on the given target CPU "
        "architectures+features. Comma-separated list. Each comma-delimited "
        "entry is of the form `arch[+feature1[+feature2...]]` where "
        "`arch` is a valid arch as in the first component of a llvm target "
        "triple, and `feature1`, `feature2`, ... are as in "
        "`--iree-llvm-target-cpu-features`. "
        "Example: \"--iree-vmvx-tune-cpu=aarch64,aarch64+dotprod\" means "
        "generate two variants, one for vanilla AArch64 and one for AArch64 "
        "with the ARMv8.2 optional feature `dotprod`. "),
    llvm::cl::init(""));

static IREE::HAL::ExecutableTargetAttr getVMVXExecutableTarget(
    MLIRContext *context, StringRef backend, StringRef format,
    StringRef tuneCPU) {
  SmallVector<NamedAttribute> config;
  auto addConfig = [context](SmallVector<NamedAttribute> &config,
                             StringRef name, StringRef value) {
    config.emplace_back(StringAttr::get(context, name),
                        StringAttr::get(context, value));
  };
  if (!tuneCPU.empty()) {
    auto initialSplit = tuneCPU.split('+');
    addConfig(config, "target_triple", initialSplit.first);
    StringRef cpu_features_to_parse = initialSplit.second;
    std::string cpu_features_out;
    while (!cpu_features_to_parse.empty()) {
      if (!cpu_features_out.empty()) cpu_features_out += ',';
      cpu_features_out += '+';
      auto split = cpu_features_to_parse.split('+');
      cpu_features_out += split.first;
      cpu_features_to_parse = split.second;
    }
    addConfig(config, "cpu_features", cpu_features_out);
  }

  return IREE::HAL::ExecutableTargetAttr::get(
      context, StringAttr::get(context, backend),
      StringAttr::get(context, format), DictionaryAttr::get(context, config));
}

static ArrayAttr getVMVXExecutableTargets(MLIRContext *context,
                                          StringRef backend, StringRef format) {
  SmallVector<Attribute> targetAttrs;
  StringRef commaSeparatedTuneCPU = clVMVXTuneCPU;
  do {
    auto split = commaSeparatedTuneCPU.split(',');
    targetAttrs.push_back(
        getVMVXExecutableTarget(context, backend, format, split.first));
    commaSeparatedTuneCPU = split.second;
  } while (!commaSeparatedTuneCPU.empty());
  return ArrayAttr::get(context, targetAttrs);
}

class VMVXTargetBackend final : public TargetBackend {
 public:
  VMVXTargetBackend() = default;

  std::string name() const override { return "vmvx"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Codegen::IREECodegenDialect, IREE::VM::VMDialect,
                    IREE::VMVX::VMVXDialect>();
  }

  IREE::HAL::DeviceTargetAttr getDefaultDeviceTarget(
      MLIRContext *context) const override {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;

    configItems.emplace_back(
        b.getStringAttr("executable_targets"),
        getVMVXExecutableTargets(context, "vmvx", "vmvx-bytecode-fb"));

    auto configAttr = b.getDictionaryAttr(configItems);
    return IREE::HAL::DeviceTargetAttr::get(
        context, b.getStringAttr(deviceID()), configAttr);
  }

  void buildTranslationPassPipeline(OpPassManager &passManager) override {
    IREE::VMVX::buildVMVXTransformPassPipeline(passManager);

    OpPassManager &nestedModulePM = passManager.nest<ModuleOp>();

    // TODO(benvanik): derive these from a vm target triple.
    auto vmOptions = IREE::VM::TargetOptions::FromFlags::get();
    vmOptions.f32Extension = true;
    vmOptions.optimizeForStackSize = false;
    IREE::VM::buildVMTransformPassPipeline(nestedModulePM, vmOptions);
  }

  void buildLinkingPassPipeline(OpPassManager &passManager) override {
    buildVMVXLinkingPassPipeline(passManager);
  }

  LogicalResult serializeExecutable(const SerializationOptions &options,
                                    IREE::HAL::ExecutableVariantOp variantOp,
                                    OpBuilder &executableBuilder) override {
    // Add reflection information used at runtime specific to the HAL interface.
    SymbolTable symbolTable(variantOp.getInnerModule());
    for (auto exportOp : variantOp.getBlock().getOps<ExecutableExportOp>()) {
      auto funcOp = symbolTable.lookup<IREE::VM::FuncOp>(exportOp.getName());

      // Optionally entry points may specify that they require workgroup local
      // memory. We fetch that value here and plumb it through so the runtime
      // knows how much memory to reserve and pass in.
      auto localMemorySizeAttr = exportOp.getWorkgroupLocalMemoryAttr();
      if (localMemorySizeAttr) {
        funcOp.setReflectionAttr("local_memory", localMemorySizeAttr);
      }
    }

    // Serialize the VM module to bytes and embed it directly.
    SmallVector<char> moduleData;
    {
      IREE::VM::BytecodeTargetOptions bytecodeOptions;
      llvm::raw_svector_ostream stream(moduleData);
      if (failed(translateModuleToBytecode(variantOp.getInnerModule(),
                                           bytecodeOptions, stream))) {
        return variantOp.emitOpError()
               << "failed to serialize VM bytecode module";
      }
    }
    if (!options.dumpBinariesPath.empty()) {
      dumpDataToPath<char>(options.dumpBinariesPath, options.dumpBaseName,
                           variantOp.getName(), ".vmfb", moduleData);
    }

    auto bufferAttr = DenseIntElementsAttr::get(
        VectorType::get({static_cast<int64_t>(moduleData.size())},
                        IntegerType::get(executableBuilder.getContext(), 8)),
        std::move(moduleData));

    // Add the binary data to the target executable.
    // NOTE: this snapshots the FlatBuffer builder data at the time it is called
    // and future changes to the target op will not be observed.
    auto binaryOp = executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
        variantOp.getLoc(), variantOp.getSymName(),
        variantOp.getTarget().getFormat(), bufferAttr);
    binaryOp.setMimeTypeAttr(
        executableBuilder.getStringAttr("application/x-flatbuffers"));

    return success();
  }
};

class VMVXInlineTargetBackend final : public TargetBackend {
 public:
  VMVXInlineTargetBackend() = default;

  std::string name() const override { return "vmvx-inline"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<IREE::Codegen::IREECodegenDialect, IREE::VMVX::VMVXDialect>();
  }

  IREE::HAL::DeviceTargetAttr getDefaultDeviceTarget(
      MLIRContext *context) const override {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;

    configItems.emplace_back(
        b.getStringAttr("executable_targets"),
        getVMVXExecutableTargets(context, "vmvx-inline", "vmvx-ir"));

    auto configAttr = b.getDictionaryAttr(configItems);
    return IREE::HAL::DeviceTargetAttr::get(
        context, b.getStringAttr(deviceID()), configAttr);
  }

  void buildTranslationPassPipeline(OpPassManager &passManager) override {
    IREE::VMVX::buildVMVXTransformPassPipeline(passManager);
  }
};

void registerVMVXTargetBackends() {
  // #hal.device.target<"vmvx", ...
  // #hal.executable.target<"vmvx", ...
  static TargetBackendRegistration registration0(
      "vmvx", [=]() { return std::make_shared<VMVXTargetBackend>(); });
  static TargetBackendRegistration registration1("vmvx-inline", [=]() {
    return std::make_shared<VMVXInlineTargetBackend>();
  });
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
