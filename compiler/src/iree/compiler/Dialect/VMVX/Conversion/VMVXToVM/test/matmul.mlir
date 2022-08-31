// RUN: iree-opt --iree-vm-target-index-bits=64 --split-input-file \
// RUN:   --iree-vm-conversion --canonicalize %s | FileCheck %s

// CHECK-LABEL: @matmul_f32f32f32
  // CHECK-SAME: %[[lhs_buffer:[^:]+]]: !vm.buffer, %[[lhs_offset:[^:]+]]: i64,
  // CHECK-SAME: %[[rhs_buffer:[^:]+]]: !vm.buffer, %[[rhs_offset:[^:]+]]: i64,
  // CHECK-SAME: %[[out_buffer:[^:]+]]: !vm.buffer, %[[out_offset:[^:]+]]: i64,
  // CHECK-SAME: %[[lhs_stride:[^:]+]]: i64, %[[rhs_stride:[^:]+]]: i64, %[[out_stride:[^:]+]]: i64,
  // CHECK-SAME: %[[m:[^:]+]]: i64, %[[n:[^:]+]]: i64, %[[k:[^:]+]]: i64) {

  // CHECK-DAG: %[[flags:.+]] = vm.const.i32 1
  //      CHECK: vm.call @vmvx.matmul.f32f32f32(
  // CHECK-SAME: %[[lhs_buffer]], %[[lhs_offset]],
  // CHECK-SAME: %[[rhs_buffer]], %[[rhs_offset]],
  // CHECK-SAME: %[[out_buffer]], %[[out_offset]],
  // CHECK-SAME: %[[lhs_stride]], %[[rhs_stride]], %[[out_stride]],
  // CHECK-SAME: %[[m]], %[[n]], %[[k]],
  // CHECK-SAME: %[[flags]]) : (!vm.buffer, i64, !vm.buffer, i64, !vm.buffer, i64, i64, i64, i64, i64, i64, i64, i32) -> ()


func.func @matmul_f32f32f32(
    // LHS
    %lhs_buffer : !util.buffer, %lhs_offset : index,
    // RHS
    %rhs_buffer : !util.buffer, %rhs_offset : index,
    // OUT
    %out_buffer : !util.buffer, %out_offset : index,
    // STRIDES
    %lhs_stride : index, %rhs_stride : index, %out_stride : index,
    // MNK
    %m : index, %n : index, %k : index
    ) {
  vmvx.matmul lhs(%lhs_buffer offset %lhs_offset row_stride %lhs_stride : !util.buffer)
             rhs(%rhs_buffer offset %rhs_offset row_stride %rhs_stride : !util.buffer)
             out(%out_buffer offset %out_offset row_stride %out_stride : !util.buffer)
             mnk(%m, %n, %k)
             flags(1) : (f32, f32, f32)
  func.return
}

// CHECK-LABEL: @matmul_i8i8i32
  // CHECK-SAME: %[[lhs_buffer:[^:]+]]: !vm.buffer, %[[lhs_offset:[^:]+]]: i64,
  // CHECK-SAME: %[[rhs_buffer:[^:]+]]: !vm.buffer, %[[rhs_offset:[^:]+]]: i64,
  // CHECK-SAME: %[[out_buffer:[^:]+]]: !vm.buffer, %[[out_offset:[^:]+]]: i64,
  // CHECK-SAME: %[[lhs_stride:[^:]+]]: i64, %[[rhs_stride:[^:]+]]: i64, %[[out_stride:[^:]+]]: i64,
  // CHECK-SAME: %[[m:[^:]+]]: i64, %[[n:[^:]+]]: i64, %[[k:[^:]+]]: i64) {

  // CHECK-DAG: %[[flags:.+]] = vm.const.i32 1
  //      CHECK: vm.call @vmvx.matmul.i8i8i32(
  // CHECK-SAME: %[[lhs_buffer]], %[[lhs_offset]],
  // CHECK-SAME: %[[rhs_buffer]], %[[rhs_offset]],
  // CHECK-SAME: %[[out_buffer]], %[[out_offset]],
  // CHECK-SAME: %[[lhs_stride]], %[[rhs_stride]], %[[out_stride]],
  // CHECK-SAME: %[[m]], %[[n]], %[[k]],
  // CHECK-SAME: %[[flags]]) : (!vm.buffer, i64, !vm.buffer, i64, !vm.buffer, i64, i64, i64, i64, i64, i64, i64, i32) -> ()


func.func @matmul_i8i8i32(
    // LHS
    %lhs_buffer : !util.buffer, %lhs_offset : index,
    // RHS
    %rhs_buffer : !util.buffer, %rhs_offset : index,
    // OUT
    %out_buffer : !util.buffer, %out_offset : index,
    // STRIDES
    %lhs_stride : index, %rhs_stride : index, %out_stride : index,
    // MNK
    %m : index, %n : index, %k : index
    ) {
  vmvx.matmul lhs(%lhs_buffer offset %lhs_offset row_stride %lhs_stride : !util.buffer)
             rhs(%rhs_buffer offset %rhs_offset row_stride %rhs_stride : !util.buffer)
             out(%out_buffer offset %out_offset row_stride %out_stride : !util.buffer)
             mnk(%m, %n, %k)
             flags(1) : (i8, i8, i32)
  func.return
}
