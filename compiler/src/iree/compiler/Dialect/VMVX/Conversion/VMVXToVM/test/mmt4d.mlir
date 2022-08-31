// RUN: iree-opt --iree-vm-target-index-bits=64 --split-input-file \
// RUN:   --iree-vm-conversion --canonicalize %s | FileCheck %s

// CHECK-LABEL: @mmt4d_f32f32f32
  // CHECK-SAME: %[[lhs_buffer:[^:]+]]: !vm.buffer, %[[lhs_offset:[^:]+]]: i64,
  // CHECK-SAME: %[[rhs_buffer:[^:]+]]: !vm.buffer, %[[rhs_offset:[^:]+]]: i64,
  // CHECK-SAME: %[[out_buffer:[^:]+]]: !vm.buffer, %[[out_offset:[^:]+]]: i64,
  // CHECK-SAME: %[[lhs_stride:[^:]+]]: i64, %[[rhs_stride:[^:]+]]: i64, %[[out_stride:[^:]+]]: i64,
  // CHECK-SAME: %[[m:[^:]+]]: i64, %[[n:[^:]+]]: i64, %[[k:[^:]+]]: i64,
  // CHECK-SAME: %[[tile_m:[^:]+]]: i64, %[[tile_n:[^:]+]]: i64, %[[tile_k:[^:]+]]: i64) {

  // CHECK-DAG: %[[flags:.+]] = vm.const.i32 1
  // CHECK-DAG: %[[tile_m_i32:.+]] = vm.trunc.i64.i32 %[[tile_m]] : i64 -> i32
  // CHECK-DAG: %[[tile_n_i32:.+]] = vm.trunc.i64.i32 %[[tile_n]] : i64 -> i32
  // CHECK-DAG: %[[tile_k_i32:.+]] = vm.trunc.i64.i32 %[[tile_k]] : i64 -> i32
  //      CHECK: vm.call @vmvx.mmt4d.f32f32f32(
  // CHECK-SAME: %[[lhs_buffer]], %[[lhs_offset]],
  // CHECK-SAME: %[[rhs_buffer]], %[[rhs_offset]],
  // CHECK-SAME: %[[out_buffer]], %[[out_offset]],
  // CHECK-SAME: %[[lhs_stride]], %[[rhs_stride]], %[[out_stride]],
  // CHECK-SAME: %[[m]], %[[n]], %[[k]],
  // CHECK-SAME: %[[tile_m_i32]], %[[tile_n_i32]], %[[tile_k_i32]],
  // CHECK-SAME: %[[flags]]) : (!vm.buffer, i64, !vm.buffer, i64, !vm.buffer, i64, i64, i64, i64, i64, i64, i64, i32, i32, i32, i32) -> ()


func.func @mmt4d_f32f32f32(
    // LHS
    %lhs_buffer : !util.buffer, %lhs_offset : index,
    // RHS
    %rhs_buffer : !util.buffer, %rhs_offset : index,
    // OUT
    %out_buffer : !util.buffer, %out_offset : index,
    // STRIDES
    %lhs_stride : index, %rhs_stride : index, %out_stride : index,
    // MNK
    %m : index, %n : index, %k : index,
    // TILE_MNK
    %tile_m : index, %tile_n : index, %tile_k : index
    ) {
  vmvx.mmt4d lhs(%lhs_buffer offset %lhs_offset row_stride %lhs_stride : !util.buffer)
             rhs(%rhs_buffer offset %rhs_offset row_stride %rhs_stride : !util.buffer)
             out(%out_buffer offset %out_offset row_stride %out_stride : !util.buffer)
             mnk(%m, %n, %k)
             tile_mnk(%tile_m, %tile_n, %tile_k)
             flags(1) : (f32, f32, f32)
  func.return
}

// CHECK-LABEL: @mmt4d_i8i8i32
  // CHECK-SAME: %[[lhs_buffer:[^:]+]]: !vm.buffer, %[[lhs_offset:[^:]+]]: i64,
  // CHECK-SAME: %[[rhs_buffer:[^:]+]]: !vm.buffer, %[[rhs_offset:[^:]+]]: i64,
  // CHECK-SAME: %[[out_buffer:[^:]+]]: !vm.buffer, %[[out_offset:[^:]+]]: i64,
  // CHECK-SAME: %[[lhs_stride:[^:]+]]: i64, %[[rhs_stride:[^:]+]]: i64, %[[out_stride:[^:]+]]: i64,
  // CHECK-SAME: %[[m:[^:]+]]: i64, %[[n:[^:]+]]: i64, %[[k:[^:]+]]: i64,
  // CHECK-SAME: %[[tile_m:[^:]+]]: i64, %[[tile_n:[^:]+]]: i64, %[[tile_k:[^:]+]]: i64) {

  // CHECK-DAG: %[[flags:.+]] = vm.const.i32 1
  // CHECK-DAG: %[[tile_m_i32:.+]] = vm.trunc.i64.i32 %[[tile_m]] : i64 -> i32
  // CHECK-DAG: %[[tile_n_i32:.+]] = vm.trunc.i64.i32 %[[tile_n]] : i64 -> i32
  // CHECK-DAG: %[[tile_k_i32:.+]] = vm.trunc.i64.i32 %[[tile_k]] : i64 -> i32
  //      CHECK: vm.call @vmvx.mmt4d.i8i8i32(
  // CHECK-SAME: %[[lhs_buffer]], %[[lhs_offset]],
  // CHECK-SAME: %[[rhs_buffer]], %[[rhs_offset]],
  // CHECK-SAME: %[[out_buffer]], %[[out_offset]],
  // CHECK-SAME: %[[lhs_stride]], %[[rhs_stride]], %[[out_stride]],
  // CHECK-SAME: %[[m]], %[[n]], %[[k]],
  // CHECK-SAME: %[[tile_m_i32]], %[[tile_n_i32]], %[[tile_k_i32]],
  // CHECK-SAME: %[[flags]]) : (!vm.buffer, i64, !vm.buffer, i64, !vm.buffer, i64, i64, i64, i64, i64, i64, i64, i32, i32, i32, i32) -> ()


func.func @mmt4d_i8i8i32(
    // LHS
    %lhs_buffer : !util.buffer, %lhs_offset : index,
    // RHS
    %rhs_buffer : !util.buffer, %rhs_offset : index,
    // OUT
    %out_buffer : !util.buffer, %out_offset : index,
    // STRIDES
    %lhs_stride : index, %rhs_stride : index, %out_stride : index,
    // MNK
    %m : index, %n : index, %k : index,
    // TILE_MNK
    %tile_m : index, %tile_n : index, %tile_k : index
    ) {
  vmvx.mmt4d lhs(%lhs_buffer offset %lhs_offset row_stride %lhs_stride : !util.buffer)
             rhs(%rhs_buffer offset %rhs_offset row_stride %rhs_stride : !util.buffer)
             out(%out_buffer offset %out_offset row_stride %out_stride : !util.buffer)
             mnk(%m, %n, %k)
             tile_mnk(%tile_m, %tile_n, %tile_k)
             flags(1) : (i8, i8, i32)
  func.return
}
