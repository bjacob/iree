func.func @foo(%lhs: tensor<@Mx@Kx8x4x16x2x4xf16>, %rhs: tensor<@Nx@Kx4x2x4x16x2x4xf16>) -> tensor<@Mx@Nx8x4x2x4x16x4xf32> {
  %c0_f32 = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<@Mx@Nx8x4x2x4x16x4xf32>
  %zero = linalg.fill ins(%c0_f32 : f32) outs(%empty : tensor<@Mx@Nx8x4x2x4x16x4xf32>) -> tensor<@Mx@Nx8x4x2x4x16x4xf32>
  %result = iree_gpu.multi_mma %lhs, %rhs, %zero {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ], iterator_types = [
      #iree_gpu.iterator_type<parallel>,
      #iree_gpu.iterator_type<parallel>,
      #iree_gpu.iterator_type<reduction>
    ], kind = #iree_gpu.data_tiled_mma_layout<
      intrinsic =  MFMA_F32_16x16x16_F16,
      unroll_m = 8,
      unroll_n = 2,
      unroll_n_to_subgroups = 4,
      unroll_k = 2
    >} : tensor<@Mx@Kx8x4x16x2x4xf16>, tensor<@Nx@Kx4x2x4x16x2x4xf16> into tensor<@Mx@Nx8x4x2x4x16x4xf32>
  return %result : tensor<@Mx@Nx8x4x2x4x16x4xf32>
}
