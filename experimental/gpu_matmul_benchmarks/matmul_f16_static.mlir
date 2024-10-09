func.func @foo(%lhs: tensor<@Mx@Kxf16>, %rhs: tensor<@Kx@Nxf16>) -> tensor<@Mx@Nxf32> {
  %c0_f32 = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<@Mx@Nxf32>
  %zero = linalg.fill ins(%c0_f32 : f32) outs(%empty : tensor<@Mx@Nxf32>) -> tensor<@Mx@Nxf32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<@Mx@Kxf16>, tensor<@Kx@Nxf16>) outs(%zero: tensor<@Mx@Nxf32>) -> tensor<@Mx@Nxf32>
  return %result: tensor<@Mx@Nxf32>
}
