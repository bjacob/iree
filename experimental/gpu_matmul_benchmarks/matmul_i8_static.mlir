func.func @foo(%lhs: tensor<@Mx@Kxi8>, %rhs: tensor<@Kx@Nxi8>) -> tensor<@Mx@Nxi32> {
  %c0_i32 = arith.constant 0 : i32
  %empty = tensor.empty() : tensor<@Mx@Nxi32>
  %zero = linalg.fill ins(%c0_i32 : i32) outs(%empty : tensor<@Mx@Nxi32>) -> tensor<@Mx@Nxi32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<@Mx@Kxi8>, tensor<@Kx@Nxi8>) outs(%zero: tensor<@Mx@Nxi32>) -> tensor<@Mx@Nxi32>
  return %result: tensor<@Mx@Nxi32>
}
