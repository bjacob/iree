func private @actual_matmul(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>, %acc : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %result =  linalg.matmul ins(%lhs, %rhs: tensor<?x?xf32>, tensor<?x?xf32>)
                    outs(%acc: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}

#matmul_accesses = [
  affine_map<(m, k, n) -> (m, k)>,
  affine_map<(m, k, n) -> (k, n)>,
 affine_map< (m, k, n) -> (m, n)>
]
#matmul_trait = {
  indexing_maps = #matmul_accesses,
  iterator_types = ["parallel", "reduction", "parallel"]
}

func private @expected_matmul(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>, %acc : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %result =  linalg.generic #matmul_trait ins(%lhs, %rhs: tensor<?x?xf32>, tensor<?x?xf32>)
                    outs(%acc: tensor<?x?xf32>) {
    ^bb0(%lhs_value: f32, %rhs_value: f32, %acc_value: f32):
      %product = std.mulf %lhs_value, %rhs_value: f32
      %sum = std.addf %product, %acc_value: f32
      linalg.yield %sum: f32
  } -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}

func private @zero_matrix(%rows : index, %cols : index) -> tensor<?x?xf32> {
  %0 = linalg.init_tensor [%rows, %cols] : tensor<?x?xf32>
  %c0 = constant 0.0 : f32
  %1 = linalg.fill (%c0, %0) : f32, tensor<?x?xf32> -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

func private @pseudorandom_matrix(%rows : index, %cols : index) -> tensor<?x?xf32> {
  %0 = linalg.init_tensor [%rows, %cols] : tensor<?x?xf32>
  %min = constant -1.0 : f64
  %max = constant 1.0 : f64
  %seed = constant 12345 : i32
  %1 = linalg.fill_rng_2d ins(%min, %max, %seed: f64, f64, i32) outs (%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

func private @identity_matrix(%rows : index, %cols : index) -> tensor<?x?xf32> {
  %0 = linalg.init_tensor [%rows, %cols] : tensor<?x?xf32>
  %1 = linalg.generic {
         indexing_maps = [
           affine_map<(d0, d1) -> (d0, d1)>,
           affine_map<(d0, d1) -> (d0, d1)>
         ],
         iterator_types = ["parallel", "parallel"]
       }
       ins(%0 : tensor<?x?xf32>)
       outs(%0 : tensor<?x?xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
    %8 = linalg.index 0 : index
    %9 = linalg.index 1 : index
    %10 = cmpi eq, %8, %9 : index
    %cst0 = constant 0.000000e+00 : f32
    %cst1 = constant 1.000000e+00 : f32
    %11 = select %10, %cst1, %cst0 : f32
    linalg.yield %11 : f32
  } -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
