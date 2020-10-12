func @dot_general_4_x384x32x384() -> tensor<4x384x384xf32> attributes { iree.module.export } {
    %lhs = iree.unfoldable_constant dense<1.0> : tensor<4x384x32xf32>
    %rhs = iree.unfoldable_constant dense<1.0> : tensor<4x32x384xf32>
    %0 = "mhlo.dot_general"(%lhs, %rhs) {
        dot_dimension_numbers = {
            lhs_batching_dimensions = dense<0> : tensor<1xi64>, 
            lhs_contracting_dimensions = dense<2> : tensor<1xi64>, 
            rhs_batching_dimensions = dense<0> : tensor<1xi64>, 
            rhs_contracting_dimensions = dense<1> : tensor<1xi64>
        }
    } : (tensor<4x384x32xf32>, tensor<4x32x384xf32>) -> tensor<4x384x384xf32>
    return %0 : tensor<4x384x384xf32>
}

func @dot_general_4_x384x384x32() -> tensor<4x384x32xf32> attributes { iree.module.export } {
    %lhs = iree.unfoldable_constant dense<1.0> : tensor<4x384x384xf32>
    %rhs = iree.unfoldable_constant dense<1.0> : tensor<4x384x32xf32>
    %0 = "mhlo.dot_general"(%lhs, %rhs) {
        dot_dimension_numbers = {
            lhs_batching_dimensions = dense<0> : tensor<1xi64>, 
            lhs_contracting_dimensions = dense<2> : tensor<1xi64>, 
            rhs_batching_dimensions = dense<0> : tensor<1xi64>, 
            rhs_contracting_dimensions = dense<1> : tensor<1xi64>
        }
    } : (tensor<4x384x384xf32>, tensor<4x384x32xf32>) -> tensor<4x384x32xf32>
    return %0 : tensor<4x384x32xf32>
}
