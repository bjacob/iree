// RUN: iree-compile --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-enable-microkernels %s | \
// RUN: iree-check-module \
// RUN:     --device=local-sync \
// RUN:     --executable_plugin=$IREE_BINARY_DIR/samples/custom_dispatch/cpu/plugin/builtin_ukernel_system_plugin$IREE_DYLIB_EXT \
// RUN:     --function=test_mmt4d \
// RUN:     --module=-

func.func @test_mmt4d() {
  %lhs = util.unfoldable_constant
      dense<
        [
            [
                [
                    [1.0],
                    [2.0],
                    [3.0],
                    [4.0],
                    [5.0],
                    [6.0],
                    [7.0],
                    [8.0]
                ]
            ]
        ]> : tensor<1x1x8x1xf32>
  %rhs = util.unfoldable_constant
      dense<
        [
            [
                [
                    [1.0e-1],
                    [1.0e-2],
                    [1.0e-3],
                    [1.0e-4],
                    [1.0e-5],
                    [1.0e-6],
                    [1.0e-7],
                    [1.0e-8]
                ]
            ]
        ]> : tensor<1x1x8x1xf32>
  %init_acc = util.unfoldable_constant
      dense<
        [
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
                ]
            ]
        ]> : tensor<1x1x8x8xf32>

    %result = linalg.mmt4d ins(%lhs, %rhs : tensor<1x1x8x1xf32>, tensor<1x1x8x1xf32>)
      outs(%init_acc : tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xf32>

        check.expect_almost_eq_const(%result, dense<
            [
                [
                    [
                        [1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8],
                        [2.0e-1, 2.0e-2, 2.0e-3, 2.0e-4, 2.0e-5, 2.0e-6, 2.0e-7, 2.0e-8],
                        [3.0e-1, 3.0e-2, 3.0e-3, 3.0e-4, 3.0e-5, 3.0e-6, 3.0e-7, 3.0e-8],
                        [4.0e-1, 4.0e-2, 4.0e-3, 4.0e-4, 4.0e-5, 4.0e-6, 4.0e-7, 3.0e-8],
                        [5.0e-1, 5.0e-2, 5.0e-3, 5.0e-4, 5.0e-5, 5.0e-6, 5.0e-7, 4.0e-8],
                        [6.0e-1, 6.0e-2, 6.0e-3, 6.0e-4, 6.0e-5, 6.0e-6, 6.0e-7, 5.0e-8],
                        [7.0e-1, 7.0e-2, 7.0e-3, 7.0e-4, 7.0e-5, 7.0e-6, 7.0e-7, 6.0e-8],
                        [8.0e-1, 8.0e-2, 8.0e-3, 8.0e-4, 8.0e-5, 8.0e-6, 8.0e-7, 1.0]
                    ]
                ]
            ]> : tensor<1x1x8x8xf32>) : tensor<1x1x8x8xf32>

    return
}
