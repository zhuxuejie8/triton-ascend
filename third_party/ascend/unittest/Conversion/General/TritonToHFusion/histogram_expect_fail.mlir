// RUN: ! triton-opt %s -triton-to-hfusion 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL-ZERO
// Expected failure case, only for branch coverage
// CHECK-FAIL-ZERO: error: 'hfusion.histogram' op output length (0) must equal num_bins
<<<<<<< HEAD
tt.func @test_histogram_fail_with_zero_n(%arg0 : tensor<16xi32>) -> tensor<0xi32> {
    %res = tt.histogram %arg0 : tensor<16xi32> -> tensor<0xi32>
    tt.return %res : tensor<0xi32>
=======
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func @test_histogram_fail_with_zero_n(%arg0 : tensor<16xi32>) -> tensor<0xi32> {
    %res = tt.histogram %arg0 : tensor<16xi32> -> tensor<0xi32>
    tt.return %res : tensor<0xi32>
  }
>>>>>>> release-3.2.2-0625-b79d137
}


// RUN: ! triton-opt %s -triton-to-hfusion 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL-DYN
// Expected failure case, only for branch coverage
// CHECK-FAIL-DYN: error: 'hfusion.histogram' op output must have static shape
<<<<<<< HEAD
tt.func @test_histogram_fail_with_dyn_n(%arg0 : tensor<16xi32>) -> tensor<?xi32> {
    %res = tt.histogram %arg0 : tensor<16xi32> -> tensor<?xi32>
    tt.return %res : tensor<?xi32>
}
=======
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func @test_histogram_fail_with_dyn_n(%arg0 : tensor<16xi32>) -> tensor<?xi32> {
    %res = tt.histogram %arg0 : tensor<16xi32> -> tensor<?xi32>
    tt.return %res : tensor<?xi32>
  }
}
>>>>>>> release-3.2.2-0625-b79d137
