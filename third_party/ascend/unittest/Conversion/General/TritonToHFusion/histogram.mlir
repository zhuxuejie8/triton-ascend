// RUN: triton-opt %s -triton-to-hfusion  | FileCheck %s


// CHECK-LABEL: tt.func @test_histogram
<<<<<<< HEAD
tt.func @test_histogram(%arg0 : tensor<16xi32>) -> tensor<2xi32> {
    // CHECK: hfusion.histogram %{{.*}}, 2
    %res = tt.histogram %arg0 : tensor<16xi32> -> tensor<2xi32>
    tt.return %res : tensor<2xi32>
}
=======
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func @test_histogram(%arg0 : tensor<16xi32>) -> tensor<2xi32> {
    // CHECK: hfusion.histogram %{{.*}}, 2
    %res = tt.histogram %arg0 : tensor<16xi32> -> tensor<2xi32>
    tt.return %res : tensor<2xi32>
  }
}
>>>>>>> release-3.2.2-0625-b79d137
