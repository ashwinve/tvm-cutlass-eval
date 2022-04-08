#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/packed_func.h>
#include <dlpack/dlpack.h>
#include <cuda_fp16.h>
#include <cutlass/cutlass.h>
#include <cutlass/coord.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/conv/kernel/default_conv2d_fprop.h>
#include <cutlass/conv/kernel/default_conv2d_wgrad.h>
#include <cutlass/conv/kernel/default_conv2d_dgrad.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/epilogue/thread/linear_combination_bias_relu.h>
#include <cutlass/epilogue/thread/linear_combination_gelu.h>
#include <cutlass/epilogue/thread/linear_combination_sigmoid.h>
#include <cutlass/epilogue/thread/linear_combination_silu.h>
#include <cutlass/epilogue/thread/linear_combination_hardswish.h>
#include <cutlass/epilogue/thread/linear_combination_residual_block.h>
#include <cutlass/conv/kernel/default_conv2d_fprop_with_broadcast.h>
#include <cutlass/reduction/device/reduce_split_k.h>
#include <cutlass/reduction/thread/reduction_operators.h>
void tvmgen_default_cutlass_main_0_(DLTensor* cutlass_0_i0, DLTensor* cutlass_0_i1, DLTensor* out0) {

    
  // Conv2dFprop Optimized kernel instance "cutlass_tensorop_h1688fprop_optimized_256x128_32x2_nhwc_align8"
  using cutlass_tensorop_h1688fprop_optimized_256x128_32x2_nhwc_align8 =
  typename cutlass::conv::kernel::DefaultConv2dFprop<
    cutlass::half_t,
    cutlass::layout::TensorNHWC,
    cutlass::half_t,
    cutlass::layout::TensorNHWC,
    cutlass::half_t,
    cutlass::layout::TensorNHWC,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<256, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32 >,
    cutlass::gemm::GemmShape<16, 8, 8>,
    
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      8,
      cutlass::half_t,
      cutlass::half_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
    2,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized,
    cutlass::conv::StrideSupport::kStrided,
    8,
    8
  >::Kernel;

  
  using Operation_cutlass_tensorop_h1688fprop_optimized_256x128_32x2_nhwc_align8 = cutlass::conv::device::ImplicitGemmConvolution<cutlass_tensorop_h1688fprop_optimized_256x128_32x2_nhwc_align8>;
  using Conv2d = Operation_cutlass_tensorop_h1688fprop_optimized_256x128_32x2_nhwc_align8;
  using ElementInputA = Conv2d::ElementA;
  using ElementInputB = Conv2d::ElementB;
  using ElementComputeEpilogue = Conv2d::ElementAccumulator;
  int N = 256;
  int H = 56;
  int W = 56;
  int C = 64;
  int K = 256;
  int R = 1;
  int S = 1;
  int P = 56;
  int Q = 56;
  int pad_h = 0;
  int pad_w = 0;
  int stride_h = 1;
  int stride_w = 1;
  int dilation_h = 1;
  int dilation_w = 1;
  int split_k_slices = 1;
  cutlass::conv::Conv2dProblemSize problem_size(N, H, W, C, K, R, S, P, Q, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, cutlass::conv::Mode::kCrossCorrelation, split_k_slices);
  const cutlass::conv::SplitKMode split_k_mode = cutlass::conv::SplitKMode::kSerial;
  void* ptr_a = (void*)(cutlass_0_i0->data);
  void* ptr_b = (void*)(cutlass_0_i1->data);
  void* ptr_out = (void*)(out0->data);
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);
  using cutlass::layout::TensorNHWC;
  auto activation_shape = TensorNHWC::packed(cutlass::make_Coord(N, H, W, C));
  auto weight_shape = TensorNHWC::packed(cutlass::make_Coord(K, R, S, C));
  auto output_oshape = TensorNHWC::packed(cutlass::make_Coord(N, P, Q, K));
  TensorNHWC layout_A(activation_shape);
  TensorNHWC layout_B(weight_shape);
  TensorNHWC layout_C(output_oshape);

  TensorNHWC layout_D(output_oshape);

  using ElementOutput = Conv2d::ElementC;
  cutlass::TensorRef<ElementOutput, TensorNHWC> tensor_c{static_cast<ElementOutput*>(ptr_out), layout_C};
  cutlass::TensorRef<ElementOutput, TensorNHWC> tensor_d{static_cast<ElementOutput*>(ptr_out),layout_D};
  typename Conv2d::Arguments arguments{
   problem_size,
   {static_cast<ElementInputA*>(ptr_a), layout_A},
   {static_cast<ElementInputB*>(ptr_b), layout_B},
   tensor_c,
   tensor_d,
  {alpha, beta},
  split_k_mode
};
  Conv2d conv2d_op;
  size_t workspace_size = conv2d_op.get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = conv2d_op.can_implement(arguments);
  CHECK(status == cutlass::Status::kSuccess);

  status = conv2d_op.initialize(arguments, workspace.get());
  CHECK(status == cutlass::Status::kSuccess);

  status = conv2d_op();
  CHECK(status == cutlass::Status::kSuccess);


}

int tvmgen_default_cutlass_main_0_wrapper_(DLTensor* arg0,
	DLTensor* arg1,
	DLTensor* out0) {
  tvmgen_default_cutlass_main_0_(arg0,
  arg1,
  out0);
  return 0;
}

#ifdef __cplusplus
extern "C" {
#endif
TVM_DLL int32_t tvmgen_default_cutlass_main_0(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code) {
  DLTensor* arg0 = (DLTensor*)(((TVMValue*)args)[0].v_handle);
  DLTensor* arg1 = (DLTensor*)(((TVMValue*)args)[1].v_handle);
  DLTensor* ret2 = (DLTensor*)(((TVMValue*)args)[2].v_handle);
  tvmgen_default_cutlass_main_0_wrapper_(arg0,arg1,ret2);
  return 0;
}
#ifdef __cplusplus
}
#endif
