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

    using ElementInputA = float;
  using ElementInputB = float;
  using ElementOutput = float;
  using ElementComputeEpilogue = float;
  
  // Gemm operator cutlass_tensorop_s1688gemm_128x128_16x3_tn_align4
  using Operation_cutlass_tensorop_s1688gemm_128x128_16x3_tn_align4 = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<32, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    
    cutlass::epilogue::thread::LinearCombination<
      float,
      4,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    4,
    4,
    false,
    cutlass::arch::OpMultiplyAddFastF32
  >;
  using Gemm = Operation_cutlass_tensorop_s1688gemm_128x128_16x3_tn_align4;
  int M = 2048;
  int N = 2048;
  int K = 2048;
  cutlass::gemm::GemmCoord problem_size(M, N, K);
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);
  void* ptr_a = (void*)(cutlass_0_i0->data);
  void* ptr_b = (void*)(cutlass_0_i1->data);
  void* ptr_out = (void*)(out0->data);
  typename Gemm::Arguments arguments{
   problem_size,
   {static_cast<ElementInputA*>(ptr_a), K},
   {static_cast<ElementInputB*>(ptr_b), K},
   {static_cast<ElementOutput*>(ptr_out), N},
   {static_cast<ElementOutput*>(ptr_out), N},
   {alpha, beta},
   1};
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  Gemm gemm_op;
  cutlass::Status status = gemm_op.can_implement(arguments);
  CHECK(status == cutlass::Status::kSuccess);
  status = gemm_op.initialize(arguments, workspace.get());
  CHECK(status == cutlass::Status::kSuccess);
  status = gemm_op();
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
