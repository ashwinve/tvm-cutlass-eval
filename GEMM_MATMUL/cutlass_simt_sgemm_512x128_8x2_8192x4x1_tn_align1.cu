
#include <iostream>
#include <sstream>
#include <vector>
#include <chrono>

#include "cuda_runtime.h"
#include "cutlass/gemm/device/gemm.h"

#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

template<typename DTypeA, typename DTypeB, typename DTypeC>
cudaError_t CutlassGemmRCR(
    int M,
    int N,
    int K,
    DTypeC alpha,
    DTypeA const *A,
    int lda,
    DTypeB const *B,
    int ldb,
    DTypeC beta,
    DTypeC *C,
    int ldc) {
  using namespace std::chrono;
  
  // Gemm operator cutlass_simt_sgemm_512x128_8x2_8192x4x1_tn_align1
  using Operation_cutlass_simt_sgemm_512x128_8x2_8192x4x1_tn_align1 = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm50,
    cutlass::gemm::GemmShape<512, 128, 8>,
    cutlass::gemm::GemmShape<0, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    
    cutlass::epilogue::thread::LinearCombination<
      float,
      1,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    2,
    1,
    1,
    false,
    cutlass::arch::OpMultiplyAdd
  >;

  Operation_cutlass_simt_sgemm_512x128_8x2_8192x4x1_tn_align1 gemm_operator;
  Operation_cutlass_simt_sgemm_512x128_8x2_8192x4x1_tn_align1::Arguments args({M, N, K},
                              {A, lda},
                              {B, ldb},
                              {C, ldc},
                              {C, ldc},
                              {alpha, beta});
  cutlass::Status status = gemm_operator(args);
  CUTLASS_CHECK(status)

  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  for (int i = 0; i < 100; ++i) {
    status = gemm_operator(args);
  }
  cudaDeviceSynchronize();
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
  std::cout << time_span.count() << std::endl;
  return cudaSuccess;
}


template<typename DType>
cudaError_t AllocateMatrix(DType **matrix, int ldm, int rows, int columns, int seed = 0) {
  cudaError_t result;

  size_t sizeof_matrix = sizeof(DType) * rows * columns;

  // Allocate device memory.
  result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Clear the allocation.
  result = cudaMemset(*matrix, 0, sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to clear matrix device memory: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  if (result != cudaSuccess) {
    std::cerr << "Failed to initialize matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  return result;
}

template<typename DTypeA, typename DTypeB, typename DTypeC>
cudaError_t TestCutlassGemm(int M, int N, int K, DTypeC alpha, DTypeC beta) {
  cudaError_t result;

  int lda = K;
	int ldb = K;
	int ldc = N;

  // size_t sizeof_C = sizeof(DTypeC) * ldc * N;
  DTypeA *A;
  DTypeB *B;
  DTypeC *C_cutlass;
  result = AllocateMatrix<DTypeA>(&A, lda, M, K, 0);
  if (result !=  cudaSuccess) {
    return result;
  }
  result = AllocateMatrix<DTypeB>(&B, ldb, K, N, 17);
  if (result !=  cudaSuccess) {
    cudaFree(A);
    return result;
  }
  result = AllocateMatrix<DTypeC>(&C_cutlass, ldc, M, N, 101);
  if (result != cudaSuccess) {
    cudaFree(A);
    cudaFree(B);
    return result;
  }
  result = CutlassGemmRCR<DTypeA, DTypeB, DTypeC>(M, N, K, alpha, A, lda, B, ldb,
                                                  beta, C_cutlass, ldc);
  if (result != cudaSuccess) {
    std::cerr << "CUTLASS GEMM kernel failed: "
      << cudaGetErrorString(result) << std::endl;
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }
  cudaFree(C_cutlass);
  cudaFree(B);
  cudaFree(A);
  return cudaSuccess;
}

int main(int argc, const char *arg[]) {
  int problem[3] = { 4096, 4096, 4096 };
  for (int i = 1; i < argc && i < 4; ++i) {
    std::stringstream ss(arg[i]);
    ss >> problem[i - 1];
  }
  float scalars[2] = { 1, 0 };
  cudaError_t result = TestCutlassGemm< float, float, float>(
    problem[0],     // GEMM M dimension
    problem[1],     // GEMM N dimension
    problem[2],     // GEMM K dimension
    static_cast<float>(scalars[0]),     // alpha
    static_cast<float>(scalars[1])      // beta
  );
  return result == cudaSuccess ? 0 : -1;
}