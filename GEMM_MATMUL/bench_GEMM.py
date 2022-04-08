import tvm
import re
from tvm import relay
import numpy as np
from tvm.runtime.vm import VirtualMachine
from tvm.topi import testing
from tvm.relay.op.contrib.cutlass import partition_for_cutlass
from tvm.contrib import cudnn
from tvm.contrib.cutlass import (
    tune_cutlass_kernels,
    build_cutlass_kernels,
)

import logging

logging.basicConfig(level=logging.INFO)

def has_cutlass():
    return tvm.get_global_func("relay.ext.cutlass", True) != None


def get_output(rt_mod, names, inputs):
    for name, inp in zip(names, inputs):
        rt_mod.set_input(name, inp)
    rt_mod.run()
    return rt_mod.get_output(0).asnumpy()


def profile_and_build(
    mod, params, sm, split_k_slices=[1], tmp_dir="./tmp", lib_path="compile.so", use_multiprocessing=False,
    BENCHMARK_TRIALS=1, USETENSORCORE=True
):
    # params : dict of str to NDArray
    # Input parameters to the graph that do not change
    # during inference time. Used for constant folding.

    # split_k_slices is ignored for GEMM, kept here to preserve semantics

    mod = partition_for_cutlass(mod)
    mod, num_cutlass_partition = tune_cutlass_kernels(
        mod,
        sm,
        split_k_slices=split_k_slices,
        find_first_valid=False,
        use_multiprocessing=use_multiprocessing,
        tmp_dir=tmp_dir,
        BENCHMARK_TRIALS=BENCHMARK_TRIALS,
        USETENSORCORE=USETENSORCORE
    )
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target="cuda", params=params)
    lib = build_cutlass_kernels(lib, sm, tmp_dir, lib_path)
    dev = tvm.device("cuda", 0)
    rt_mod = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    return mod, rt_mod, dev, num_cutlass_partition


def verify_GEMM(mod_GEMM, params, input_names, inputs, sm=75, split_k_slices=[1],
    filename="best_dense_gemm_schedule.txt", use_multiprocessing=False, BENCHMARK_TRIALS=1, USETENSORCORE=True):
    # Setting default sm to 75 for cloud AWS system

    # Input parameters to the graph that do not change
    # during inference time. Used for constant folding.
    # removed from input to function and placed here

    # split_k_slices is ignored for GEMM, kept here to preserve semantics
    if not has_cutlass():
        return

    typ = relay.transform.InferType()(mod_GEMM)
    # print(typ)
    # print("oshape:", typ["main"].body.checked_type.shape)

    best_mod, rt_mod, dev, num_cutlass_partition = profile_and_build(
        mod_GEMM, params, sm, split_k_slices, use_multiprocessing=use_multiprocessing,
        BENCHMARK_TRIALS=BENCHMARK_TRIALS, USETENSORCORE=USETENSORCORE
    )
    
    # console has a warning when exporting astext(); can be ignored
    exported_schedule_string = best_mod.astext()

    with open(filename, "w") as outfile:
        outfile.writelines(exported_schedule_string)
    # print(best_mod)
    # Ashwin: Useful??
    out = get_output(rt_mod, input_names, inputs)

    # assert num_cutlass_partition > 0
    
    cutlass_time = rt_mod.benchmark(dev, number=1, repeat=1000).mean * 1000
    print("GEMM CUTLASS Time: ", cutlass_time, " ms")
    
    return cutlass_time


def get_GEMM(shape_a, shape_b, in_dtype="float16", out_dtype="float16"):
    # defining GEMM representation with dense layer
    # naive matmul is not supported with CUTLASS for tensor core
    # batched matmul throwing error in partioning stage
    tensor_a = relay.var("data", shape=shape_a, dtype=in_dtype)
    tensor_b = relay.var("weight", shape=shape_b, dtype=in_dtype)

    # By default, batched_matmul has matrix B as transposed
    # return tvm.IRModule.from_expr(
    #     relay.nn.batch_matmul(
    #         tensor_a=tensor_a,
    #         tensor_b=tensor_b,
    #         out_dtype=out_dtype,
    #         transpose_a=False,
    #         transpose_b=True)
    #         )

    return tvm.IRModule.from_expr(
        tvm.relay.nn.dense(
            data=tensor_a,
            weight=tensor_b,
            out_dtype=out_dtype)
            )

def Benchmark_GEMM(sm_num, shape_a, shape_b, filename,
                    in_dtype="float16", out_dtype="float16", use_multiprocessing=False,
                    BENCHMARK_TRIALS=1, USETENSORCORE=True):

    if not(USETENSORCORE):
        assert in_dtype=="float32" and out_dtype=="float32"

    mod_GEMM = get_GEMM(shape_a=shape_a, shape_b=shape_b, in_dtype=in_dtype, out_dtype=out_dtype)

    np_data = np.random.uniform(-1, 1, shape_a).astype(in_dtype)
    np_weight = np.random.uniform(-1, 1, shape_b).astype(in_dtype)
    params = {"weight": np_weight}
    input_names = ["data"]
    inputs = [np_data]
    split_k_slices = [1]

    verify_GEMM(mod_GEMM, params, input_names, inputs, sm=sm_num, split_k_slices=split_k_slices,
                filename=filename, use_multiprocessing=use_multiprocessing, BENCHMARK_TRIALS=BENCHMARK_TRIALS, USETENSORCORE=USETENSORCORE)

if __name__=="__main__":
    sm_num = 75
    M = 1024
    N = 1024
    K = 1024
    use_multiprocessing = False
    BENCHMARK_TRIALS = 1

    shape_a = (M, K)
    shape_b = (K, N)
    
    filename = "SM_" + str(sm_num) + "_" + "best_dense_gemm_schedule_" + str(M) + "_" + str(N) + "_" + str(K) + ".txt"

    USETENSORCORE = False
    
    print("Measuring for (M,N,K): (" + str(M) + "," +  str(N) + "," + str(K) + ")")

    Benchmark_GEMM(sm_num, shape_a, shape_b, filename=filename,
                    in_dtype="float32", out_dtype="float32", use_multiprocessing=use_multiprocessing,
                    BENCHMARK_TRIALS=BENCHMARK_TRIALS, USETENSORCORE=USETENSORCORE)
