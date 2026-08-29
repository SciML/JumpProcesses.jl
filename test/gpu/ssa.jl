using CUDA
include(joinpath(@__DIR__, "..", "ssa_kernel_tests.jl"))

run_ssa_kernel_tests(CUDABackend(), 100_000)
