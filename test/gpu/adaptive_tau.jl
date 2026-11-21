using CUDA
include(joinpath(@__DIR__, "..", "adaptive_tau_kernel_tests.jl"))

run_adaptive_tau_kernel_tests(CUDABackend(), 100_000)
