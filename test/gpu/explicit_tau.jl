using CUDA
include(joinpath(@__DIR__, "..", "explicit_tau_kernel_tests.jl"))

run_explicit_tau_kernel_tests(CUDABackend(), 100_000)
