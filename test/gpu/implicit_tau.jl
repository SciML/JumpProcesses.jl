using CUDA
include(joinpath(@__DIR__, "..", "implicit_tau_kernel_tests.jl"))

run_implicit_tau_kernel_tests(CUDABackend(), 100_000)
