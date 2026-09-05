include(joinpath(@__DIR__, "explicit_tau_kernel_tests.jl"))

# The same kernel, run through KernelAbstractions' CPU backend so it is covered
# by the ordinary test suite as well as by the GPU run. Fewer trajectories here
# since the CPU backend runs them without a GPU's parallelism.
run_explicit_tau_kernel_tests(CPU(), 20_000)
