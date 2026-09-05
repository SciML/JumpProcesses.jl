include(joinpath(@__DIR__, "ssa_kernel_tests.jl"))

# The same kernel, run through KernelAbstractions' CPU backend so the SSA kernel
# is covered by the ordinary test suite as well as by the GPU run. Fewer
# trajectories here since the CPU backend runs them without a GPU's parallelism.
run_ssa_kernel_tests(CPU(), 20_000)
