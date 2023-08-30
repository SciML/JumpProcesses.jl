# This file is not directly included in a test case, but is used to
# benchmark and compare changes to the broadcsting
using JumpProcesses, StableRNGs, FastBroadcast, BenchmarkTools

rng = StableRNG(123)

base_case_out = zeros(500000 * 2)
base_case_in = rand(rng, 500000 * 2)
benchmark_out = ExtendedJumpArray(zeros(500000), zeros(500000))
benchmark_in = ExtendedJumpArray(rand(rng, 500000), rand(rng, 500000))

function test_single_dot(out, array)
    @inbounds @. out = array + 1.23 * array
end

function test_double_dot(out, array)
    @inbounds @.. out = array + 1.23 * array
end

println("Base-case normal broadcasting")
@benchmark test_single_dot(base_case_out, base_case_in)
println("EJA normal broadcasting")
@benchmark test_single_dot(benchmark_out, benchmark_in)
println("Base-case fast broadcasting")
@benchmark test_double_dot(base_case_out, base_case_in)
println("EJA fast broadcasting")
@benchmark test_double_dot(benchmark_out, benchmark_in)
