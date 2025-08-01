using Test, JumpProcesses, DiffEqBase, OrdinaryDiffEq, SciMLBase
using FastBroadcast
using StableRNGs

rng = StableRNG(123)

# Check that the new broadcast norm gives the same result as the old one
rand_array = ExtendedJumpArray{Float64, 1, Vector{Float64}, Vector{Float64}}(rand(rng, 5),
    rand(rng, 2))
old_norm = Base.FastMath.sqrt_fast(DiffEqBase.UNITLESS_ABS2(rand_array) / max(DiffEqBase.recursive_length(rand_array), 1))
new_norm = DiffEqBase.ODE_DEFAULT_NORM(rand_array, 0.0)
@test old_norm ≈ new_norm

# Check for an ExtendedJumpArray where the types differ (Float64/Int64)
rand_array = ExtendedJumpArray{Float64, 1, Vector{Float64}, Vector{Int64}}(rand(rng, 5),
    rand(rng, 1:1000,
        2))
old_norm = Base.FastMath.sqrt_fast(DiffEqBase.UNITLESS_ABS2(rand_array) / max(DiffEqBase.recursive_length(rand_array), 1))
new_norm = DiffEqBase.ODE_DEFAULT_NORM(rand_array, 0.0)
@test old_norm ≈ new_norm

# Check that we no longer allocate. Run inside function so @allocated works properly
norm_check_alloc(jump_array, t) = @allocated DiffEqBase.ODE_DEFAULT_NORM(jump_array, t)
norm_check_alloc(rand_array, 0.0);
@test 0 == norm_check_alloc(rand_array, 0.0)

## Broadcasting
bc_eja_1 = ExtendedJumpArray(rand(rng, 10), rand(rng, 2))
bc_eja_2 = ExtendedJumpArray(rand(rng, 10), rand(rng, 2))
bc_out = ExtendedJumpArray(zeros(10), zeros(2))

# Test that broadcasting gives the same output as non-broadcasted math
@test bc_eja_1 + bc_eja_2 ≈ bc_eja_1 .+ bc_eja_2
@test 3.14 * bc_eja_1 + 2.7 * bc_eja_2 ≈ 3.14 .* bc_eja_1 .+ 2.7 .* bc_eja_2

# Test that non-allocating (copyto!) gives the same result, both w/ and w/o the dot macro
bc_out .= 3.14 .* bc_eja_1 + 2.7 .* bc_eja_2
@test bc_out ≈ 3.14 * bc_eja_1 + 2.7 * bc_eja_2
@. bc_out = 3.14 * bc_eja_1 + 2.7 * bc_eja_2
@test bc_out ≈ 3.14 * bc_eja_1 + 2.7 * bc_eja_2

# Test that mismatched arrays cannot be broadcasted
bc_mismatch = ExtendedJumpArray(rand(rng, 8), rand(rng, 4))
@test_throws DimensionMismatch bc_mismatch+bc_eja_1
@test_throws DimensionMismatch bc_mismatch .+ bc_eja_1

# Test that datatype mixing persists through broadcasting
bc_dtype_1 = ExtendedJumpArray(rand(rng, 10), rand(rng, 1:10, 2))
bc_dtype_2 = ExtendedJumpArray(rand(rng, 10), rand(rng, 1:10, 2))
result = bc_dtype_1 + bc_dtype_2 * 2
@test eltype(result.jump_u) == Int64
out_result = ExtendedJumpArray(zeros(10), zeros(2))
out_result .= bc_dtype_1 .+ bc_dtype_2 .* 2
@test eltype(result.jump_u) == Int64
@test out_result ≈ result

# Test that fast broadcasting also gives the correct results
@.. bc_out = 3.14 * bc_eja_1 + 2.7 * bc_eja_2
@test bc_out ≈ 3.14 * bc_eja_1 + 2.7 * bc_eja_2

# Test both the in-place and allocating problems (https://github.com/SciML/JumpProcesses.jl/issues/321)
# to check that an ExtendedJumpArray is not getting downgraded into a Vector
oop_test_rate(u, p, t) = exp(t)
function oop_test_affect!(integrator)
    integrator.u[1] += 1
    nothing
end
oop_test_jump = VariableRateJump(oop_test_rate, oop_test_affect!)

# Test in-place
u₀ = [0.0]
inplace_prob = ODEProblem((du, u, p, t) -> (du .= 0), u₀, (0.0, 2.0), nothing)
jump_prob = JumpProblem(inplace_prob, Direct(), oop_test_jump; vr_aggregator = VR_FRM())
sol = solve(jump_prob, Tsit5())
@test sol.retcode == ReturnCode.Success
sol.u

# Test out-of-place
u₀ = [0.0]
oop_prob = ODEProblem((u, p, t) -> [0.0], u₀, (0.0, 2.0), nothing) # only difference is use of OOP ode function
jump_prob = JumpProblem(oop_prob, Direct(), oop_test_jump)
sol = solve(jump_prob, Tsit5())
@test sol.retcode == ReturnCode.Success

# Test saveat https://github.com/SciML/JumpProcesses.jl/issues/179 and https://github.com/SciML/JumpProcesses.jl/issues/322
let
    f(du, u, p, t) = (du[1] = u[1]; nothing)
    prob = ODEProblem(f, [0.2], (0.0, 10.0))
    rate(u, p, t) = u[1]
    affect!(integrator) = (integrator.u.u[1] = integrator.u.u[1] / 2; nothing)
    jump = VariableRateJump(rate, affect!)
    jump_prob = JumpProblem(prob, Direct(), jump; vr_aggregator = VR_FRM())
    sol = solve(jump_prob, Tsit5(); saveat = 0.5)
    times = range(0.0, 10.0; step = 0.5)
    @test issubset(times, sol.t)
end

# Test for u0 promotion https://github.com/SciML/JumpProcesses.jl/issues/275
let
    p = (λ = 2.0, μ = 1.5)

    deathrate(u, p, t) = p.μ * u[1]
    deathaffect!(integrator) = (integrator.u[1] -= 1; integrator.u[2] += 1)
    deathvrj = VariableRateJump(deathrate, deathaffect!)

    rate1(u, p, t) = p.λ * (sin(pi * t / 2) + 1)
    affect1!(integrator) = (integrator.u[1] += 1)
    vrj = VariableRateJump(rate1, affect1!)

    function f!(du, u, p, t)
        du .= 0
        nothing
    end
    u₀ = [0, 0]
    oprob = ODEProblem(f!, u₀, (0.0, 10.0), p)
    jprob = JumpProblem(oprob, Direct(), vrj, deathvrj; vr_aggregator = VR_FRM())
    sol = solve(jprob, Tsit5())
    @test eltype(sol.u) <: ExtendedJumpArray{Float64, 1, Vector{Float64}, Vector{Float64}}
    @test SciMLBase.plottable_indices(sol.u[1]) == 1:length(u₀)
end
