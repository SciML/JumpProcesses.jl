using Test, JumpProcesses, DiffEqBase
using StableRNGs

rng = StableRNG(123)

# Check that the new broadcast norm gives the same result as the old one
rand_array = ExtendedJumpArray{Float64, 1, Vector{Float64}, Vector{Float64}}(rand(rng, 5), rand(rng, 2))
old_norm = Base.FastMath.sqrt_fast(DiffEqBase.UNITLESS_ABS2(rand_array) / max(DiffEqBase.recursive_length(rand_array), 1))
new_norm = DiffEqBase.ODE_DEFAULT_NORM(rand_array, 0.0)
@test old_norm ≈ new_norm

# Check for an ExtendedJumpArray where the types differ (Float64/Int64)
rand_array = ExtendedJumpArray{Float64, 1, Vector{Float64}, Vector{Int64}}(rand(rng, 5), rand(rng, 1:1000, 2))
old_norm = Base.FastMath.sqrt_fast(DiffEqBase.UNITLESS_ABS2(rand_array) / max(DiffEqBase.recursive_length(rand_array), 1))
new_norm = DiffEqBase.ODE_DEFAULT_NORM(rand_array, 0.0)
@test old_norm ≈ new_norm

# Check that we no longer allocate. Run inside function so @allocated works properly
norm_check_alloc(jump_array, t) = @allocated DiffEqBase.ODE_DEFAULT_NORM(jump_array, t)
norm_check_alloc(rand_array, 0.0);
@test 0 == norm_check_alloc(rand_array, 0.0)