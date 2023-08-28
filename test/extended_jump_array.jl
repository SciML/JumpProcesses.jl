using Test, JumpProcesses, DiffEqBase
using StableRNGs

rng = StableRNG(123)

# Check that the new broadcast norm gives the same result as the old one
rand_array = ExtendedJumpArray{Float64, 1, Vector{Float64}, Vector{Float64}}(rand(rng, 5),
                                                                             rand(rng, 2))
old_norm = Base.FastMath.sqrt_fast(DiffEqBase.UNITLESS_ABS2(rand_array) /
                                   max(DiffEqBase.recursive_length(rand_array), 1))
new_norm = DiffEqBase.ODE_DEFAULT_NORM(rand_array, 0.0)
@test old_norm ≈ new_norm

# Check for an ExtendedJumpArray where the types differ (Float64/Int64)
rand_array = ExtendedJumpArray{Float64, 1, Vector{Float64}, Vector{Int64}}(rand(rng, 5),
                                                                           rand(rng, 1:1000,
                                                                                2))
old_norm = Base.FastMath.sqrt_fast(DiffEqBase.UNITLESS_ABS2(rand_array) /
                                   max(DiffEqBase.recursive_length(rand_array), 1))
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
@test_throws DimensionMismatch bc_mismatch.+bc_eja_1

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
using FastBroadcast
@.. bc_out = 3.14 * bc_eja_1 + 2.7 * bc_eja_2
@test bc_out ≈ 3.14 * bc_eja_1 + 2.7 * bc_eja_2
