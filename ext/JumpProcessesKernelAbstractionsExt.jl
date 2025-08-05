module JumpProcessesKernelAbstractionsExt

using JumpProcesses, SciMLBase
using KernelAbstractions, Adapt
using StaticArrays
using Random
using LogExpFunctions: log1pmx
using SpecialFunctions: loggamma

function SciMLBase.__solve(ensembleprob::SciMLBase.AbstractEnsembleProblem,
        alg::SimpleTauLeaping,
        ensemblealg::EnsembleGPUKernel;
        trajectories,
        seed = nothing,
        dt = error("dt is required for SimpleTauLeaping."),
        kwargs...)
    if trajectories == 1
        return SciMLBase.__solve(ensembleprob, alg, EnsembleSerial(); trajectories = 1,
            seed, dt, kwargs...)
    end

    ensemblealg.backend === nothing ? backend = CPU() :
    backend = ensemblealg.backend

    jump_prob = ensembleprob.prob

    # boilerplate from SimpleTauLeaping method
    @assert isempty(jump_prob.jump_callback.continuous_callbacks) # still needs to be a regular jump
    @assert isempty(jump_prob.jump_callback.discrete_callbacks)
    prob = jump_prob.prob

    probs = [remake(jump_prob) for _ in 1:trajectories]

    # Run vectorized solve
    ts,
    us = vectorized_solve(
        probs, jump_prob, SimpleTauLeaping(); backend, trajectories, seed, dt)

    # Convert to CPU for inspection
    _ts = Array(ts)
    _us = Array(us)

    time = @elapsed sol = [begin
                               ts = @view _ts[:, i]
                               us = @view _us[:, :, i]
                               sol_idx = findlast(x -> x != probs[i].prob.tspan[1], ts)
                               if sol_idx === nothing
                                   @error "No solution found" tspan=probs[i].tspan[1] ts
                                   error("Batch solve failed")
                               end
                               @views ensembleprob.output_func(
                                   SciMLBase.build_solution(probs[i].prob,
                                       alg,
                                       ts[1:sol_idx],
                                       [us[j, :] for j in 1:sol_idx],
                                       k = nothing,
                                       stats = nothing,
                                       calculate_error = false,
                                       retcode = sol_idx !=
                                                 length(ts) ?
                                                 ReturnCode.Terminated :
                                                 ReturnCode.Success),
                                   i)[1]
                           end
                           for i in eachindex(probs)]
    return SciMLBase.EnsembleSolution(sol, time, true)
end

# Define an immutable struct to hold trajectory-specific data
struct TrajectoryData{U <: StaticArray, P, T}
    u0::U
    p::P
    tspan::Tuple{T, T}
end

# Define an immutable struct to hold common jump data
struct JumpData{R, C}
    rate::R
    c::C
    numjumps::Int
end

# SimpleTauLeaping kernel
@kernel function simple_tau_leaping_kernel(
        @Const(probs_data), _us, _ts, dt, @Const(rj_data),
        current_u_buf, rate_cache_buf, counts_buf, local_dc_buf,
        seed::UInt64)
    i = @index(Global, Linear)

    # Get thread-local buffers
    @inbounds begin
        current_u = view(current_u_buf, :, i)
        rate_cache = view(rate_cache_buf, :, i)
        counts = view(counts_buf, :, i)
        local_dc = view(local_dc_buf, :, i)
    end

    # Get problem data
    @inbounds prob_data = probs_data[i]
    u0 = prob_data.u0
    p = prob_data.p
    tspan = prob_data.tspan

    # Extract jump data
    rate = rj_data.rate
    num_jumps = rj_data.numjumps
    c = rj_data.c

    # Initialize current_u from u0
    @inbounds for k in 1:length(u0)
        current_u[k] = u0[k]
    end

    n = Int((tspan[2] - tspan[1]) / dt) + 1
    state_dim = length(u0)

    # Get input/output arrays
    ts_view = @inbounds view(_ts, :, i)
    us_view = @inbounds view(_us,:,:,i)

    # Initialize first time step and state
    @inbounds ts_view[1] = tspan[1]
    @inbounds for k in 1:state_dim
        us_view[1, k] = current_u[k]
    end

    # Main loop
    for j in 2:n
        tprev = tspan[1] + (j-2) * dt

        # Compute rates and scale by dt
        rate(rate_cache, current_u, p, tprev)
        rate_cache .*= dt

        # Poisson sampling
        @inbounds for k in 1:num_jumps
            counts[k] = pois_rand(PassthroughRNG(), rate_cache[k])
        end

        # Apply changes
        c(local_dc, current_u, p, tprev, counts, nothing)
        current_u .+= local_dc

        # Store results
        @inbounds for k in 1:state_dim
            us_view[j, k] = current_u[k]
        end
        @inbounds ts_view[j] = tspan[1] + (j-1) * dt
    end
end

# Vectorized solve function
function vectorized_solve(probs, prob::JumpProblem, alg::SimpleTauLeaping;
        backend, trajectories, seed, dt, kwargs...)
    # Extract common jump data
    rj = prob.regular_jump
    rj_data = JumpData(rj.rate, rj.c, rj.numjumps)

    # Extract trajectory-specific data without static typing
    probs_data = [TrajectoryData(SA{eltype(p.prob.u0)}[p.prob.u0...], p.prob.p, p.prob.tspan)
                  for p in probs]

    # Adapt to GPU
    probs_data_gpu = adapt(backend, probs_data)
    rj_data_gpu = adapt(backend, rj_data)

    # Extract problem parameters
    state_dim = length(first(probs_data).u0)
    tspan = prob.prob.tspan
    dt = Float64(dt)
    n_steps = Int((tspan[2] - tspan[1]) / dt) + 1
    n_trajectories = length(probs)
    num_jumps = rj_data.numjumps

    # Validate dimensions
    @assert state_dim > 0 "Dimension of state must be positive"
    @assert num_jumps >= 0 "Number of jumps must be positive"

    # Allocate time and state arrays
    ts = allocate(backend, eltype(prob.prob.tspan), (n_steps, n_trajectories))
    us = allocate(backend, eltype(prob.prob.u0), (n_steps, state_dim, n_trajectories))

    # Pre-allocate thread-local buffers
    current_u_buf = allocate(backend, eltype(prob.prob.u0), (state_dim, n_trajectories))
    rate_cache_buf = allocate(backend, eltype(prob.prob.u0), (num_jumps, n_trajectories))
    counts_buf = allocate(backend, eltype(prob.prob.u0), (num_jumps, n_trajectories))
    local_dc_buf = allocate(backend, eltype(prob.prob.u0), (state_dim, n_trajectories))

    # Initialize current_u_buf with u0 values
    @kernel function init_buffers_kernel(@Const(probs_data), current_u_buf)
        i = @index(Global, Linear)
        @inbounds u0 = probs_data[i].u0
        @inbounds for k in 1:length(u0)
            current_u_buf[k, i] = u0[k]
        end
    end
    init_kernel = init_buffers_kernel(backend)
    init_event = init_kernel(probs_data_gpu, current_u_buf; ndrange = n_trajectories)
    KernelAbstractions.synchronize(backend)

    # Seed for Poisson sampling
    seed = seed === nothing ? UInt64(12345) : UInt64(seed);

    # Launch main kernel
    kernel = simple_tau_leaping_kernel(backend)
    main_event = kernel(probs_data_gpu, us, ts, dt, rj_data_gpu,
        current_u_buf, rate_cache_buf, counts_buf, local_dc_buf, seed;
        ndrange = n_trajectories)
    KernelAbstractions.synchronize(backend)

    return ts, us
end

# GPU-compatible Poisson sampling PassthroughRNG
struct PassthroughRNG <: AbstractRNG end

rand(rng::PassthroughRNG) = Random.rand()
randexp(rng::PassthroughRNG) = Random.randexp()
randn(rng::PassthroughRNG) = Random.randn()

count_rand(λ) = count_rand(Random.GLOBAL_RNG, λ)
function count_rand(rng::AbstractRNG, λ)
    n = 0
    c = randexp(rng)
    while c < λ
        n += 1
        c += randexp(rng)
    end
    return n
end

# Algorithm from:
#
#   J.H. Ahrens, U. Dieter (1982)
#   "Computer Generation of Poisson Deviates from Modified Normal Distributions"
#   ACM Transactions on Mathematical Software, 8(2):163-179
#
#   For μ sufficiently large, (i.e. >= 10.0)
#
ad_rand(λ) = ad_rand(Random.GLOBAL_RNG, λ)
function ad_rand(rng::AbstractRNG, λ)
    s = sqrt(λ)
    d = 6 * λ^2
    L = floor(Int, λ - 1.1484)
    # Step N
    G = λ + s * randn(rng)

    if G >= 0
        K = floor(Int, G)
        # Step I
        if K >= L
            return K
        end

        # Step S
        U = rand(rng)
        if d * U >= (λ - K)^3
            return K
        end

        # Step P
        px, py, fx, fy = procf(λ, K, s)

        # Step Q
        if fy * (1 - U) <= py * exp(px - fx)
            return K
        end
    end

    while true
        # Step E
        E = randexp(rng)
        U = 2 * rand(rng) - 1
        T = 1.8 + copysign(E, U)
        if T <= -0.6744
            continue
        end

        K = floor(Int, λ + s * T)
        px, py, fx, fy = procf(λ, K, s)
        c = 0.1069 / λ

        # Step H
        @fastmath if c * abs(U) <= py * exp(px + E) - fy * exp(fx + E)
            return K
        end
    end
end

# Procedure F
function procf(λ, K::Int, s::Float64)
    # can be pre-computed, but does not seem to affect performance
    INV_SQRT_2PI = inv(sqrt(2pi))
    ω = INV_SQRT_2PI / s
    b1 = inv(24) / λ
    b2 = 0.3 * b1 * b1
    c3 = inv(7) * b1 * b2
    c2 = b2 - 15 * c3
    c1 = b1 - 6 * b2 + 45 * c3
    c0 = 1 - b1 + 3 * b2 - 15 * c3

    if K < 10
        px = -λ
        log_py = K * log(λ) - loggamma(K + 1) # log(K!) via loggamma
        py = exp(log_py)
    else
        δ = inv(12) / K
        δ -= 4.8 * δ^3
        V = (λ - K) / K
        px = K * log1pmx(V) - δ # avoids need for table
        py = INV_SQRT_2PI / sqrt(K)
    end
    X = (K - λ + 0.5) / s
    X2 = X^2
    fx = X2 / -2 # missing negation in pseudo-algorithm, but appears in fortran code.
    fy = ω * (((c3 * X2 + c2) * X2 + c1) * X2 + c0)
    return px, py, fx, fy
end

"""
```julia
pois_rand(λ)
pois_rand(rng::AbstractRNG, λ)
```

Generates Poisson(λ) distributed random numbers using a fast polyalgorithm.

## Examples

```julia
# Simple Poisson random
pois_rand(λ)

# Using another RNG
using RandomNumbers
rng = Xorshifts.Xoroshiro128Plus()
pois_rand(rng, λ)

# Simple Poisson random on GPU
pois_rand(PassthroughRNG(), λ)
```
"""
pois_rand(λ) = pois_rand(Random.GLOBAL_RNG, λ)
pois_rand(rng::AbstractRNG, λ) = λ < 6 ? count_rand(rng, λ) : ad_rand(rng, λ)

end
