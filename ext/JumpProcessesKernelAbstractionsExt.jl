module JumpProcessesKernelAbstractionsExt

using JumpProcesses, SciMLBase
using KernelAbstractions, Adapt
using StaticArrays
using PoissonRandom, Random

# Kernels draw from the backend's own RNG; a host RNG object cannot be used
# inside device code. CPU kernels use Julia's task-local generator, which can be
# seeded here. Device generators must be seeded with their backend's API before
# calling solve (for example, CUDA.seed! for CUDA).
function seed_kernel_backend!(backend, seed, rng, kwargs)
    rng === nothing || throw(ArgumentError(
        "EnsembleGPUKernel does not support a host `rng` with multiple trajectories. " *
        "Use `seed` with the CPU backend or seed the device RNG through its backend API."))
    haskey(kwargs, :rng_func) && throw(ArgumentError(
        "EnsembleGPUKernel does not support `rng_func` with multiple trajectories. " *
        "Kernels draw from the backend's own RNG."))
    if seed !== nothing
        backend isa CPU || throw(ArgumentError(
            "EnsembleGPUKernel only supports the `seed` keyword with the CPU backend. " *
            "Seed the device RNG through its backend API before calling solve."))
        Random.seed!(seed)
    end
    return nothing
end

# Keywords stored on the problem wrapped by a `JumpProblem`, e.g. `DiscreteProblem(...;
# alias_jump = true)`.
wrapped_problem_kwargs(jump_prob::JumpProblem) = wrapped_problem_kwargs(jump_prob.prob)
wrapped_problem_kwargs(prob) = hasproperty(prob, :kwargs) ? prob.kwargs : (;)

# Kernels always build device-owned state, so the removed `alias_jump` keyword and
# `alias` controls are rejected rather than silently ignored, whether passed to `solve`
# or stored on the wrapped problem. These paths bypass SciMLBase's keyword validation, so
# the never-supported plural spelling is rejected here too, passed or stored.
function reject_kernel_alias_kwargs(jump_prob, kwargs)
    stored = wrapped_problem_kwargs(jump_prob)
    (haskey(kwargs, :alias_jump) || haskey(stored, :alias_jump)) &&
        throw(ArgumentError(JumpProcesses.ALIAS_JUMP_REMOVED_MSG))
    (haskey(kwargs, :alias_jumps) || haskey(stored, :alias_jumps)) && throw(ArgumentError(
        "`alias_jumps` is not a keyword argument; it is a field of some alias " *
        "specifiers. EnsembleGPUKernel always builds device-owned jump state."))
    (get(kwargs, :alias, nothing) === nothing &&
     get(stored, :alias, nothing) === nothing) ||
        throw(ArgumentError(
            "EnsembleGPUKernel does not support the `alias` keyword, whether passed to " *
            "`solve` or stored on the problem, with multiple trajectories, since kernels " *
            "always build device-owned state."))
    return nothing
end

include("ssa_stepper.jl")
include("explicit_tau_leaping.jl")

function SciMLBase.__solve(ensembleprob::SciMLBase.AbstractEnsembleProblem,
        alg::SimpleTauLeaping,
        ensemblealg::EnsembleGPUKernel;
        trajectories,
        seed = nothing,
        rng = nothing,
        dt = error("dt is required for SimpleTauLeaping."),
        kwargs...)
    if trajectories == 1
        return SciMLBase.__solve(ensembleprob, alg, EnsembleSerial(); trajectories = 1,
            seed, rng, dt, kwargs...)
    end

    ensemblealg.backend === nothing ? backend = CPU() :
    backend = ensemblealg.backend
    reject_kernel_alias_kwargs(ensembleprob.prob, kwargs)
    seed_kernel_backend!(backend, seed, rng, kwargs)

    jump_prob = ensembleprob.prob

    # Validate that this is a PureLeaping JumpProblem
    JumpProcesses.validate_pure_leaping_inputs(jump_prob, alg) ||
        error("SimpleTauLeaping requires a PureLeaping JumpProblem with a MassActionJump or a RegularJump.")

    # Mass-action rate constants are uploaded once from the template jump. A
    # RegularJump rate closure still receives each trajectory's `p`, but a
    # parameter-mapped MassActionJump cannot refresh those rates per trajectory.
    if jump_prob.regular_jump === nothing &&
            JumpProcesses.using_params(jump_prob.massaction_jump) &&
            ensembleprob.prob_func !== SciMLBase.DEFAULT_PROB_FUNC
        error("EnsembleGPUKernel with SimpleTauLeaping does not support MassActionJump \
               rate constants that depend on parameters (`param_idxs` / `param_mapper`) \
               together with a custom `prob_func`. Those rates are taken once from the \
               template problem and ignore per-trajectory `p`. Use a RegularJump whose \
               rate reads `p`, or a CPU ensemble algorithm such as EnsembleSerial.")
    end

    template_tspan = jump_prob.prob.tspan
    template_state_dim = length(jump_prob.prob.u0)
    template_massaction = jump_prob.regular_jump === nothing ?
        deepcopy(JumpProcesses.massaction_data(jump_prob.massaction_jump)) : nothing
    probs = map(1:trajectories) do i
        ctx = SciMLBase.EnsembleContext(i, 1, 0, nothing, nothing, nothing)
        _prob = ensembleprob.safetycopy ? deepcopy(jump_prob) : jump_prob
        trajectory = ensembleprob.prob_func(_prob, ctx)
        length(trajectory.prob.u0) == template_state_dim || throw(
            ArgumentError(
                "EnsembleGPUKernel with SimpleTauLeaping requires prob_func to preserve the template state dimension."
            )
        )
        trajectory.prob.tspan == template_tspan || throw(
            ArgumentError(
                "EnsembleGPUKernel with SimpleTauLeaping requires prob_func to preserve the template tspan."
            )
        )
        JumpProcesses.validate_pure_leaping_inputs(trajectory, alg) ||
            throw(ArgumentError("prob_func must return a PureLeaping JumpProblem with a MassActionJump or a RegularJump."))
        (trajectory.regular_jump === nothing) == (template_massaction !== nothing) ||
            throw(ArgumentError("prob_func must preserve the template jump representation."))
        if template_massaction !== nothing
            isequal(JumpProcesses.massaction_data(trajectory.massaction_jump), template_massaction) ||
                throw(
                ArgumentError(
                    "EnsembleGPUKernel with SimpleTauLeaping requires prob_func to preserve MassActionJump rate constants and stoichiometry."
                )
            )
        end
        trajectory
    end

    # Run vectorized solve
    ts,
    us = vectorized_solve(
        probs, jump_prob, SimpleTauLeaping(); backend, trajectories, dt)

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

Adapt.@adapt_structure JumpData

gpu_num_jumps(jump::JumpData) = jump.numjumps
leaping_rates!(out, jump::JumpData, u, p, t) = jump.rate(out, u, p, t)
leaping_change!(du, jump::JumpData, u, p, t, counts) = jump.c(du, u, p, t, counts, nothing)

function leaping_rates!(out, jump::GPUMassActionJump, u, p, t)
    for j in eachindex(out)
        out[j] = gpu_evalrxrate(u, j, jump, eltype(out))
    end
    return nothing
end

function leaping_change!(du, jump::GPUMassActionJump, u, p, t, counts)
    fill!(du, zero(eltype(du)))
    for j in eachindex(counts)
        for k in jump.ns_offsets[j]:(jump.ns_offsets[j + 1] - 1)
            du[jump.ns_species[k]] += jump.ns_coeffs[k] * counts[j]
        end
    end
    return nothing
end

# SimpleTauLeaping kernel
@kernel function simple_tau_leaping_kernel(
        @Const(probs_data), _us, _ts, dt, @Const(rj_data),
        current_u_buf, rate_cache_buf, counts_buf, local_dc_buf)
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
    num_jumps = gpu_num_jumps(rj_data)

    # Initialize current_u from u0
    @inbounds for k in 1:length(u0)
        current_u[k] = u0[k]
    end

    n = Int((tspan[2] - tspan[1]) / dt) + 1
    state_dim = length(u0)

    # Get input/output arrays
    ts_view = @inbounds view(_ts, :, i)
    us_view = @inbounds view(_us, :, :, i)

    # Initialize first time step and state
    @inbounds ts_view[1] = tspan[1]
    @inbounds for k in 1:state_dim
        us_view[1, k] = current_u[k]
    end

    # Main loop
    for j in 2:n
        tprev = tspan[1] + (j-2) * dt

        # Compute rates and scale by dt
        leaping_rates!(rate_cache, rj_data, current_u, p, tprev)
        rate_cache .*= dt

        # Poisson sampling
        @inbounds for k in 1:num_jumps
            counts[k] = pois_rand(PoissonRandom.PassthroughRNG(), rate_cache[k])
        end

        # Apply changes
        leaping_change!(local_dc, rj_data, current_u, p, tprev, counts)
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
        backend, trajectories, dt, kwargs...)
    # Extract common jump data
    rj = prob.regular_jump
    rj_data = if rj === nothing
        GPUMassActionJump(prob.massaction_jump, prob.prob.p, backend, float(eltype(prob.prob.tspan)))
    else
        JumpData(rj.rate, rj.c, rj.numjumps)
    end

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
    num_jumps = gpu_num_jumps(rj_data)

    # Validate dimensions
    @assert state_dim > 0 "Dimension of state must be positive"
    @assert num_jumps >= 0 "Number of jumps must be positive"

    # Allocate time and state arrays
    ts = allocate(backend, eltype(prob.prob.tspan), (n_steps, n_trajectories))
    us = allocate(backend, eltype(prob.prob.u0), (n_steps, state_dim, n_trajectories))

    # Pre-allocate thread-local buffers
    current_u_buf = allocate(backend, eltype(prob.prob.u0), (state_dim, n_trajectories))
    rate_cache_buf = allocate(backend, float(eltype(prob.prob.u0)), (num_jumps, n_trajectories))
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

    # Launch main kernel
    kernel = simple_tau_leaping_kernel(backend)
    main_event = kernel(probs_data_gpu, us, ts, dt, rj_data_gpu,
        current_u_buf, rate_cache_buf, counts_buf, local_dc_buf;
        ndrange = n_trajectories)
    KernelAbstractions.synchronize(backend)

    return ts, us
end

end
