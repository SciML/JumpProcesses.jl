# Modified make_gpu_jump_data to handle arbitrary dependencies
function make_gpu_jump_data(agg::JumpProcesses.DirectJumpAggregation, prob::JumpProblem, backend)
    num_jumps = length(agg.rates)
    state_dim = length(prob.prob.u0)
    p = prob.prob.p
    t = prob.prob.tspan[1]
    rates = agg.rates
    affects = agg.affects!

    # Initialize arrays for affect increments
    affect_increments = zeros(Int64, num_jumps, state_dim)
    u_test = ones(Float64, state_dim)

    # Extract affect increments
    for k in 1:num_jumps
        u = copy(u_test)
        mock_integrator = (u=u, p=p, t=t)
        affects[k](mock_integrator)
        for i in 1:state_dim
            affect_increments[k, i] = Int64(u[i] - u_test[i])
        end
    end

    # Analyze rate dependencies
    rate_coeffs = zeros(Float64, num_jumps)
    depend_indices = Int64[]  # Flattened array of dependency indices
    depend_starts = zeros(Int64, num_jumps)  # Start index for each jump
    depend_counts = zeros(Int64, num_jumps)  # Number of dependencies per jump

    for k in 1:num_jumps
        rate_base = rates[k](u_test, p, t)
        deps = Int64[]
        for i in 1:state_dim
            u_perturbed = copy(u_test)
            u_perturbed[i] = 2.0
            rate_perturbed = rates[k](u_perturbed, p, t)
            delta_rate = rate_perturbed - rate_base
            if abs(delta_rate) > 1e-10
                push!(deps, i)
            end
        end

        depend_starts[k] = length(depend_indices) + 1
        depend_counts[k] = length(deps)
        append!(depend_indices, deps)

        if isempty(deps)
            # Constant rate
            rate_coeffs[k] = rate_base
        else
            # Assume polynomial rate: k * prod(u[i] for i in deps)
            u_prod = prod(u_test[i] for i in deps)
            rate_coeffs[k] = rate_base / u_prod
        end
    end

    # Adapt to GPU
    num_jumps = adapt(backend, num_jumps)
    rate_coeffs_gpu = adapt(backend, rate_coeffs)
    affect_increments_gpu = adapt(backend, affect_increments)
    depend_indices_gpu = adapt(backend, depend_indices)
    depend_starts_gpu = adapt(backend, depend_starts)
    depend_counts_gpu = adapt(backend, depend_counts)
    return (num_jumps, rate_coeffs_gpu, affect_increments_gpu, depend_indices_gpu, depend_starts_gpu, depend_counts_gpu)
end

# Modified vectorized_gillespie_direct
function vectorized_gillespie_direct(probs, prob::JumpProblem, alg::SSAStepper;
                                     backend, trajectories, seed, max_steps, rj_data)
    num_jumps, rate_coeffs, affect_increments, depend_indices, depend_starts, depend_counts = rj_data
    probs_data = [TrajectoryDataSSA(SA{eltype(p.prob.u0)}[p.prob.u0...], 
                                   p.prob.p, 
                                   p.prob.tspan[1], 
                                   p.prob.tspan[2]) for p in probs]
    probs_data_gpu = adapt(backend, probs_data)

    state_dim = length(first(probs_data).u0)

    ts = allocate(backend, Float64, (max_steps, trajectories))
    us = allocate(backend, Float64, (max_steps, state_dim, trajectories))
    current_u_buf = allocate(backend, Float64, (state_dim, trajectories))
    rate_cache_buf = allocate(backend, Float64, (num_jumps, trajectories))

    @kernel function init_buffers_kernel(@Const(probs_data), current_u_buf)
        i = @index(Global, Linear)
        if i <= size(current_u_buf, 2)
            u0 = probs_data[i].u0
            @inbounds for k in 1:length(u0)
                current_u_buf[k, i] = u0[k]
            end
        end
    end
    init_kernel = init_buffers_kernel(backend)
    init_event = init_kernel(probs_data_gpu, current_u_buf; ndrange=trajectories)
    synchronize(backend)

    seed_val = seed === nothing ? UInt64(12345) : UInt64(seed)
    kernel = gillespie_direct_kernel(backend)
    kernel_event = kernel(probs_data_gpu, num_jumps, rate_coeffs, affect_increments, 
                         depend_indices, depend_starts, depend_counts, 
                         us, ts, current_u_buf, rate_cache_buf, seed_val, max_steps;
                         ndrange=trajectories)
    synchronize(backend)

    return ts, us
end

# Modified Gillespie Direct kernel for arbitrary dependencies
@kernel function gillespie_direct_kernel(@Const(prob_data), @Const(num_jumps),
                                        @Const(rate_coeffs), @Const(affect_increments),
                                        @Const(depend_indices), @Const(depend_starts), @Const(depend_counts),
                                        us_out, ts_out, current_u_buf, rate_cache_buf, seed::UInt64, max_steps)
    i = @index(Global, Linear)
    if i <= size(current_u_buf, 2)
        current_u = view(current_u_buf, :, i)
        rate_cache = view(rate_cache_buf, :, i)

        prob_i = prob_data[i]
        u0 = prob_i.u0
        t_start = prob_i.t_start
        t_end = prob_i.t_end

        state_dim = length(u0)
        @inbounds for k in 1:state_dim
            current_u[k] = u0[k]
        end

        t = t_start
        step_idx = 1
        ts_view = view(ts_out, :, i)
        us_view = view(us_out, :, :, i)

        @inbounds for j in 1:max_steps
            ts_view[j] = NaN
            @inbounds for k in 1:state_dim
                us_view[j, k] = NaN
            end
        end

        ts_view[1] = t
        @inbounds for k in 1:state_dim
            us_view[1, k] = current_u[k]
        end

        while t < t_end && step_idx < max_steps
            total_rate = 0.0
            @inbounds for k in 1:num_jumps
                rate = rate_coeffs[k]
                start_idx = depend_starts[k]
                count = depend_counts[k]
                for d in 0:(count-1)
                    state_idx = depend_indices[start_idx + d]
                    rate *= current_u[state_idx]
                end
                rate_cache[k] = max(0.0, rate)
                total_rate += rate_cache[k]
            end

            if total_rate <= 0.0
                # Extend trajectory to t_end with constant state
                while t < t_end && step_idx < max_steps
                    step_idx += 1
                    t = min(t + 0.1, t_end)  # Match saveat interval
                    ts_view[step_idx] = t
                    @inbounds for k in 1:state_dim
                        us_view[step_idx, k] = current_u[k]
                    end
                end
                break
            end

            delta_t = exponential_rand(total_rate, seed + UInt64(i * max_steps + step_idx), i)
            next_t = t + delta_t

            r = total_rate * uniform_rand(seed + UInt64(i * max_steps + step_idx + 1), i)
            cum_rate = 0.0
            jump_idx = 0
            @inbounds for k in 1:num_jumps
                cum_rate += rate_cache[k]
                if r <= cum_rate
                    jump_idx = k
                    break
                end
            end

            if next_t <= t_end && jump_idx > 0 && step_idx < max_steps
                t = next_t
                @inbounds for j in 1:state_dim
                    current_u[j] = max(0.0, current_u[j] + affect_increments[jump_idx, j])  # Prevent negative states
                end
                step_idx += 1
                ts_view[step_idx] = t
                @inbounds for k in 1:state_dim
                    us_view[step_idx, k] = current_u[k]
                end
            else
                t = t_end
                # Ensure final state is recorded
                if step_idx < max_steps
                    step_idx += 1
                    ts_view[step_idx] = t
                    @inbounds for k in 1:state_dim
                        us_view[step_idx, k] = current_u[k]
                    end
                end
            end
        end
    end
end

# Modified SciMLBase.__solve with proper interpolation
function SciMLBase.__solve(
    ensembleprob::SciMLBase.AbstractEnsembleProblem,
    alg::SSAStepper,
    ensemblealg::EnsembleGPUKernel;
    trajectories,
    seed=nothing,
    saveat=0.1,
    save_everystep=true,
    save_start=true,
    save_end=true,
    kwargs...
)
    if trajectories == 1
        return SciMLBase.__solve(ensembleprob, alg, EnsembleSerial();
                                 trajectories=1, seed, saveat, kwargs...)
    end

    prob = ensembleprob.prob
    @assert isa(prob, JumpProblem) "Only JumpProblems supported"
    @assert isempty(prob.jump_callback.continuous_callbacks) "No continuous callbacks allowed"
    @assert prob.prob isa DiscreteProblem "SSAStepper only supports DiscreteProblems"

    backend = ensemblealg.backend === nothing ? CPU() : ensemblealg.backend
    probs = [remake(prob) for _ in 1:trajectories]

    rate_funcs = prob.jump_callback.discrete_callbacks[end].condition.rates
    u0 = prob.prob.u0
    p = prob.prob.p
    t0 = prob.prob.tspan[1]
    total_rate = sum(rate_func(u0, p, t0) for rate_func in rate_funcs)
    max_steps = Int(ceil(max(10000, prob.prob.tspan[2] * total_rate * 2)))
    @assert max_steps > 0 "max_steps must be positive"

    rj_data = make_gpu_jump_data(prob.jump_callback.discrete_callbacks[end].condition, prob, backend)
    rj_data_gpu = adapt(backend, rj_data)

    ts, us = vectorized_gillespie_direct(probs, prob, alg; backend, trajectories, seed, max_steps, rj_data=rj_data_gpu)

    _ts = Array(ts)
    _us = Array(us)

    time = @elapsed sol = [begin
        ts_view = @view _ts[:, i]
        us_view = @view _us[:, :, i]
        sol_idx = findlast(!isnan, ts_view)
        if sol_idx === nothing
            @error "No valid solution for trajectory $i" tspan=probs[i].prob.tspan ts=ts_view
            error("Batch solve failed")
        end
        @views ensembleprob.output_func(
            SciMLBase.build_solution(
                probs[i].prob,
                alg,
                ts_view[1:sol_idx],
                [SVector{length(us_view[1, :]), eltype(us_view[1, :])}(us_view[j, :]) for j in 1:sol_idx],
                k = nothing,
                stats = nothing,
                calculate_error = false,
                retcode = sol_idx < max_steps ? ReturnCode.Success : ReturnCode.Terminated
            ),
            i)[1]
    end for i in eachindex(probs)]

    return SciMLBase.EnsembleSolution(sol, time, true)
end

# Struct to hold trajectory-specific data
struct TrajectoryDataSSA{U <: StaticArray, P, T}
    u0::U
    p::P
    t_start::T
    t_end::T
end

# GPU-compatible random number generation
@inline function exponential_rand(lambda::T, seed::UInt64, idx::Int64) where T
    seed = (1103515245 * (seed ⊻ UInt64(idx)) + 12345) % 2^31
    u = Float64(seed) / 2^31
    return -log(u) / lambda
end

@inline function uniform_rand(seed::UInt64, idx::Int64)
    seed = (1103515245 * (seed ⊻ UInt64(idx)) + 12345) % 2^31
    return Float64(seed) / 2^31
end