function make_gpu_jump_data(agg::JumpProcesses.DirectJumpAggregation, prob::JumpProblem, backend)
    num_jumps = length(agg.rates)
    state_dim = length(prob.prob.u0)  # Get state dimension from DiscreteProblem
    p = prob.prob.p
    t = prob.prob.tspan[1]
    rates = agg.rates
    affects = agg.affects!

    # Initialize arrays
    rate_coeffs = zeros(Float64, num_jumps)
    affect_increments = zeros(Int64, num_jumps, state_dim)
    depend_idx = zeros(Int64, num_jumps)

    # Test point for evaluating rate functions
    u_test = ones(Float64, state_dim)

    # Extract rate coefficients and dependency indices
    for k in 1:num_jumps
        rate_base = rates[k](u_test, p, t)
        found_dep = false
        for i in 1:state_dim
            u_perturbed = copy(u_test)
            u_perturbed[i] = 2.0
            rate_perturbed = rates[k](u_perturbed, p, t)
            delta_rate = rate_perturbed - rate_base
            if abs(delta_rate) > 1e-10  # Detect significant dependence
                rate_coeffs[k] = delta_rate / (u_perturbed[i] - u_test[i])
                depend_idx[k] = i
                found_dep = true
                break
            end
        end
        if !found_dep
            rate_coeffs[k] = rate_base  # Constant rate (no state dependency)
            depend_idx[k] = 1  # Default to first state
        end
    end

    # Extract affect increments
    for k in 1:num_jumps
        u = copy(u_test)
        mock_integrator = (u=u, p=p, t=t)
        affects[k](mock_integrator)
        for i in 1:state_dim
            affect_increments[k, i] = Int64(u[i] - u_test[i])
        end
    end

    # Adapt to GPU
    num_jumps = adapt(backend, num_jumps)
    rate_coeffs_gpu = adapt(backend, rate_coeffs)
    affect_increments_gpu = adapt(backend, affect_increments)
    depend_idx_gpu = adapt(backend, depend_idx)
    return (num_jumps, rate_coeffs_gpu, affect_increments_gpu, depend_idx_gpu)
end

# Entry point for solving ensembles on GPU
function SciMLBase.__solve(
    ensembleprob::SciMLBase.AbstractEnsembleProblem,
    alg::SSAStepper,
    ensemblealg::EnsembleGPUKernel;
    trajectories,
    seed=nothing,
    saveat=nothing,
    save_everystep=true,
    save_start=true,
    save_end=true,
    max_steps=nothing,
    kwargs...
)
    if trajectories == 1
        return SciMLBase.__solve(ensembleprob, alg, EnsembleSerial();
                                 trajectories=1, seed, max_steps, kwargs...)
    end

    prob = ensembleprob.prob
    @assert isa(prob, JumpProblem) "Only JumpProblems supported"
    @assert isempty(prob.jump_callback.continuous_callbacks) "No continuous callbacks allowed"
    @assert prob.prob isa DiscreteProblem "SSAStepper only supports DiscreteProblems"

    # Select backend
    backend = ensemblealg.backend === nothing ? CPU() : ensemblealg.backend
    probs = [remake(prob) for _ in 1:trajectories]

    # Get aggregation and validate
    agg = prob.jump_callback.discrete_callbacks[end].condition
    @assert agg isa JumpProcesses.DirectJumpAggregation "Only DirectJumpAggregation is supported"

    # Prepare max_steps estimate
    rate_funcs = agg.rates
    u0 = prob.prob.u0
    p = prob.prob.p
    t0 = prob.prob.tspan[1]
    total_rate = sum(rate_func(u0, p, t0) for rate_func in rate_funcs)
    max_steps = max_steps === nothing ? Int(ceil(max(1000, prob.prob.tspan[2] * total_rate * 2))) : max_steps
    @assert max_steps > 0 "max_steps must be positive"

    rj_data = make_gpu_jump_data(agg, prob, backend)
    rj_data_gpu = adapt(backend, rj_data)

    ts, us = vectorized_gillespie_direct(probs, prob, alg; backend, trajectories, seed, max_steps, rj_data=rj_data_gpu)

    # Bring results back to CPU
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

# Main vectorized solver
function vectorized_gillespie_direct(probs, prob::JumpProblem, alg::SSAStepper;
                                    backend, trajectories, seed, max_steps, rj_data)
    num_jumps, rate_coeffs, affect_increments, depend_idx = rj_data  # Unpack the tuple
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
    kernel_event = kernel(probs_data_gpu, num_jumps, rate_coeffs, affect_increments, depend_idx, us, ts, current_u_buf, rate_cache_buf, seed_val, max_steps;
                         ndrange=trajectories)
    synchronize(backend)

    return ts, us
end

# Main Gillespie Direct kernel
@kernel function gillespie_direct_kernel(@Const(prob_data), @Const(num_jumps),
                                        @Const(rate_coeffs), @Const(affect_increments),
                                        @Const(depend_idx), us_out, ts_out, current_u_buf, rate_cache_buf, seed::UInt64, max_steps)
    i = @index(Global, Linear)
    if i <= size(current_u_buf, 2)
        current_u = view(current_u_buf, :, i)
        rate_cache = view(rate_cache_buf, :, i)

        prob_i = prob_data[i]
        u0 = prob_i.u0
        p = prob_i.p
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
                rate = rate_coeffs[k] * current_u[depend_idx[k]]
                rate_cache[k] = max(0.0, rate)
                total_rate += rate_cache[k]
            end

            if total_rate <= 0.0
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
                    current_u[j] += affect_increments[jump_idx, j]
                end
                step_idx += 1
                ts_view[step_idx] = t
                @inbounds for k in 1:state_dim
                    us_view[step_idx, k] = current_u[k]
                end
            else
                t = t_end
            end
        end
    end
end
