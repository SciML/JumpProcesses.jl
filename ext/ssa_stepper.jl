# Define a GPU-compatible jump data structure
struct GPUJumpData{RF, AF}
    num_jumps::Int
    rates::RF
    affects::AF
end

# Helper to convert DirectJumpAggregation into GPUJumpData
function make_gpu_jump_data(agg::JumpProcesses.DirectJumpAggregation)
    rates = agg.rates
    affects = agg.affects!
    return GPUJumpData(length(rates), rates, affects)
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
                                 trajectories=1, seed, saveat, save_everystep, save_start, save_end, kwargs...)
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
    max_steps = max_steps === nothing ? Int(ceil(max(1000, prob.prob.tspan[2] * total_rate * 2) + length(saveat isa Number ? collect(prob.prob.tspan[1]:saveat:prob.prob.tspan[2]) : saveat))) : max_steps
    @assert max_steps > 0 "max_steps must be positive"

    # Build GPU jump data
    rj_data = make_gpu_jump_data(agg)
    rj_data_gpu = adapt(backend, GPUJumpData(rj_data.num_jumps, rj_data.rates, rj_data.affects))

    # Run vectorized Gillespie Direct SSA
    ts, us = vectorized_gillespie_direct(probs, prob, alg; backend, trajectories, seed, saveat, save_everystep, save_start, save_end, max_steps, rj_data=rj_data_gpu)

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
    saveat::Vector{T}
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
                                    backend, trajectories, seed, saveat, save_everystep, save_start, save_end, max_steps, rj_data)
    # Prepare saveat
    _saveat = saveat isa Number ? collect(prob.prob.tspan[1]:saveat:prob.prob.tspan[2]) : saveat
    _saveat = save_start && _saveat !== nothing && !isempty(_saveat) && _saveat[1] != prob.prob.tspan[1] ?
              vcat(prob.prob.tspan[1], _saveat) : _saveat
    _saveat = save_end && _saveat !== nothing && !isempty(_saveat) && _saveat[end] != prob.prob.tspan[2] ?
              vcat(_saveat, prob.prob.tspan[2]) : _saveat
    _saveat = _saveat === nothing ? Float64[] : _saveat

    # Convert to static arrays
    probs_data = [TrajectoryDataSSA(SA{eltype(p.prob.u0)}[p.prob.u0...], 
        p.prob.p, 
        p.prob.tspan[1],  # t_start
        p.prob.tspan[2],  # t_end
        _saveat) for p in probs]
    probs_data_gpu = adapt(backend, probs_data)

    state_dim = length(first(probs_data).u0)
    num_jumps = rj_data.num_jumps

    # Allocate buffers
    ts = allocate(backend, Float64, (max_steps, trajectories))
    us = allocate(backend, Float64, (max_steps, state_dim, trajectories))
    current_u_buf = allocate(backend, Float64, (state_dim, trajectories))
    rate_cache_buf = allocate(backend, Float64, (num_jumps, trajectories))

    # Initialize current_u_buf with u0
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
    kernel_event = kernel(probs_data_gpu, rj_data, us, ts, current_u_buf, rate_cache_buf, seed_val, max_steps;
                         ndrange=trajectories)
    synchronize(backend)

    return ts, us
end

# Main Gillespie Direct kernel
@kernel function gillespie_direct_kernel(@Const(prob_data), @Const(rj_data),
                                         us_out, ts_out, current_u_buf, rate_cache_buf, seed::UInt64, max_steps)
    i = @index(Global, Linear)
    if i <= size(current_u_buf, 2)
        current_u = view(current_u_buf, :, i)
        rate_cache = view(rate_cache_buf, :, i)

        prob_i = prob_data[i]
        u0 = prob_i.u0
        p = prob_i.p
        t_start = prob_i.t_start
        t_end = prob_i.t_end
        saveat = prob_i.saveat

        state_dim = length(u0)
        @inbounds for k in 1:state_dim
            current_u[k] = u0[k]
        end

        t = t_start
        step_idx = 1
        saveat_idx = 1
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
            @inbounds for k in 1:rj_data.num_jumps
                rate = rj_data.rates[k](current_u, p, t)
                rate_cache[k] = max(0.0, rate)
                total_rate += rate_cache[k]
            end

            if total_rate <= 0.0
                if !isempty(saveat)
                    while saveat_idx <= length(saveat) && step_idx < max_steps && saveat[saveat_idx] <= t_end
                        step_idx += 1
                        ts_view[step_idx] = saveat[saveat_idx]
                        @inbounds for k in 1:state_dim
                            us_view[step_idx, k] = current_u[k]
                        end
                        saveat_idx += 1
                    end
                end
                break
            end

            delta_t = exponential_rand(total_rate, seed + UInt64(i * max_steps + step_idx), i)
            next_t = t + delta_t

            if !isempty(saveat)
                while saveat_idx <= length(saveat) && saveat[saveat_idx] <= next_t && step_idx < max_steps
                    step_idx += 1
                    ts_view[step_idx] = saveat[saveat_idx]
                    @inbounds for k in 1:state_dim
                        us_view[step_idx, k] = current_u[k]
                    end
                    saveat_idx += 1
                end
            end

            r = total_rate * uniform_rand(seed + UInt64(i * max_steps + step_idx + 1), i)
            cum_rate = 0.0
            jump_idx = 0
            @inbounds for k in 1:rj_data.num_jumps
                cum_rate += rate_cache[k]
                if r <= cum_rate
                    jump_idx = k
                    break
                end
            end

            if next_t <= t_end && jump_idx > 0 && step_idx < max_steps
                t = next_t
                mock_integrator = (u=current_u, p=p, t=t)
                rj_data.affects[jump_idx](mock_integrator)
                step_idx += 1
                ts_view[step_idx] = t
                @inbounds for k in 1:state_dim
                    us_view[step_idx, k] = current_u[k]
                end
            else
                t = t_end
            end
        end

        while saveat_idx <= length(saveat) && step_idx < max_steps
            step_idx += 1
            ts_view[step_idx] = saveat[saveat_idx]
            @inbounds for k in 1:state_dim
                us_view[step_idx, k] = current_u[k]
            end
            saveat_idx += 1
        end
    end
end