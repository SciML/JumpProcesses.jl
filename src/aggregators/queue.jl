"""
Queue method. This method handles conditional intensity rates.
"""
mutable struct QueueMethodJumpAggregation{T, S, F1, F2, F3, RNG, DEPGR, PQ} <:
               AbstractSSAJumpAggregator
    next_jump::Int # the next jump to execute
    prev_jump::Int # the previous jump that was executed
    next_jump_time::T # the time of the next jump
    end_time::T # the time to stop a simulation
    cur_rates::F1 # vector of current propensity values
    sum_rate::T # sum of current propensity values
    ma_jumps::S # any MassActionJumps for the system (scalar form)
    rates::F2 # vector of rate functions for ConditionalRateJumps
    affects!::F3 # vector of affect functions for ConditionalRateJumps
    save_positions::Tuple{Bool, Bool} # tuple for whether to save the jumps before and/or after event
    rng::RNG # random number generator
    dep_gr::DEPGR # dependency graph
    pq::PQ # priority queue of next time
    h::Array{Array{T}, 1} # history of jumps
end

function QueueMethodJumpAggregation(nj::Int, njt::T, et::T, crs::F1, sr::T,
                                    maj::S, rs::F2, affs!::F3, sps::Tuple{Bool, Bool},
                                    rng::RNG; dep_graph = nothing,
                                    kwargs...) where {T, S, F1, F2, F3, RNG}
    if get_num_majumps(maj) > 0
        error("Mass-action jumps are not supported with the Queue Method.")
    end

    if dep_graph === nothing
        if !isempty(rs)
            error("To use ConstantRateJumps with Queue Method algorithm a dependency graph must be supplied.")
        end
    else
        dg = dep_graph
        # make sure each jump depends on itself
        add_self_dependencies!(dg)
    end

    pq = MutableBinaryMinHeap{T}()
    h = Array{Array{T}, 1}(undef, length(rs))
    QueueMethodJumpAggregation{T, S, F1, F2, F3, RNG, typeof(dg), typeof(pq)}(nj, nj, njt,
                                                                              et,
                                                                              crs, sr, maj,
                                                                              rs,
                                                                              affs!, sps,
                                                                              rng,
                                                                              dg, pq, h)
end

# creating the JumpAggregation structure (tuple-based constant jumps)
function aggregate(aggregator::QueueMethod, u, p, t, end_time, conditional_jumps,
                   ma_jumps, save_positions, rng; dep_graph, kwargs...)
    # TODO: Fix FunctionWrapper as it unstable with more than 2 processes
    # U, P, T, G = typeof(u), typeof(p), typeof(t), typeof(dep_graph)
    # RateWrapper = FunctionWrappers.FunctionWrapper{T, Tuple{U, P, T}}
    # ConditionalRateWrapper = FunctionWrappers.FunctionWrapper{
    #                                                           Tuple{RateWrapper, T, T, T},
    #                                                           Tuple{Int, G,
    #                                                                 Array{Array{T, 1}}, U,
    #                                                                 P, T}
    #                                                           }
    # AffectWrapper = FunctionWrappers.FunctionWrapper{Nothing, Tuple{Any}}
    # if (conditional_jumps !== nothing) && !isempty(conditional_jumps)
    #     rates = [ConditionalRateWrapper(c.rate) for c in conditional_jumps]
    #     affects! = [AffectWrapper(x -> (c.affect!(x); nothing)) for c in conditional_jumps]
    # else
    #     rates = Vector{ConditionalRateWrapper}()
    #     affects! = Vector{AffectWrapper}()
    # end
    # cur_rates = Vector{RateWrapper}(undef, length(conditional_jumps))
    AffectWrapper = FunctionWrappers.FunctionWrapper{Nothing, Tuple{Any}}
    if (conditional_jumps !== nothing) && !isempty(conditional_jumps)
        rates = [c.rate for c in conditional_jumps]
        affects! = [AffectWrapper(x -> (c.affect!(x); nothing)) for c in conditional_jumps]
    else
        rates = Vector{Any}()
        affects! = Vector{AffectWrapper}()
    end
    cur_rates = Array{Any}(undef, length(conditional_jumps))
    sum_rate = zero(typeof(t))
    next_jump = 0
    next_jump_time = typemax(typeof(t))
    QueueMethodJumpAggregation(next_jump, next_jump_time, end_time, cur_rates, sum_rate,
                               ma_jumps, rates, affects!, save_positions, rng; dep_graph,
                               kwargs...)
end

# set up a new simulation and calculate the first jump / jump time
function initialize!(p::QueueMethodJumpAggregation, integrator, u, params, t)
    p.end_time = integrator.sol.prob.tspan[2]
    p.h = [eltype(p.h)(undef, 0) for _ in 1:length(p.h)]
    fill_rates_and_get_times!(p, u, params, t)
    generate_jumps!(p, integrator, u, params, t)
    nothing
end

# execute one jump, changing the system state
function execute_jumps!(p::QueueMethodJumpAggregation, integrator, u, params, t)
    # execute jump
    u = update_state!(p, integrator, u)
    # update current jump rates and times
    update_dependent_rates!(p, u, params, t)
    nothing
end

# calculate the next jump / jump time
function generate_jumps!(p::QueueMethodJumpAggregation, integrator, u, params, t)
    p.next_jump_time, p.next_jump = top_with_handle(p.pq)
    if p.next_jump_time < p.end_time
        push!(p.h[p.next_jump], p.next_jump_time)
    else
        # throw the history away once simulation is over
        p.h = Array{eltype(p.h)}(undef, length(p.h))
    end
    nothing
end

######################## SSA specific helper routines ########################
function update_dependent_rates!(p::QueueMethodJumpAggregation, u, params, t)
    @inbounds dep_rxs = p.dep_gr[p.next_jump]
    @unpack cur_rates, rates = p

    @inbounds for rx in dep_rxs
        @inbounds trx, cur_rates[rx] = next_time(p, rx, rates[rx], u, params, t)
        update!(p.pq, rx, trx)
    end

    nothing
end

function next_time(p::QueueMethodJumpAggregation, rx, rate, u, params, t)
    @unpack end_time, rng, dep_gr, h = p
    cur_rate = nothing
    while t < end_time
        # println("HERE rx ", rx, " rate ", rate)
        # println("HERE rx ", rx, " h ", h, " dep_gr ", dep_gr, " params ", params, " t ", t, " u ", u)
        cur_rate, lrate, urate, L = rate(rx, dep_gr, h, u, params, t)
        # println("HERE 1")
        if lrate > urate
            error("The lower bound should be lower than the upper bound rate for t = $(t) and rx = $(rx), but lower bound = $(lrate) > upper bound = $(urate)")
        end
        # println("HERE 2")
        s = randexp(rng) / urate
        if s > L
            t = t + L
            continue
        end
        # println("HERE 3")
        v = rand(rng)
        # the first inequality is less expensive and short-circuits the evaluation
        if (v > lrate / urate) && (v > cur_rate(u, params, t + s) / urate)
            t = t + s
            continue
        end
        # println("HERE 4")
        t = t + s
        return t, cur_rate
    end
    return typemax(t), cur_rate
end

# reevaulate all rates, recalculate all jump times, and reinit the priority queue
function fill_rates_and_get_times!(p::QueueMethodJumpAggregation, u, params, t)
    @unpack cur_rates, rates = p
    pqdata = Vector{eltype(t)}(undef, length(rates))
    @inbounds for (rx, rate) in enumerate(rates)
        @inbounds trx, cur_rates[rx] = next_time(p, rx, rate, u, params, t)
        pqdata[rx] = trx
    end
    p.pq = MutableBinaryMinHeap(pqdata)
    nothing
end
