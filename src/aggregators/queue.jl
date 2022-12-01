"""
Queue method. This method handles variable intensity rates.
"""
mutable struct QueueMethodJumpAggregation{T, S, F1, F2, RNG, GR, PQ} <:
               AbstractSSAJumpAggregator
    next_jump::Int                    # the next jump to execute
    prev_jump::Int                    # the previous jump that was executed
    next_jump_time::T                 # the time of the next jump
    end_time::T                       # the time to stop a simulation
    cur_rates::Nothing                # not used
    sum_rate::Nothing                 # not used
    ma_jumps::S                       # not used
    rates::F1                         # vector of rate functions
    affects!::F2                      # vector of affect functions for VariableRateJumps
    save_positions::Tuple{Bool, Bool} # tuple for whether to save the jumps before and/or after event
    rng::RNG                          # random number generator
    dep_gr::GR                        # map from jumps to jumps depending on it
    pq::PQ                            # priority queue of next time
    lrates::F1                        # vector of rate lower bound functions
    urates::F1                        # vector of rate upper bound functions
    Ls::F1                            # vector of interval length functions
end

function QueueMethodJumpAggregation(nj::Int, njt::T, et::T, crs::Nothing, sr::Nothing,
                                    maj::S, rs::F1, affs!::F2, sps::Tuple{Bool, Bool},
                                    rng::RNG; u::U,
                                    dep_graph = nothing,
                                    lrates, urates, Ls) where {T, S, F1, F2, RNG, U}
    if dep_graph === nothing
        if (get_num_majumps(maj) == 0) || !isempty(rs)
            error("To use VariableRateJumps with the Queue Method algorithm a dependency graph between jumps must be supplied.")
        else
            dg = make_dependency_graph(length(u), maj)
        end
    end

    num_jumps = get_num_majumps(maj) + length(rs)

    # using a Set to ensure that edges are not duplicate
    dg = [Set{Int}(append!([], jumps, [var]))
          for (var, jumps) in enumerate(dep_graph)]
    dg = [sort!(collect(i)) for i in dg]

    if length(dg) != num_jumps
        error("Number of nodes in the dependency graph must be the same as the number of jumps.")
    end

    pq = MutableBinaryMinHeap{T}()
    QueueMethodJumpAggregation{T, S, F1, F2, RNG, typeof(dg),
                               typeof(pq)
                               }(nj, nj, njt,
                                 et,
                                 crs, sr, maj,
                                 rs,
                                 affs!, sps,
                                 rng,
                                 dg, pq,
                                 lrates, urates, Ls)
end

# creating the JumpAggregation structure (tuple-based variable jumps)
function aggregate(aggregator::QueueMethod, u, p, t, end_time, variable_jumps,
                   ma_jumps, save_positions, rng;
                   dep_graph = nothing,
                   kwargs...)
    AffectWrapper = FunctionWrappers.FunctionWrapper{Nothing, Tuple{Any}}
    RateWrapper = FunctionWrappers.FunctionWrapper{typeof(t),
                                                   Tuple{typeof(u), typeof(p), typeof(t)}}
    if (variable_jumps !== nothing) && !isempty(variable_jumps)
        affects! = [AffectWrapper((integrator) -> (c.affect!(integrator); nothing))
                    for c in variable_jumps]
        rates = [RateWrapper(c.rate) for c in variable_jumps]
        lrates = Any[RateWrapper(c.lrate) for c in variable_jumps]
        urates = Any[RateWrapper(c.urate) for c in variable_jumps]
        Ls = Any[RateWrapper(c.L) for c in variable_jumps]
    else
        affects! = Vector{AffectWrapper}()
        rates = Vector{RateWrapper}()
        lrates = Vector{RateWrapper}()
        urates = Vector{RateWrapper}()
        Ls = Vector{RateWrapper}()
    end
    cur_rates = nothing
    sum_rate = nothing
    next_jump = 0
    next_jump_time = typemax(t)
    QueueMethodJumpAggregation(next_jump, next_jump_time, end_time, cur_rates, sum_rate,
                               ma_jumps, rates, affects!, save_positions, rng;
                               u = u,
                               dep_graph = dep_graph,
                               lrates = lrates, urates = urates, Ls = Ls)
end

# set up a new simulation and calculate the first jump / jump time
function initialize!(p::QueueMethodJumpAggregation, integrator, u, params, t)
    p.end_time = integrator.sol.prob.tspan[2]
    fill_rates_and_get_times!(p, u, params, t)
    generate_jumps!(p, integrator, u, params, t)
    nothing
end

# execute one jump, changing the system state
function execute_jumps!(p::QueueMethodJumpAggregation, integrator, u, params, t)
    # execute jump
    u = update_state!(p, integrator, u, params, t)
    # update current jump rates and times
    update_dependent_rates!(p, u, params, t)
    nothing
end

# calculate the next jump / jump time
function generate_jumps!(p::QueueMethodJumpAggregation, integrator, u, params, t)
    p.next_jump_time, p.next_jump = top_with_handle(p.pq)
    nothing
end

@inline function update_state!(p::QueueMethodJumpAggregation, integrator, u, params, t)
    @unpack ma_jumps, next_jump = p
    num_majumps = get_num_majumps(ma_jumps)
    if next_jump <= num_majumps
        if u isa SVector
            integrator.u = executerx(u, next_jump, ma_jumps)
        else
            @inbounds executerx!(u, next_jump, ma_jumps)
        end
    else
        idx = next_jump - num_majumps
        @inbounds p.affects![idx](integrator)
    end
    p.prev_jump = next_jump
    return integrator.u
end

######################## SSA specific helper routines ########################
function update_dependent_rates!(p::QueueMethodJumpAggregation, u, params, t)
    @inbounds deps = p.dep_gr[p.next_jump]
    @unpack end_time, pq = p
    for (ix, i) in enumerate(deps)
        ti = next_time(p, u, params, t, i, end_time)
        update!(pq, i, ti)
    end
    nothing
end

function get_rates(p::QueueMethodJumpAggregation, i, u)
    ma_jumps = p.ma_jumps
    num_majumps = get_num_majumps(ma_jumps)
    if i <= num_majumps
        _rate = evalrxrate(u, i, ma_jumps)
        rate(u, p, t) = _rate
        lrate = rate
        urate = rate
        L(u, p, t) = typemax(t)
    else
        idx = i - num_majumps
        rate, lrate, urate, L = p.rates[idx], p.lrates[idx], p.urates[idx], p.Ls[idx]
    end
    return rate, lrate, urate, L
end

function next_time(p::QueueMethodJumpAggregation{T}, u, params, t, i, tstop::T) where {T}
    @unpack rng = p
    rate, lrate, urate, L = get_rates(p, i, u)
    while t < tstop
        _urate = urate(u, params, t)
        _L = L(u, params, t)
        s = _urate == zero(t) ? typemax(t) : randexp(rng) / _urate
        _t = t + s
        if s > _L
            t = t + _L
            continue
        end
        if _t >= tstop
            break
        end
        _lrate = lrate(u, params, t)
        if _lrate > _urate
            error("The lower bound should be lower than the upper bound rate for t = $(t) and i = $(i), but lower bound = $(_lrate) > upper bound = $(_urate)")
        elseif _lrate < _urate
            # when the lower and upper bound are the same, then v < 1 = _lrate / _urate = _rate / _urate
            v = rand(rng)
            # first inequality is less expensive and short-circuits the evaluation
            if (v > _lrate / _urate)
                _rate = rate(u, params, _t)
                if (v > _rate / _urate)
                    t = _t
                    continue
                end
            end
        end
        return _t
    end
    return typemax(t)
end

# reevaulate all rates, recalculate all jump times, and reinit the priority queue
function fill_rates_and_get_times!(p::QueueMethodJumpAggregation, u, params, t)
    @unpack end_time = p
    num_jumps = get_num_majumps(p.ma_jumps) + length(p.rates)
    jump_times = Vector{eltype(t)}(undef, num_jumps)
    @inbounds for i in 1:num_jumps
        jump_times[i] = next_time(p, u, params, t, i, end_time)
    end
    p.pq = MutableBinaryMinHeap(jump_times)
    nothing
end
