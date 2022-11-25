"""
Queue method. This method handles variable intensity rates.
"""
mutable struct QueueMethodJumpAggregation{T, S, F1, F2, RNG, INVDEPGR, DEPGR, PQ} <:
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
    dep_gr::DEPGR                     # map from jumps to jumps depending on it
    inv_dep_gr::INVDEPGR              # map from jumsp to jumps it depends on
    pq::PQ                            # priority queue of next time
    lrates::F1                        # vector of rate lower bound functions
    urates::F1                        # vector of rate upper bound functions
    Ls::F1                            # vector of interval length functions
end

function QueueMethodJumpAggregation(nj::Int, njt::T, et::T, crs::Nothing, sr::Nothing,
                                    maj::S, rs::F1, affs!::F2, sps::Tuple{Bool, Bool},
                                    rng::RNG; u::U, inv_dep_gr = nothing,
                                    dep_gr = nothing,
                                    lrates, urates, Ls) where {T, S, F1, F2, RNG, U}
    if inv_dep_gr === nothing && dep_gr === nothing
        if (get_num_majumps(maj) == 0) || !isempty(rs)
            error("To use VariableRateJumps with the Queue Method algorithm a dependency graph between jumps and/or its inverse must be supplied.")
        else
            dg = make_dependency_graph(length(u), maj)
            idg = dg
        end
    end

    num_jumps = get_num_majumps(maj) + length(rs)

    if dep_gr !== nothing
        # using a Set to ensure that edges are not duplicate
        dg = [Set{Int}(append!([], jumps, [var]))
              for (var, jumps) in enumerate(dep_gr)]
    end

    if inv_dep_gr !== nothing
        # using a Set to ensure that edges are not duplicate
        idg = [Set{Int}(append!([], vars, [jump]))
               for (jump, vars) in enumerate(inv_dep_gr)]
    end

    if dep_gr === nothing
        dg = idg
    end

    if inv_dep_gr === nothing
        idg = dg
    end

    if length(dg) != num_jumps
        error("Number of nodes in the dependency graph must be the same as the number of jumps.")
    end

    if length(idg) != num_jumps
        error("Number of nodes in the inverse dependency graph must be the same as the number of jumps.")
    end

    pq = PriorityQueue{Int, T}()
    QueueMethodJumpAggregation{T, S, F1, F2, RNG, typeof(dg), typeof(idg),
                               typeof(pq)
                               }(nj, nj, njt,
                                 et,
                                 crs, sr, maj,
                                 rs,
                                 affs!, sps,
                                 rng,
                                 dg, idg, pq,
                                 lrates, urates, Ls)
end

# creating the JumpAggregation structure (tuple-based variable jumps)
function aggregate(aggregator::QueueMethod, u, p, t, end_time, variable_jumps,
                   ma_jumps, save_positions, rng;
                   dep_gr = nothing, inv_dep_gr = nothing,
                   kwargs...)
    # TODO: FunctionWrapper slows down big problems
    # keeping the problematic implementation here for future reference
    # U, P, T, H = typeof(u), typeof(p), typeof(t), typeof(t)
    # G = Vector{Vector{Int}}
    # if (variable_jumps !== nothing) && !isempty(variable_jumps)
    #     marks = [c.mark for c in variable_jumps]
    #     if eltype(marks) === Nothing
    #         MarkWrapper = Nothing
    #         AffectWrapper = FunctionWrappers.FunctionWrapper{Nothing, Tuple{Any}}
    #         RateWrapper = FunctionWrappers.FunctionWrapper{T,
    #                                                        Tuple{U, P, T, G,
    #                                                              Vector{Vector{H}}}}
    #     else
    #         MarkWrapper = FunctionWrappers.FunctionWrapper{U,
    #                                                        Tuple{U, P, T, G,
    #                                                              Vector{Vector{H}}}}
    #         AffectWrapper = FunctionWrappers.FunctionWrapper{Nothing,
    #                                                          Tuple{Any, U}}
    #         H = Tuple{T, U}
    #         RateWrapper = FunctionWrappers.FunctionWrapper{T,
    #                                                        Tuple{U, P, T, G,
    #                                                              Vector{Vector{H}}}}
    #     end
    # else
    #     MarkWrapper = Nothing
    #     AffectWrapper = FunctionWrappers.FunctionWrapper{Nothing, Tuple{Int, Any}}
    #     RateWrapper = FunctionWrappers.FunctionWrapper{T,
    #                                                    Tuple{U, P, T, G, Vector{Vector{H}}}}
    # end

    # if (variable_jumps !== nothing) && !isempty(variable_jumps)
    #     if eltype(marks) === Nothing
    #         marks = nothing
    #         affects! = [AffectWrapper((integrator) -> (c.affect!(integrator); nothing))
    #                     for c in variable_jumps]
    #     else
    #         marks = [convert(MarkWrapper, m) for m in marks]
    #         affects! = [AffectWrapper((integrator, m) -> (c.affect!(integrator, m); nothing))
    #                     for c in variable_jumps]
    #     end

    #     history = [jump.history for jump in variable_jumps]
    #     if eltype(history) === Nothing
    #         history = [Vector{H}() for _ in variable_jumps]
    #     else
    #         history = [convert(H, h) for h in history]
    #     end

    #     rates = [RateWrapper(c.rate) for c in variable_jumps]
    #     lrates = [RateWrapper(c.lrate) for c in variable_jumps]
    #     urates = [RateWrapper(c.urate) for c in variable_jumps]
    #     Ls = [RateWrapper(c.L) for c in variable_jumps]
    # else
    #     marks = nothing
    #     history = Vector{Vector{H}}()
    #     affects! = Vector{AffectWrapper}()
    #     rates = Vector{RateWrapper}()
    #     lrates = Vector{RateWrapper}()
    #     urates = Vector{RateWrapper}()
    #     Ls = Vector{RateWrapper}()
    # end
    if (variable_jumps !== nothing) && !isempty(variable_jumps)
        AffectWrapper = FunctionWrappers.FunctionWrapper{Nothing, Tuple{Any}}
        affects! = [AffectWrapper((integrator) -> (c.affect!(integrator); nothing))
                    for c in variable_jumps]
        rates = Any[c.rate for c in variable_jumps]
        lrates = Any[c.lrate for c in variable_jumps]
        urates = Any[c.urate for c in variable_jumps]
        Ls = Any[c.L for c in variable_jumps]
    else
        AffectWrapper = FunctionWrappers.FunctionWrapper{Nothing, Tuple{Any}}
        affects! = Vector{AffectWrapper}()
        rates = Vector{Any}()
        lrates = Vector{Any}()
        urates = Vector{Any}()
        Ls = Vector{Any}()
    end
    cur_rates = nothing
    sum_rate = nothing
    next_jump = 0
    next_jump_time = typemax(typeof(t))
    QueueMethodJumpAggregation(next_jump, next_jump_time, end_time, cur_rates, sum_rate,
                               ma_jumps, rates, affects!, save_positions, rng;
                               u = u,
                               dep_gr = dep_gr,
                               inv_dep_gr = inv_dep_gr,
                               lrates = lrates, urates = urates, Ls = Ls, kwargs...)
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
    p.next_jump, p.next_jump_time = peek(p.pq)
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
        @inbounds p.affects![next_jump](integrator)
    end
    p.prev_jump = next_jump
    return integrator.u
end

######################## SSA specific helper routines ########################
function update_dependent_rates!(p::QueueMethodJumpAggregation, u, params, t)
    @unpack next_jump, rates, pq = p
    @inbounds vars = collect(p.dep_gr[next_jump])
    shuffle!(vars)
    for i in vars
        pq[i] = typemax(t)
    end
    while !isempty(vars)
        i = pop!(vars)
        ti = next_time(p, i, u, params, t)
        pq[i] = ti
    end
    nothing
end

function get_rates(p::QueueMethodJumpAggregation, i, u, t)
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

function next_time(p::QueueMethodJumpAggregation, i, u, params, t)
    @unpack end_time, rng, pq = p
    rate, lrate, urate, L = get_rates(p, i, u, t)
    inv_dep_gr = p.inv_dep_gr[i]
    tstop = end_time
    for j in inv_dep_gr
        if j == i
            continue
        end
        if pq[j] < end_time
            tstop = pq[j]
            break
        end
    end
    while t < tstop
        _urate = urate(u, params, t)
        _L = L(u, params, t)
        s = randexp(rng) / _urate
        if s > _L
            t = t + _L
            continue
        end
        _lrate = lrate(u, params, t)
        if _lrate > _urate
            error("The lower bound should be lower than the upper bound rate for t = $(t) and i = $(i), but lower bound = $(_lrate) > upper bound = $(_urate)")
        end
        v = rand(rng)
        # first inequality is less expensive and short-circuits the evaluation
        if (v > _lrate / _urate)
            _rate = rate(u, params, t + s)
            if (v > _rate / _urate)
                t = t + s
                continue
            end
        end
        t = t + s
        return t
    end
    return typemax(t)
end

# reevaulate all rates, recalculate all jump times, and reinit the priority queue
function fill_rates_and_get_times!(p::QueueMethodJumpAggregation, u, params, t)
    @unpack rates, end_time = p
    num_jumps = get_num_majumps(p.ma_jumps) + length(rates)
    pqdata = Vector{Tuple{Int, eltype(t)}}(undef, num_jumps)
    @inbounds for i in 1:num_jumps
        pqdata[i] = (i, typemax(t))
    end
    pq = PriorityQueue(pqdata)
    p.pq = pq
    @inbounds for i in shuffle(1:num_jumps)
        @inbounds ti = next_time(p, i, u, params, t)
        pq[i] = ti
    end
    nothing
end
