"""
Queue method. This method handles variable intensity rates.
"""
mutable struct QueueMethodJumpAggregation{T, S, F1, F2, F3, F4, RNG, DEPGR, PQ, H} <:
               AbstractSSAJumpAggregator
    next_jump::Int                    # the next jump to execute
    prev_jump::Int                    # the previous jump that was executed
    next_jump_time::T                 # the time of the next jump
    end_time::T                       # the time to stop a simulation
    cur_rates::F1                     # not used
    sum_rate::T                       # not used
    ma_jumps::S                       # not used
    rates::F2                         # vector of rate functions
    affects!::F3                      # vector of affect functions for VariableRateJumps
    save_positions::Tuple{Bool, Bool} # tuple for whether to save the jumps before and/or after event
    rng::RNG                          # random number generator
    dep_gr::DEPGR                     # dependency graph
    pq::PQ                            # priority queue of next time
    marks::F4                         # vector of mark functions for VariableRateJumps
    h::H                              # history of jumps
    save_history::Bool                # whether to save event history after solving
    lrates::F2                        # vector of rate lower bound functions
    urates::F2                        # vector of rate upper bound functions
    Ls::F2                            # vector of interval length functions
end

function QueueMethodJumpAggregation(nj::Int, njt::T, et::T, crs::F1, sr::T,
                                    maj::S, rs::F2, affs!::F3, sps::Tuple{Bool, Bool},
                                    rng::RNG; dep_graph = nothing, marks::F4,
                                    history::H,
                                    save_history = false,
                                    lrates, urates, Ls) where {T, S, F1, F2, F3, F4, RNG, H}
    if get_num_majumps(maj) > 0
        error("Mass-action jumps are not supported with the Queue Method.")
    end

    if dep_graph === nothing
        gr = [Int[] for _ in length(rs)]
    elseif length(dep_graph) != length(rs)
        error("Dependency graph must have same length as the number of jumps.")
    else
        gr = [sort(Int[d for d in deps]) for deps in dep_graph]
    end

    pq = PriorityQueue{Int, T}()
    QueueMethodJumpAggregation{T, S, F1, F2, F3, F4, RNG, typeof(gr), typeof(pq),
                               typeof(history)}(nj, nj, njt,
                                                et,
                                                crs, sr, maj,
                                                rs,
                                                affs!, sps,
                                                rng,
                                                gr, pq, marks, history, save_history,
                                                lrates, urates, Ls)
end

# creating the JumpAggregation structure (tuple-based variable jumps)
function aggregate(aggregator::QueueMethod, u, p, t, end_time, variable_jumps,
                   ma_jumps, save_positions, rng; dep_graph = nothing, save_history = false,
                   kwargs...)
    # TODO: FunctionWrapper slows down big problems
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
    U, T, H = typeof(u), typeof(t), typeof(t)
    if (variable_jumps !== nothing) && !isempty(variable_jumps)
        marks = [c.mark for c in variable_jumps]
        if eltype(marks) === Nothing
            marks = nothing
            AffectWrapper = FunctionWrappers.FunctionWrapper{Nothing, Tuple{Any}}
            affects! = [AffectWrapper((integrator) -> (c.affect!(integrator); nothing))
                        for c in variable_jumps]
        else
            AffectWrapper = FunctionWrappers.FunctionWrapper{Nothing,
                                                             Tuple{Any, eltype(u)}}
            affects! = [AffectWrapper((integrator, m) -> (c.affect!(integrator, m); nothing))
                        for c in variable_jumps]
            H = Tuple{T, eltype(u)}
        end
        history = [jump.history for jump in variable_jumps]
        if eltype(history) === Nothing
            history = [Vector{H}() for _ in variable_jumps]
        else
            history = [convert(Vector{H}, h) for h in history]
        end
        rates = Any[c.rate for c in variable_jumps]
        lrates = Any[c.lrate for c in variable_jumps]
        urates = Any[c.urate for c in variable_jumps]
        Ls = Any[c.L for c in variable_jumps]
    else
        marks = nothing
        AffectWrapper = FunctionWrappers.FunctionWrapper{Nothing, Tuple{Any}}
        affects! = Vector{AffectWrapper}()
        history = Vector{Vector{H}}()
        rates = Vector{Any}()
        lrates = Vector{Any}()
        urates = Vector{Any}()
        Ls = Vector{Any}()
    end
    cur_rates = nothing
    sum_rate = zero(typeof(t))
    next_jump = 0
    next_jump_time = typemax(typeof(t))
    QueueMethodJumpAggregation(next_jump, next_jump_time, end_time, cur_rates, sum_rate,
                               ma_jumps, rates, affects!, save_positions, rng;
                               dep_graph = dep_graph, marks = marks,
                               history = history, save_history = save_history,
                               lrates = lrates, urates = urates, Ls = Ls, kwargs...)
end

# set up a new simulation and calculate the first jump / jump time
function initialize!(p::QueueMethodJumpAggregation, integrator, u, params, t)
    p.end_time = integrator.sol.prob.tspan[2]
    reset_history!(p, integrator)
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
    # println("GENERATE_JUMPS! ", peek(p.pq), " p.next_jump_time ", p.next_jump_time,
    #         " p.next_jump ",
    #         p.next_jump)
    if (p.next_jump_time > p.end_time) && !p.save_history
        # throw the history away once simulation is over
        reset_history!(p, integrator)
    end
    nothing
end

@inline function update_state!(p::QueueMethodJumpAggregation, integrator, u, params, t)
    @unpack next_jump, dep_gr, marks, h = p
    if marks === nothing
        # println("UPDATE_STATE!", " p.next_jump ", p.next_jump, " p.next_jump_time ",
        # p.next_jump_time, " integrator ", integrator)
        push!(p.h[p.next_jump], p.next_jump_time)
        @inbounds p.affects![next_jump](integrator)
    else
        m = marks[next_jump](u, params, t, dep_gr, h)
        push!(p.h[p.next_jump], (p.next_jump_time, m))
        @inbounds p.affects![next_jump](integrator, m)
    end
    p.prev_jump = next_jump
    return integrator.u
end

######################## SSA specific helper routines ########################
function update_dependent_rates!(p::QueueMethodJumpAggregation, u, params, t)
    @unpack next_jump, end_time, rates, pq = p
    @inbounds deps = copy(p.dep_gr[next_jump])

    Base.insert!(deps, searchsortedfirst(deps, next_jump), next_jump)

    while !isempty(deps)
        j = pop!(deps)
        @inbounds tj = next_time(p, j, u, params, t, end_time, deps)
        pq[j] = tj
    end

    nothing
end

function next_time(p::QueueMethodJumpAggregation, i, u, params, t, tstop, ignore)
    @unpack pq, dep_gr, h = p
    # println("NEXT_TIME i ", i)
    # we only need to determine whether a jump will take place before any of the dependents
    for (j, _t) in pairs(pq)
        # println("NEXT_TIME pq LOOP1 j ", j, " _t ", _t)
        @inbounds if j != i && !insorted(j, ignore) && insorted(j, dep_gr[i]) && _t < tstop
            # println("NEXT_TIME pq LOOP2 j ", j, " tstop ", _t)
            tstop = _t
            break
        end
    end
    return next_time(p, i, u, params, t, tstop)
end

function next_time(p::QueueMethodJumpAggregation, i, u, params, t, tstop)
    @unpack rng, pq, dep_gr, h = p
    rate, lrate, urate, L = p.rates[i], p.lrates[i], p.urates[i], p.Ls[i]
    while t < tstop
        # println("NEXT_TIME i ", i, " u ", u, " params ", params, " t ", t, " dep_gr ",
        # dep_gr,
        # " h ", h)
        _urate = urate(u, params, t, dep_gr, h)
        _L = L(u, params, t, dep_gr, h)
        s = randexp(rng) / _urate
        if s > _L
            t = t + _L
            continue
        end
        _lrate = lrate(u, params, t, dep_gr, h)
        if _lrate > _urate
            error("The lower bound should be lower than the upper bound rate for t = $(t) and i = $(i), but lower bound = $(_lrate) > upper bound = $(_urate)")
        end
        v = rand(rng)
        # first inequality is less expensive and short-circuits the evaluation
        if (v > _lrate / _urate)
            _rate = rate(u, params, t + s, dep_gr, h)
            if (v > _rate / _urate)
                t = t + s
                continue
            end
        end
        t = t + s
        # println("NEXT TIME RETURN t ", t, " rate ", rate(u, params, t, dep_gr, h))
        return t
    end
    return typemax(t)
end

# reevaulate all rates, recalculate all jump times, and reinit the priority queue
function fill_rates_and_get_times!(p::QueueMethodJumpAggregation, u, params, t)
    @unpack rates, end_time = p
    pqdata = Vector{Tuple{Int, eltype(t)}}(undef, length(rates))
    @inbounds for i in 1:length(rates)
        @inbounds ti = next_time(p, i, u, params, t, end_time)
        pqdata[i] = (i, ti)
    end
    p.pq = PriorityQueue(pqdata)
    nothing
end

function reset_history!(p::QueueMethodJumpAggregation, integrator)
    start_time = integrator.sol.prob.tspan[1]
    @unpack h = p
    @inbounds for i in 1:length(h)
        hi = h[i]
        ix = 0
        if eltype(h) <: Tuple
            while ((ix + 1) <= length(hi)) && hi[ix + 1][1] <= start_time
                ix += 1
            end
        else
            while ((ix + 1) <= length(hi)) && hi[ix + 1] <= start_time
                ix += 1
            end
        end
        h[i] = ix == 0 ? eltype(h)[] : hi[1:ix]
    end
    nothing
end
