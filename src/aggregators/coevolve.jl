"""
Queue method. This method handles variable intensity rates.
"""
mutable struct CoevolveJumpAggregation{T, S, F1, F2, RNG, GR, PQ} <:
               AbstractSSAJumpAggregator
    next_jump::Int                    # the next jump to execute
    prev_jump::Int                    # the previous jump that was executed
    next_jump_time::T                 # the time of the next jump
    end_time::T                       # the time to stop a simulation
    cur_rates::Vector{T}              # the last computed upper bound for each rate
    sum_rate::Nothing                 # not used
    ma_jumps::S                       # MassActionJumps
    rates::F1                         # vector of rate functions
    affects!::F2                      # vector of affect functions for VariableRateJumps
    save_positions::Tuple{Bool, Bool} # tuple for whether to save the jumps before and/or after event
    rng::RNG                          # random number generator
    dep_gr::GR                        # map from jumps to jumps depending on it
    pq::PQ                            # priority queue of next time
    lrates::F1                        # vector of rate lower bound functions
    urates::F1                        # vector of rate upper bound functions
    rateintervals::F1                 # vector of interval length functions
end

function CoevolveJumpAggregation(nj::Int, njt::T, et::T, crs::Vector{T}, sr::Nothing,
                                 maj::S, rs::F1, affs!::F2, sps::Tuple{Bool, Bool},
                                 rng::RNG; u::U, dep_graph = nothing, lrates, urates,
                                 rateintervals) where {T, S, F1, F2, RNG, U}
    if dep_graph === nothing
        if (get_num_majumps(maj) == 0) || !isempty(rs)
            error("To use Coevolve a dependency graph between jumps must be supplied.")
        else
            dg = make_dependency_graph(length(u), maj)
        end
    else
        # using a Set to ensure that edges are not duplicate
        dgsets = [Set{Int}(append!(Int[], jumps, [var]))
                  for (var, jumps) in enumerate(dep_graph)]
        dg = [sort!(collect(i)) for i in dgsets]
    end

    num_jumps = get_num_majumps(maj) + length(urates)

    if length(dg) != num_jumps
        error("Number of nodes in the dependency graph must be the same as the number of jumps.")
    end

    pq = MutableBinaryMinHeap{T}()
    CoevolveJumpAggregation{T, S, F1, F2, RNG, typeof(dg),
                            typeof(pq)}(nj, nj, njt, et, crs, sr, maj, rs, affs!, sps, rng,
                                        dg, pq, lrates, urates, rateintervals)
end

# creating the JumpAggregation structure (tuple-based variable jumps)
function aggregate(aggregator::Coevolve, u, p, t, end_time, constant_jumps,
                   ma_jumps, save_positions, rng; dep_graph = nothing,
                   variable_jumps = nothing, kwargs...)
    AffectWrapper = FunctionWrappers.FunctionWrapper{Nothing, Tuple{Any}}
    RateWrapper = FunctionWrappers.FunctionWrapper{typeof(t),
                                                   Tuple{typeof(u), typeof(p), typeof(t)}}
    affects! = Vector{AffectWrapper}()
    rates = Vector{RateWrapper}()
    lrates = Vector{RateWrapper}()
    urates = Vector{RateWrapper}()
    rateintervals = Vector{RateWrapper}()

    if (constant_jumps !== nothing) && !isempty(constant_jumps)
        append!(affects!,
                [AffectWrapper((integrator) -> (j.affect!(integrator); nothing))
                 for j in constant_jumps])
        append!(urates, [RateWrapper(j.rate) for j in constant_jumps])
    end

    if (variable_jumps !== nothing) && !isempty(variable_jumps)
        append!(affects!,
                [AffectWrapper((integrator) -> (j.affect!(integrator); nothing))
                 for j in variable_jumps])
        append!(rates, [RateWrapper(j.rate) for j in variable_jumps])
        append!(lrates, [RateWrapper(j.lrate) for j in variable_jumps])
        append!(urates, [RateWrapper(j.urate) for j in variable_jumps])
        append!(rateintervals, [RateWrapper(j.rateinterval) for j in variable_jumps])
    end

    num_jumps = get_num_majumps(ma_jumps) + length(urates)
    cur_rates = Vector{typeof(t)}(undef, num_jumps)
    sum_rate = nothing
    next_jump = 0
    next_jump_time = typemax(t)
    CoevolveJumpAggregation(next_jump, next_jump_time, end_time, cur_rates, sum_rate,
                            ma_jumps, rates, affects!, save_positions, rng;
                            u, dep_graph, lrates, urates, rateintervals)
end

# set up a new simulation and calculate the first jump / jump time
function initialize!(p::CoevolveJumpAggregation, integrator, u, params, t)
    p.end_time = integrator.sol.prob.tspan[2]
    fill_rates_and_get_times!(p, u, params, t)
    generate_jumps!(p, integrator, u, params, t)
    nothing
end

# execute one jump, changing the system state
function execute_jumps!(p::CoevolveJumpAggregation, integrator, u, params, t)
    # execute jump
    u = update_state!(p, integrator, u)
    # update current jump rates and times
    update_dependent_rates!(p, u, params, t)
    nothing
end

# calculate the next jump / jump time
function generate_jumps!(p::CoevolveJumpAggregation, integrator, u, params, t)
    p.next_jump_time, p.next_jump = top_with_handle(p.pq)
    nothing
end

######################## SSA specific helper routines ########################
function update_dependent_rates!(p::CoevolveJumpAggregation, u, params, t)
    @inbounds deps = p.dep_gr[p.next_jump]
    @unpack cur_rates, end_time, pq = p
    for (ix, i) in enumerate(deps)
        ti, last_urate_i = next_time(p, u, params, t, i, end_time)
        update!(pq, i, ti)
        cur_rates[i] = last_urate_i
    end
    nothing
end

@inline function get_ma_urate(p::CoevolveJumpAggregation, i, u, params, t)
    return evalrxrate(u, i, p.ma_jumps)
end

@inline function get_urate(p::CoevolveJumpAggregation, uidx, u, params, t)
    @inbounds return p.urates[uidx](u, params, t)
end

@inline function get_rateinterval(p::CoevolveJumpAggregation, lidx, u, params, t)
    @inbounds return p.rateintervals[lidx](u, params, t)
end

@inline function get_lrate(p::CoevolveJumpAggregation, lidx, u, params, t)
    @inbounds return p.lrates[lidx](u, params, t)
end

@inline function get_rate(p::CoevolveJumpAggregation, lidx, u, params, t)
    @inbounds return p.rates[lidx](u, params, t)
end

function next_time(p::CoevolveJumpAggregation{T}, u, params, t, i, tstop::T) where {T}
    @unpack rng = p
    num_majumps = get_num_majumps(p.ma_jumps)
    num_cjumps = length(p.urates) - length(p.rates)
    uidx = i - num_majumps
    lidx = i - num_majumps - num_cjumps
    urate = uidx > 0 ? get_urate(p, uidx, u, params, t) :
            get_ma_urate(p, i, u, params, t)
    last_urate = p.cur_rates[i]
    if i != p.next_jump && last_urate > zero(t)
        s = urate == zero(t) ? typemax(t) : last_urate / urate * (p.pq[i] - t)
    else
        s = urate == zero(t) ? typemax(t) : randexp(rng) / urate
    end
    _t = t + s
    if lidx > 0
        while t < tstop
            rateinterval = get_rateinterval(p, lidx, u, params, t)
            if s > rateinterval
                t = t + rateinterval
                urate = get_urate(p, uidx, u, params, t)
                s = urate == zero(t) ? typemax(t) : randexp(rng) / urate
                _t = t + s
                continue
            end
            if _t >= tstop
                break
            end
            lrate = p.urates[uidx] === p.lrates[lidx] ? urate :
                    get_lrate(p, lidx, u, params, t)
            if lrate > urate
                error("The lower bound should be lower than the upper bound rate for t = $(t) and i = $(i), but lower bound = $(lrate) > upper bound = $(urate)")
            elseif lrate < urate
                # when the lower and upper bound are the same, then v < 1 = lrate / urate = urate / urate
                v = rand(rng)
                # first inequality is less expensive and short-circuits the evaluation
                if (v > lrate / urate)
                    if (v > get_rate(p, lidx, u, params, _t) / urate)
                        t = _t
                        urate = get_urate(p, uidx, u, params, t)
                        s = urate == zero(t) ? typemax(t) : randexp(rng) / urate
                        _t = t + s
                        continue
                    end
                end
            end
            break
        end
    end
    return _t, urate
end

# reevaulate all rates, recalculate all jump times, and reinit the priority queue
function fill_rates_and_get_times!(p::CoevolveJumpAggregation, u, params, t)
    @unpack end_time = p
    num_jumps = get_num_majumps(p.ma_jumps) + length(p.urates)
    p.cur_rates = zeros(typeof(t), num_jumps)
    jump_times = Vector{typeof(t)}(undef, num_jumps)
    @inbounds for i in 1:num_jumps
        jump_times[i], p.cur_rates[i] = next_time(p, u, params, t, i, end_time)
    end
    p.pq = MutableBinaryMinHeap(jump_times)
    nothing
end
