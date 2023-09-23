"""
Queue method. This method handles variable intensity rates.
"""
mutable struct CoevolveJumpAggregation{T, S, F1, F2, RNG, GR, PQ} <:
               AbstractSSAJumpAggregator{T, S, F1, F2, RNG}
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
    haslratevec::Vector{Bool}         # vector of whether an lrate was provided for this vrj
    cur_lrates::Vector{T}             # the last computed lower rate for each rate
end

function CoevolveJumpAggregation(nj::Int, njt::T, et::T, crs::Vector{T}, sr::Nothing,
    maj::S, rs::F1, affs!::F2, sps::Tuple{Bool, Bool},
    rng::RNG; u::U, dep_graph = nothing, lrates, urates,
    rateintervals, haslratevec,
    cur_lrates::Vector{T}) where {T, S, F1, F2, RNG, U}
    if dep_graph === nothing
        if (get_num_majumps(maj) == 0) || !isempty(urates)
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
    affecttype = F2 <: Tuple ? F2 : Any
    CoevolveJumpAggregation{T, S, F1, affecttype, RNG, typeof(dg),
        typeof(pq)}(nj, nj, njt, et, crs, sr, maj,
        rs, affs!, sps, rng, dg, pq,
        lrates, urates, rateintervals,
        haslratevec, cur_lrates)
end

# display
function num_constant_rate_jumps(aggregator::CoevolveJumpAggregation)
    length(aggregator.urates)
end

# condition for jump to occur
function (p::CoevolveJumpAggregation)(u, t, integrator)
    p.next_jump_time == t &&
        accept_next_jump!(p, integrator, integrator.u, integrator.p, integrator.t)
end

# executing jump at the next jump time
function (p::CoevolveJumpAggregation)(integrator::I) where {I <:
                                                            AbstractSSAIntegrator}
    if !accept_next_jump!(p, integrator, integrator.u, integrator.p, integrator.t)
        return nothing
    end
    affects! = p.affects!
    if affects! isa Vector{FunctionWrappers.FunctionWrapper{Nothing, Tuple{I}}}
        execute_jumps!(p, integrator, integrator.u, integrator.p, integrator.t, affects!)
    else
        error("Error, invalid affects! type. Expected a vector of function wrappers and got $(typeof(affects!))")
    end
    generate_jumps!(p, integrator, integrator.u, integrator.p, integrator.t)
    register_next_jump_time!(integrator, p, integrator.t)
    nothing
end

function (p::CoevolveJumpAggregation{
    T,
    S,
    F1,
    F2,
})(integrator::AbstractSSAIntegrator) where
    {T, S, F1, F2 <: Union{Tuple, Nothing}}
    if !accept_next_jump!(p, integrator, integrator.u, integrator.p, integrator.t)
        return nothing
    end
    execute_jumps!(p, integrator, integrator.u, integrator.p, integrator.t, p.affects!)
    generate_jumps!(p, integrator, integrator.u, integrator.p, integrator.t)
    register_next_jump_time!(integrator, p, integrator.t)
    nothing
end

# creating the JumpAggregation structure (tuple-based variable jumps)
function aggregate(aggregator::Coevolve, u, p, t, end_time, constant_jumps,
    ma_jumps, save_positions, rng; dep_graph = nothing,
    variable_jumps = nothing, kwargs...)
    RateWrapper = FunctionWrappers.FunctionWrapper{typeof(t),
        Tuple{typeof(u), typeof(p), typeof(t)}}

    ncrjs = (constant_jumps === nothing) ? 0 : length(constant_jumps)
    nvrjs = (variable_jumps === nothing) ? 0 : length(variable_jumps)
    nrjs = ncrjs + nvrjs
    affects! = Vector{Any}(undef, nrjs)
    rates = Vector{RateWrapper}(undef, nvrjs)
    lrates = similar(rates)
    rateintervals = similar(rates)
    urates = Vector{RateWrapper}(undef, nrjs)
    haslratevec = zeros(Bool, nvrjs)

    idx = 1
    if constant_jumps !== nothing
        for crj in constant_jumps
            affects![idx] = integ -> (crj.affect!(integ); nothing)
            urates[idx] = RateWrapper(crj.rate)
            idx += 1
        end
    end

    if variable_jumps !== nothing
        for (i, vrj) in enumerate(variable_jumps)
            affects![idx] = integ -> (vrj.affect!(integ); nothing)
            urates[idx] = RateWrapper(vrj.urate)
            idx += 1
            rates[i] = RateWrapper(vrj.rate)
            rateintervals[i] = RateWrapper(vrj.rateinterval)
            haslratevec[i] = haslrate(vrj)
            lrates[i] = haslratevec[i] ? RateWrapper(vrj.lrate) : RateWrapper(nullrate)
        end
    end

    num_jumps = get_num_majumps(ma_jumps) + nrjs
    cur_rates = Vector{typeof(t)}(undef, num_jumps)
    cur_lrates = zeros(typeof(t), nvrjs)
    sum_rate = nothing
    next_jump = 0
    next_jump_time = typemax(t)
    CoevolveJumpAggregation(next_jump, next_jump_time, end_time, cur_rates, sum_rate,
        ma_jumps, rates, affects!, save_positions, rng;
        u, dep_graph, lrates, urates, rateintervals, haslratevec,
        cur_lrates)
end

# set up a new simulation and calculate the first jump / jump time
function initialize!(p::CoevolveJumpAggregation, integrator, u, params, t)
    p.end_time = integrator.sol.prob.tspan[2]
    fill_rates_and_get_times!(p, u, params, t)
    generate_jumps!(p, integrator, u, params, t)
    nothing
end

# execute one jump, changing the system state
function execute_jumps!(p::CoevolveJumpAggregation, integrator, u, params, t,
    affects!)
    # execute jump
    update_state!(p, integrator, u, affects!)
    # update current jump rates and times
    update_dependent_rates!(p, integrator.u, integrator.p, t)
    nothing
end

# calculate the next jump / jump time
function generate_jumps!(p::CoevolveJumpAggregation, integrator, u, params, t)
    p.next_jump_time, p.next_jump = top_with_handle(p.pq)
    nothing
end

######################## SSA specific helper routines ########################
function accept_next_jump!(p::CoevolveJumpAggregation, integrator, u, params, t)
    @unpack next_jump, ma_jumps = p

    num_majumps = get_num_majumps(ma_jumps)

    (next_jump <= num_majumps) && return true

    @unpack cur_rates, rates, rng, urates, cur_lrates = p
    num_cjumps = length(urates) - length(rates)
    uidx = next_jump - num_majumps
    lidx = uidx - num_cjumps

    # lidx <= 0 indicates that we have a constant jump which we always accept
    (lidx <= 0) && return true

    @inbounds urate = cur_rates[next_jump]
    @inbounds lrate = cur_lrates[lidx]

    s = -one(t)

    # the first condition applies when the candidate time t is rejected because
    # it was longer than the rateinterval when it was proposed
    if lrate == typemax(t)
        urate = get_urate(p, uidx, u, params, t)
        s = urate == zero(t) ? typemax(t) : randexp(rng) / urate
    elseif lrate < urate
        # when the lower and upper bound are the same, then v < 1 = lrate / urate = urate / urate
        v = rand(rng) * urate
        # first inequality is less expensive and short-circuits the evaluation
        if (v > lrate)
            rate = get_rate(p, lidx, u, params, t)
            if v > rate
                urate = get_urate(p, uidx, u, params, t)
                s = urate == zero(t) ? typemax(t) : randexp(rng) / urate
            end
        end
    end

    # candidate t was accepted
    (s < 0) && return true

    # candidate t was reject, compute a new candidate time
    t = next_candidate_time!(p, u, params, t, s, lidx)
    update!(p.pq, next_jump, t)
    @inbounds cur_rates[next_jump] = urate

    p.prev_jump = next_jump
    generate_jumps!(p, integrator, integrator.u, integrator.p, integrator.t)
    register_next_jump_time!(integrator, p, integrator.t)
    u_modified!(integrator, false)

    return false
end

function update_dependent_rates!(p::CoevolveJumpAggregation, u, params, t)
    @inbounds deps = p.dep_gr[p.next_jump]
    @unpack cur_rates, pq = p
    for (ix, i) in enumerate(deps)
        ti, urate_i = next_time(p, u, params, t, i)
        update!(pq, i, ti)
        @inbounds cur_rates[i] = urate_i
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

function next_time(p::CoevolveJumpAggregation, u, params, t, i)
    @unpack next_jump, cur_rates, ma_jumps, rates, rng, pq, urates = p
    num_majumps = get_num_majumps(ma_jumps)
    num_cjumps = length(urates) - length(rates)
    uidx = i - num_majumps
    lidx = uidx - num_cjumps
    urate = uidx > 0 ? get_urate(p, uidx, u, params, t) : get_ma_urate(p, i, u, params, t)
    # we can only re-use the rng in the case of constant rates because the rng
    # used to compute the next candidate time has not been accepted or rejected
    if i != next_jump && lidx <= 0
        last_urate = cur_rates[i]
        if last_urate > zero(t)
            s = urate == zero(t) ? typemax(t) : last_urate / urate * (pq[i] - t)
            return next_candidate_time!(p, u, params, t, s, lidx), urate
        end
    end
    s = urate == zero(t) ? typemax(t) : randexp(rng) / urate
    return next_candidate_time!(p, u, params, t, s, lidx), urate
end

function next_candidate_time!(p::CoevolveJumpAggregation, u, params, t, s, lidx)
    if lidx <= 0
        return t + s
    end
    @unpack end_time, haslratevec, cur_lrates = p
    rateinterval = get_rateinterval(p, lidx, u, params, t)
    if s > rateinterval
        t = t + rateinterval
        # we set the lrate to typemax(t) to indicate rejection due to candidate being larger than rateinterval
        @inbounds cur_lrates[lidx] = typemax(t)
        return t
    end
    t = t + s
    if t < end_time
        lrate = haslratevec[lidx] ? get_lrate(p, lidx, u, params, t) : zero(t)
        @inbounds cur_lrates[lidx] = lrate
    else
        # no need to compute the lower bound when time is past the end time
        @inbounds cur_lrates[lidx] = typemax(t)
    end
    return t
end

# re-evaluates all rates, recalculate all jump times, and reinit the priority queue
function fill_rates_and_get_times!(p::CoevolveJumpAggregation, u, params, t)
    num_jumps = get_num_majumps(p.ma_jumps) + length(p.urates)
    p.cur_rates = zeros(typeof(t), num_jumps)
    jump_times = Vector{typeof(t)}(undef, num_jumps)
    @inbounds for i in 1:num_jumps
        jump_times[i], p.cur_rates[i] = next_time(p, u, params, t, i)
    end
    p.pq = MutableBinaryMinHeap(jump_times)
    nothing
end
