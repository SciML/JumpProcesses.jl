"""
Queue method. This method handles variable intensity rates.
"""
mutable struct CoevolveSyncedJumpAggregation{T, S, F1, F2, RNG, GR, PQ} <:
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
    haslratevec::Vector{Bool}         # vector of whether an lrate was provided for this vrj
    cur_lrates::Vector{T}             # the last computed lower rate for each rate
    save_everyjump::Bool              # whether to save every jump
end

function CoevolveSyncedJumpAggregation(nj::Int, njt::T, et::T, crs::Vector{T}, sr::Nothing,
                                       maj::S, rs::F1, affs!::F2, sps::Tuple{Bool, Bool},
                                       rng::RNG; u::U, dep_graph = nothing, lrates, urates,
                                       rateintervals, haslratevec,
                                       cur_lrates::Vector{T},
                                       save_everyjump::Bool) where {T, S, F1, F2, RNG, U}
    if dep_graph === nothing
        if (get_num_majumps(maj) == 0) || !isempty(urates)
            error("To use CoevolveSynced a dependency graph between jumps must be supplied.")
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
    CoevolveSyncedJumpAggregation{T, S, F1, F2, RNG, typeof(dg),
                                  typeof(pq)}(nj, nj, njt, et, crs, sr, maj, rs, affs!, sps,
                                              rng,
                                              dg, pq, lrates, urates, rateintervals,
                                              haslratevec,
                                              cur_lrates, save_everyjump)
end

# creating the JumpAggregation structure (tuple-based variable jumps)
function aggregate(aggregator::CoevolveSynced, u, p, t, end_time, constant_jumps,
                   ma_jumps, save_positions, rng; dep_graph = nothing,
                   variable_jumps = nothing, kwargs...)
    AffectWrapper = FunctionWrappers.FunctionWrapper{Nothing, Tuple{Any}}
    RateWrapper = FunctionWrappers.FunctionWrapper{typeof(t),
                                                   Tuple{typeof(u), typeof(p), typeof(t)}}

    ncrjs = (constant_jumps === nothing) ? 0 : length(constant_jumps)
    nvrjs = (variable_jumps === nothing) ? 0 : length(variable_jumps)
    nrjs = ncrjs + nvrjs
    affects! = Vector{AffectWrapper}(undef, nrjs)
    rates = Vector{RateWrapper}(undef, nvrjs)
    lrates = similar(rates)
    rateintervals = similar(rates)
    urates = Vector{RateWrapper}(undef, nrjs)
    haslratevec = zeros(Bool, nvrjs)

    idx = 1
    if constant_jumps !== nothing
        for crj in constant_jumps
            affects![idx] = AffectWrapper(integ -> (crj.affect!(integ); nothing))
            urates[idx] = RateWrapper(crj.rate)
            idx += 1
        end
    end

    if variable_jumps !== nothing
        for (i, vrj) in enumerate(variable_jumps)
            affects![idx] = AffectWrapper(integ -> (vrj.affect!(integ); nothing))
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
    save_everyjump = any(save_positions)
    CoevolveSyncedJumpAggregation(next_jump, next_jump_time, end_time, cur_rates, sum_rate,
                                  ma_jumps, rates, affects!, save_positions, rng;
                                  u, dep_graph, lrates, urates, rateintervals, haslratevec,
                                  cur_lrates, save_everyjump)
end

# set up a new simulation and calculate the first jump / jump time
function initialize!(p::CoevolveSyncedJumpAggregation, integrator, u, params, t)
    p.end_time = integrator.sol.prob.tspan[2]
    fill_rates_and_get_times!(p, u, params, t)
    generate_jumps!(p, integrator, u, params, t)
    nothing
end

# execute one jump, changing the system state
function execute_jumps!(p::CoevolveSyncedJumpAggregation, integrator, u, params, t)
    @unpack next_jump, ma_jumps, save_everyjump = p

    toggle_save_everystep!(p, integrator)

    num_majumps = get_num_majumps(ma_jumps)

    # execute jump
    if next_jump <= num_majumps # is next jump a mass action jump
        if u isa SVector
            integrator.u = executerx(u, next_jump, ma_jumps)
        else
            @inbounds executerx!(u, next_jump, ma_jumps)
        end
    else
        @unpack cur_rates, rates, rng, urates, cur_lrates = p
        num_cjumps = length(urates) - length(rates)
        uidx = next_jump - num_majumps
        lidx = uidx - num_cjumps
        if lidx > 0
            @inbounds urate = cur_rates[next_jump]
            @inbounds lrate = cur_lrates[lidx]
            s = -1
            if lrate == typemax(t)
                urate = get_urate(p, uidx, u, params, t)
                s = urate == zero(t) ? typemax(t) : randexp(rng) / urate
            elseif lrate < urate
                # when the lower and upper bound are the same, then v < 1 = lrate / urate = urate / urate
                v = rand(rng) * urate
                # first inequality is less expensive and short-circuits the evaluation
                if (v > lrate) && (v > get_rate(p, lidx, u, params, t))
                    urate = get_urate(p, uidx, u, params, t)
                    s = urate == zero(t) ? typemax(t) : randexp(rng) / urate
                end
            elseif lrate > urate
                error("The lower bound should be lower than the upper bound rate for t = $(t) and i = $(next_jump), but lower bound = $(lrate) > upper bound = $(urate)")
            end
            if s >= 0
                t = next_candidate_time!(p, u, params, t, s, lidx)
                update!(p.pq, next_jump, t)
                @inbounds cur_rates[next_jump] = urate
                # do not save step when candidate time is rejected
                toggle_save_everystep!(p, integrator; value = false)
                return nothing
            end
        end
        @inbounds p.affects![uidx](integrator)
    end

    p.prev_jump = next_jump

    # update current jump rates and times
    update_dependent_rates!(p, integrator.u, params, t)

    nothing
end

function toggle_save_everystep!(p::CoevolveSyncedJumpAggregation, integrator::T;
                                value = p.save_everyjump) where {T <: AbstractSSAIntegrator}
    integrator.save_everystep = value
end

function toggle_save_everystep!(p::CoevolveSyncedJumpAggregation, integrator;
                                value = p.save_everyjump)
    nothing
end

# calculate the next jump / jump time
function generate_jumps!(p::CoevolveSyncedJumpAggregation, integrator, u, params, t)
    p.next_jump_time, p.next_jump = top_with_handle(p.pq)
    if p.next_jump_time > p.end_time
        # restore the option to original value
        toggle_save_everystep!(p, integrator)
    end
    nothing
end

######################## SSA specific helper routines ########################
function update_dependent_rates!(p::CoevolveSyncedJumpAggregation, u, params, t)
    @inbounds deps = p.dep_gr[p.next_jump]
    @unpack cur_rates, pq = p
    for (ix, i) in enumerate(deps)
        ti, last_urate_i = next_time(p, u, params, t, i)
        update!(pq, i, ti)
        @inbounds cur_rates[i] = last_urate_i
    end
    nothing
end

@inline function get_ma_urate(p::CoevolveSyncedJumpAggregation, i, u, params, t)
    return evalrxrate(u, i, p.ma_jumps)
end

@inline function get_urate(p::CoevolveSyncedJumpAggregation, uidx, u, params, t)
    @inbounds return p.urates[uidx](u, params, t)
end

@inline function get_rateinterval(p::CoevolveSyncedJumpAggregation, lidx, u, params, t)
    @inbounds return p.rateintervals[lidx](u, params, t)
end

@inline function get_lrate(p::CoevolveSyncedJumpAggregation, lidx, u, params, t)
    @inbounds return p.lrates[lidx](u, params, t)
end

@inline function get_rate(p::CoevolveSyncedJumpAggregation, lidx, u, params, t)
    @inbounds return p.rates[lidx](u, params, t)
end

function next_time(p::CoevolveSyncedJumpAggregation, u, params, t, i)
    @unpack next_jump, cur_rates, ma_jumps, rates, rng, pq, urates = p
    num_majumps = get_num_majumps(ma_jumps)
    num_cjumps = length(urates) - length(rates)
    uidx = i - num_majumps
    lidx = uidx - num_cjumps
    urate = uidx > 0 ? get_urate(p, uidx, u, params, t) : get_ma_urate(p, i, u, params, t)
    # we can only re-use the rng in the case of contstant rates because the rng
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

function next_candidate_time!(p::CoevolveSyncedJumpAggregation, u, params, t, s, lidx)
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

# reevaulate all rates, recalculate all jump times, and reinit the priority queue
function fill_rates_and_get_times!(p::CoevolveSyncedJumpAggregation, u, params, t)
    num_jumps = get_num_majumps(p.ma_jumps) + length(p.urates)
    p.cur_rates = zeros(typeof(t), num_jumps)
    jump_times = Vector{typeof(t)}(undef, num_jumps)
    @inbounds for i in 1:num_jumps
        jump_times[i], p.cur_rates[i] = next_time(p, u, params, t, i)
    end
    p.pq = MutableBinaryMinHeap(jump_times)
    nothing
end
