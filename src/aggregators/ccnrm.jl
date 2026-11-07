# Implementation of the constant-complexity Next Reaction Method
# Kevin R. Sanft and Hans G. Othmer, Constant-complexity stochastic simulation
# algorithm with optimal binning,  Journal of Chemical Physics 143, 074108
# (2015). doi: 10.1063/1.4928635.

mutable struct CCNRMJumpAggregation{T, S, F1, F2, DEPGR, PT} <:
               AbstractSSAJumpAggregator{T, S, F1, F2}
    next_jump::Int
    prev_jump::Int
    next_jump_time::T
    end_time::T
    cur_rates::Vector{T}
    sum_rate::T
    ma_jumps::S
    maj_rates::Vector{T}
    rates::F1
    affects!::F2
    save_positions::Tuple{Bool, Bool}
    dep_gr::DEPGR
    ptt::PT
end

function CCNRMJumpAggregation(nj::Int, njt::T, et::T, crs::Vector{T}, sr::T,
        maj::S, rs::F1, affs!::F2, sps::Tuple{Bool, Bool};
        maj_rates = Vector{T}(undef, get_num_majumps(maj)),
        dep_graph,
        kwargs...) where {T, S, F1, F2}

    # `JumpProblem` passes the dependency graph in normal form, each jump listing itself
    dg = dep_graph

    binwidthconst = haskey(kwargs, :binwidthconst) ? kwargs[:binwidthconst] : 16
    numbinsconst = haskey(kwargs, :numbinsconst) ? kwargs[:numbinsconst] : 20
    ptt = PriorityTimeTable(zeros(T, length(crs)), zero(T), one(T),
        binwidthconst = binwidthconst, numbinsconst = numbinsconst) # We will re-initialize this in initialize!()

    affecttype = F2 <: Tuple ? F2 : Any
    CCNRMJumpAggregation{T, S, F1, affecttype, typeof(dg), typeof(ptt)}(
        nj, nj, njt, et,
        crs, sr, maj, maj_rates,
        rs, affs!, sps,
        dg, ptt)
end

############################# Required Functions ##############################
# creating the JumpAggregation structure (function wrapper-based constant jumps)
function aggregate(aggregator::CCNRM, u, p, t, end_time, constant_jumps,
        ma_jumps, save_positions; kwargs...)

    # handle constant jumps using function wrappers
    rates, affects! = get_jump_info_fwrappers(u, p, t, constant_jumps)

    build_jump_aggregation(CCNRMJumpAggregation, u, p, t, end_time, ma_jumps,
        rates, affects!, save_positions; kwargs...)
end

# set up a new simulation and calculate the first jump / jump time
function initialize!(p::CCNRMJumpAggregation, integrator, u, params, t::Number)
    p.end_time = integrator.sol.prob.tspan[2]
    fill_scaled_rates!(p.maj_rates, p.ma_jumps, params)
    rng = get_rng(integrator)
    initialize_rates_and_times!(p, u, params, t, rng)
    generate_jumps!(p, integrator, u, params, t)
    nothing
end

# execute one jump, changing the system state
function execute_jumps!(p::CCNRMJumpAggregation, integrator, u, params, t, affects!)
    # execute jump
    u = update_state!(p, integrator, u, affects!)

    # update current jump rates and times
    rng = get_rng(integrator)
    @inbounds dep_rxs = dependents(p.dep_gr, p.next_jump)
    update_dependent_rates!(p, dep_rxs, p.next_jump, u, params, t, rng)
    nothing
end

# calculate the next jump / jump time
# just the first reaction in the first non-empty bin in the priority table
function generate_jumps!(p::CCNRMJumpAggregation, integrator, u, params, t)
    p.next_jump, p.next_jump_time = getfirst(p.ptt)

    # Rebuild the table if no next jump is found.
    if p.next_jump == 0
        timestep = 1 / sum(p.cur_rates)
        min_time = minimum(p.ptt.times)
        rebuild!(p.ptt, min_time, timestep)
        p.next_jump, p.next_jump_time = getfirst(p.ptt)
    end

    # jumps with rate zero have the time floatmax in the table, which an infinite end time
    # would reach: no jump comes
    if p.next_jump_time == floatmax(typeof(p.next_jump_time))
        p.next_jump_time = typemax(p.next_jump_time)
    end
    nothing
end

######################## SSA specific helper routines ########################

# Recalculate the rates and times of the jumps in dep_rxs, e.g. the jumps that depend on
# executed_jump, the jump just executed (0 if none), which gets a new time
function update_dependent_rates!(p::CCNRMJumpAggregation, dep_rxs, executed_jump, u,
        params, t, rng)
    (; ptt, cur_rates, rates, ma_jumps, end_time) = p
    num_majumps = get_num_majumps(ma_jumps)

    @inbounds for rx in dep_rxs
        oldrate = cur_rates[rx]
        times = ptt.times
        oldtime = times[rx]

        # update the jump rate
        @inbounds cur_rates[rx] = calculate_jump_rate(ma_jumps, num_majumps, rates, u,
            params, t, rx, p.maj_rates)

        # Calculate new jump times for dependent jumps
        if rx != executed_jump && oldrate > zero(oldrate)
            if cur_rates[rx] > zero(eltype(cur_rates))
                update!(ptt, rx, oldtime, t + oldrate / cur_rates[rx] * (oldtime - t))
            else
                update!(ptt, rx, oldtime, floatmax(typeof(t)))
            end
        else
            if cur_rates[rx] > zero(eltype(cur_rates))
                update!(ptt, rx, oldtime, t + randexp(rng) / cur_rates[rx])
            else
                update!(ptt, rx, oldtime, floatmax(typeof(t)))
            end
        end
    end
    nothing
end

# Evaluate all the rates and initialize the times in the priority table.
function initialize_rates_and_times!(p::CCNRMJumpAggregation, u, params, t, rng)
    # Initialize next-reaction times for the mass action jumps
    majumps = p.ma_jumps
    cur_rates = p.cur_rates
    pttdata = Vector{typeof(t)}(undef, length(cur_rates))
    maj_rates = p.maj_rates
    @inbounds for i in 1:get_num_majumps(majumps)
        cur_rates[i] = evalrxrate(u, i, majumps, maj_rates)
        pttdata[i] = t + randexp(rng) / cur_rates[i]
    end

    # Initialize next-reaction times for the constant rates
    rates = p.rates
    idx = get_num_majumps(majumps) + 1
    @inbounds for rate in rates
        cur_rates[idx] = rate(u, params, t)
        pttdata[idx] = t + randexp(rng) / cur_rates[idx]
        idx += 1
    end

    # jumps with rate zero get the time floatmax, as in updates: when every rate is zero the
    # bin width is Inf, and the time Inf would give the bin index NaN
    replace!(pttdata, typemax(eltype(pttdata)) => floatmax(eltype(pttdata)))

    # Build the priority time table with the times and bin width.
    timestep = 1 / sum(cur_rates)
    p.ptt.times = pttdata
    rebuild!(p.ptt, t, timestep)
    nothing
end
