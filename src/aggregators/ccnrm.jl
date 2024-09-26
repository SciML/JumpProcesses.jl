# Implementation of the constant-complexity Next Reaction Method
# Kevin R. Sanft and Hans G. Othmer, Constant-complexity stochastic simulation
# algorithm with optimal binning,  Journal of Chemical Physics 143, 074108
# (2015). doi: 10.1063/1.4928635.

const BINWIDTH_OVER_AVGTIME = 16

mutable struct CCNRMJumpAggregation{T, S, F1, F2, RNG, DEPGR, PT} <:
               AbstractSSAJumpAggregator{T, S, F1, F2, RNG}
    next_jump::Int
    prev_jump::Int
    next_jump_time::T
    end_time::T
    cur_rates::Vector{T}
    sum_rate::T
    ma_jumps::S
    rates::F1
    affects!::F2
    save_positions::Tuple{Bool, Bool}
    rng::RNG
    dep_gr::DEPGR
    ptt::PT
end

function CCNRMJumpAggregation(nj::Int, njt::T, et::T, crs::Vector{T}, sr::T,
        maj::S, rs::F1, affs!::F2, sps::Tuple{Bool, Bool},
        rng::RNG; num_specs, dep_graph = nothing,
        kwargs...) where {T, S, F1, F2, RNG}

    # a dependency graph is needed and must be provided if there are constant rate jumps
    if dep_graph === nothing
        if (get_num_majumps(maj) == 0) || !isempty(rs)
            error("To use ConstantRateJumps with the constant-complexity Next Reaction Method (CCNRM) algorithm a dependency graph must be supplied.")
        else
            dg = make_dependency_graph(num_specs, maj)
        end
    else
        dg = dep_graph

        # make sure each jump depends on itself
        add_self_dependencies!(dg)
    end

    ptt = PriorityTimeTable(zeros(T, length(crs)), 0.0, 1.0) # We will re-initialize this in initialize!()

    affecttype = F2 <: Tuple ? F2 : Any
    CCNRMJumpAggregation{T, S, F1, affecttype, RNG, typeof(dg), typeof(ptt)}(
        nj, nj, njt, et,
        crs, sr, maj,
        rs, affs!, sps,
        rng, dg, ptt)
end

+############################# Required Functions ##############################
# creating the JumpAggregation structure (function wrapper-based constant jumps)
function aggregate(aggregator::CCNRM, u, p, t, end_time, constant_jumps,
        ma_jumps, save_positions, rng; kwargs...)

    # handle constant jumps using function wrappers
    rates, affects! = get_jump_info_fwrappers(u, p, t, constant_jumps)

    build_jump_aggregation(CCNRMJumpAggregation, u, p, t, end_time, ma_jumps,
        rates, affects!, save_positions, rng; num_specs = length(u),
        kwargs...)
end

# set up a new simulation and calculate the first jump / jump time
function initialize!(p::CCNRMJumpAggregation, integrator, u, params, t)
    p.end_time = integrator.sol.prob.tspan[2]
    initialize_rates_and_times!(p, u, params, t)
    generate_jumps!(p, integrator, u, params, t)
    nothing
end

# execute one jump, changing the system state
function execute_jumps!(p::CCNRMJumpAggregation, integrator, u, params, t, affects!)
    # execute jump
    u = update_state!(p, integrator, u, affects!)

    # update current jump rates and times
    update_dependent_rates!(p, u, params, t)
    nothing
end

# calculate the next jump / jump time
# just the first reaction in the first non-empty bin in the priority table
function generate_jumps!(p::CCNRMJumpAggregation, integrator, u, params, t)
    p.next_jump, p.next_jump_time = getfirst(p.ptt)

    # Rebuild the table if no next jump is found. 
    if p.next_jump == 0
        binwidth = BINWIDTH_OVER_AVGTIME / sum(p.cur_rates)
        min_time = minimum(p.ptt.times)
        rebuild!(p.ptt, min_time, binwidth)
        p.next_jump, p.next_jump_time = getfirst(p.ptt)
    end

    nothing
end

######################## SSA specific helper routines ########################

# Recalculate jump rates for jumps that depend on the just executed jump (p.next_jump)
function update_dependent_rates!(p::CCNRMJumpAggregation, u, params, t)
    @inbounds dep_rxs = p.dep_gr[p.next_jump]
    @unpack ptt, cur_rates, rates, ma_jumps, end_time = p
    num_majumps = get_num_majumps(ma_jumps)

    @inbounds for rx in dep_rxs
        oldrate = cur_rates[rx]
        times = ptt.times
        oldtime = times[rx]

        # update the jump rate
        @inbounds cur_rates[rx] = calculate_jump_rate(ma_jumps, num_majumps, rates, u,
            params, t, rx)

        # Calculate new jump times for dependent jumps
        if rx != p.next_jump && oldrate > zero(oldrate)
            if cur_rates[rx] > zero(eltype(cur_rates))
                update!(ptt, rx, oldtime, t + oldrate / cur_rates[rx] * (times[rx] - t))
            else
                update!(ptt, rx, oldtime, 2 * end_time)
            end
        else
            if cur_rates[rx] > zero(eltype(cur_rates))
                update!(ptt, rx, oldtime, t + randexp(p.rng) / cur_rates[rx])
            else
                update!(ptt, rx, oldtime, 2 * end_time)
            end
        end
    end
    nothing
end

# Evaluate all the rates and initialize the times in the priority table. 
function initialize_rates_and_times!(p::CCNRMJumpAggregation, u, params, t)
    # Initialize next-reaction times for the mass action jumps
    majumps = p.ma_jumps
    cur_rates = p.cur_rates
    pttdata = Vector{typeof(t)}(undef, length(cur_rates))
    @inbounds for i in 1:get_num_majumps(majumps)
        cur_rates[i] = evalrxrate(u, i, majumps)
        pttdata[i] = t + randexp(p.rng) / cur_rates[i]
    end

    # Initialize next-reaction times for the constant rates
    rates = p.rates
    idx = get_num_majumps(majumps) + 1
    @inbounds for rate in rates
        cur_rates[idx] = rate(u, params, t)
        pttdata[idx] = t + randexp(p.rng) / cur_rates[idx]
        idx += 1
    end

    # Build the priority time table with the times and bin width. 
    binwidth = BINWIDTH_OVER_AVGTIME / sum(cur_rates)
    p.ptt.times = pttdata
    rebuild!(p.ptt, t, binwidth)
    nothing
end
