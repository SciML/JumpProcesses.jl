# Implementation the original Next Reaction Method
# Gibson and Bruck, J. Phys. Chem. A, 104 (9), (2000)

mutable struct NRMJumpAggregation{T,S,F1,F2,RNG,DEPGR,PQ} <: AbstractSSAJumpAggregator
    next_jump::Int
    next_jump_time::T
    end_time::T
    cur_rates::Vector{T}
    sum_rate::T
    ma_jumps::S
    rates::F1
    affects!::F2
    save_positions::Tuple{Bool,Bool}
    rng::RNG
    dep_gr::DEPGR
    pq::PQ
    NRMJumpAggregation{T,S,F1,F2,RNG}(nj::Int, njt::T, et::T, crs::Vector{T}, sr::T, maj::S,
                                      rs::F1, affs!::F2, sps::Tuple{Bool,Bool}, rng::RNG, 
                                      dep_gr::DEPGR, pq::PQ) where {T,S,F1,F2,RNG,DEPGR,PQ} =
      new{T,S,F1,F2,RNG,DEPGR,PQ}(nj, njt, et, crs, sr, maj, rs, affs!, sps, rng, dep_gr, pq)
  end

function NRMJumpAggregation(nj::Int, njt::T, et::T, crs::Vector{T}, sr::T,
                                      maj::S, rs::F1, affs!::F2, sps::Tuple{Bool,Bool},
                                      rng::RNG; dep_graph=nothing, kwargs...) where {T,S,F1,F2,RNG}

    # a dependency graph is needed and must be provided if there are constant rate jumps
    if dep_graph == nothing
        if (get_num_majumps(maj) == 0) || !isempty(rs)
            error("To use ConstantRateJumps with the Next Reaction Method (NRM) algorithm a dependency graph must be supplied.")
        else
            dg = make_dependency_graph(get_num_majumps(maj), maj)
        end
    else
        dg = dep_graph
    end

    # make sure each jump depends on itself
    for (i,jump_deps) in enumerate(dg)
        if !contains(isequal, jump_deps, i)
            push!(jump_deps, i)
            sort!(jump_deps)
        end
    end

    pq = ArrayPQ(Vector{Pair{Int,T}}())

    NRMJumpAggregation{T,S,F1,F2,RNG}(nj, njt, et, crs, sr, maj, rs, affs!, sps, rng, dg, pq)
end

########### The following routines should be templates for all SSAs ###########

# condition for jump to occur
@inline function (p::NRMJumpAggregation)(u, t, integrator)
    p.next_jump_time == t
end

# executing jump at the next jump time
function (p::NRMJumpAggregation)(integrator)
    execute_jumps!(p, integrator, integrator.u, integrator.p, integrator.t)
    generate_jumps!(p, integrator, integrator.u, integrator.p, integrator.t)
    register_next_jump_time!(integrator, p, integrator.t)
    nothing
end

# setting up a new simulation
function (p::NRMJumpAggregation)(dj, u, t, integrator) # initialize
    initialize!(p, integrator, u, integrator.p, t)
    register_next_jump_time!(integrator, p, t)
    nothing
end


+############################# Required Functions ##############################
# creating the JumpAggregation structure (function wrapper-based constant jumps)
function aggregate(aggregator::NRM, u, p, t, end_time, constant_jumps,
                   ma_jumps, save_positions, rng; kwargs...)

    # handle constant jumps using function wrappers
    rates, affects! = get_jump_info_fwrappers(u, p, t, constant_jumps)

    build_jump_aggregation(NRMJumpAggregation, u, p, t, end_time, ma_jumps,
                           rates, affects!, save_positions, rng; kwargs...)
end

# set up a new simulation and calculate the first jump / jump time
function initialize!(p::NRMJumpAggregation, integrator, u, params, t)
    fill_rates_and_get_times!(p, u, params, t)
    generate_jumps!(p, integrator, u, params, t)
    nothing
end

# execute one jump, changing the system state
function execute_jumps!(p::NRMJumpAggregation, integrator, u, params, t)
    # execute jump
    num_ma_rates = get_num_majumps(p.ma_jumps)
    if p.next_jump <= num_ma_rates
        @inbounds executerx!(u, p.next_jump, p.ma_jumps)
    else
        idx = p.next_jump - num_ma_rates
        @inbounds p.affects![idx](integrator)
    end

    # update current jump rates and times
    update_dependent_rates!(p, u, params, t)
    nothing
end

# calculate the next jump / jump time
# just the top of the priority queue
function generate_jumps!(p::NRMJumpAggregation, integrator, u, params, t)
    @inbounds next_jump_pair = peek(p.pq)
    @inbounds p.next_jump = next_jump_pair[1]    
    @fastmath p.next_jump_time = next_jump_pair[2]
    nothing
end


######################## SSA specific helper routines ########################

# recalculate jump rates for jumps that depend on the just executed jump (p.next_jump)
function update_dependent_rates!(p::NRMJumpAggregation, u, params, t)
    @inbounds dep_rxs = p.dep_gr[p.next_jump]
    num_majumps = get_num_majumps(p.ma_jumps)
    cur_rates   = p.cur_rates
    @inbounds for rx in dep_rxs
        oldrate = cur_rates[rx]

        # update the jump rate
        if rx <= num_majumps
            cur_rates[rx] = evalrxrate(u, rx, p.ma_jumps)
        else
            cur_rates[rx] = p.rates[rx-num_majumps](u, params, t)            
        end

        # calculate new jump times for dependent jumps
        if rx != p.next_jump && oldrate > zero(oldrate)
            p.pq[rx] = cur_rates[rx] > 0. ? t + oldrate / cur_rates[rx] * (p.pq[rx] - t) : typemax(t)
        else 
            p.pq[rx] = cur_rates[rx] > 0. ? t + randexp(p.rng) / cur_rates[rx] : typemax(t)
        end
        
    end
    nothing
end


# reevaulate all rates, recalculate all jump times, and reinit the priority queue
function fill_rates_and_get_times!(p::NRMJumpAggregation, u, params, t)
    
    # mass action jumps
    majumps   = p.ma_jumps
    cur_rates = p.cur_rates
    pqdata    = Vector{Pair{Int,typeof(t)}}(length(cur_rates))
    @inbounds for i in 1:get_num_majumps(majumps)
        cur_rates[i] = evalrxrate(u, i, majumps)
        pqdata[i]    = i => t + randexp(p.rng) / cur_rates[i]
    end

    # constant rates
    rates = p.rates
    idx   = get_num_majumps(majumps) + 1
    @inbounds for rate in rates
        cur_rates[idx] = rate(u, params, t)
        pqdata[idx]    = idx => t + randexp(p.rng) / cur_rates[idx]
        idx += 1
    end

    # setup a new indexed priority queue to storing rx times
    p.pq = ArrayPQ(pqdata)

    nothing 
end