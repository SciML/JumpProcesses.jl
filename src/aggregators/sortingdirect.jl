mutable struct SortingDirectJumpAggregation{T,S,F1,F2,RNG,DEPGR} <: AbstractSSAJumpAggregator
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
    jump_search_order::Vector{Int}
    jump_search_idx::Int
    SortingDirectJumpAggregation{T,S,F1,F2,RNG}(nj::Int, njt::T, et::T, crs::Vector{T}, sr::T, maj::S,
                                                rs::F1, affs!::F2, sps::Tuple{Bool,Bool}, rng::RNG,
                                                dep_gr::DEPGR, jtoidx::Vector{Int}) where {T,S,F1,F2,RNG,DEPGR} =
      new{T,S,F1,F2,RNG,DEPGR}(nj, njt, et, crs, sr, maj, rs, affs!, sps, rng, dep_gr, jtoidx, zero(Int))
  end

function SortingDirectJumpAggregation(nj::Int, njt::T, et::T, crs::Vector{T}, sr::T,
                                      maj::S, rs::F1, affs!::F2, sps::Tuple{Bool,Bool},
                                      rng::RNG; dep_graph=nothing, kwargs...) where {T,S,F1,F2,RNG}

    # a dependency graph is needed and must be provided if there are constant rate jumps
    if dep_graph == nothing
        if (get_num_majumps(maj) == 0) || !isempty(rs)
            error("To use ConstantRateJumps with the SortingDirect algorithm a dependency graph must be supplied.")
        else
            dg = make_dependency_graph(get_num_majumps(maj), maj)
        end
    else
        dg = dep_graph
    end

    # map jump idx to idx in cur_rates
    jtoidx = collect(1:length(crs))
    SortingDirectJumpAggregation{T,S,F1,F2,RNG}(nj, njt, et, crs, sr, maj, rs, affs!, sps, rng, dg, jtoidx)
end


  ########### The following routines should be templates for all SSAs ###########

# condition for jump to occur
@inline function (p::SortingDirectJumpAggregation)(u, t, integrator)
    p.next_jump_time == t
end

# executing jump at the next jump time
function (p::SortingDirectJumpAggregation)(integrator)
    execute_jumps!(p, integrator, integrator.u, integrator.p, integrator.t)
    generate_jumps!(p, integrator, integrator.u, integrator.p, integrator.t)
    register_next_jump_time!(integrator, p, integrator.t)
    nothing
end

# setting up a new simulation
function (p::SortingDirectJumpAggregation)(dj, u, t, integrator) # initialize
    initialize!(p, integrator, u, integrator.p, t)
    register_next_jump_time!(integrator, p, t)
    nothing
end

  # creating the JumpAggregation structure (function wrapper-based constant jumps)
function aggregate(aggregator::SortingDirect, u, p, t, end_time, constant_jumps,
                   ma_jumps, save_positions, rng; kwargs...)

    # handle constant jumps using function wrappers
    rates, affects! = get_jump_info_fwrappers(u, p, t, constant_jumps)

    build_jump_aggregation(SortingDirectJumpAggregation, u, p, t, end_time, ma_jumps,
                           rates, affects!, save_positions, rng; kwargs...)
end

# set up a new simulation and calculate the first jump / jump time
function initialize!(p::SortingDirectJumpAggregation, integrator, u, params, t)
    fill_rates_and_sum!(p, u, params, t)
    generate_jumps!(p, integrator, u, params, t)
    nothing
end

# execute one jump, changing the system state
function execute_jumps!(p::SortingDirectJumpAggregation, integrator, u, params, t)
    # execute jump
    num_ma_rates = get_num_majumps(p.ma_jumps)
    if p.next_jump <= num_ma_rates
        @inbounds executerx!(u, p.next_jump, p.ma_jumps)
    else
        idx = p.next_jump - num_ma_rates
        @inbounds p.affects![idx](integrator)
    end

    # update search order
    jso   = p.jump_search_order
    jsidx = p.jump_search_idx
    if jsidx != 1
        @inbounds tmp          = jso[jsidx]
        @inbounds jso[jsidx]   = jso[jsidx-1]
        @inbounds jso[jsidx-1] = tmp
    end

    # update current jump rates
    update_dependent_rates!(p, u, params, t)
    nothing
end

# calculate the next jump / jump time
function generate_jumps!(p::SortingDirectJumpAggregation, integrator, u, params, t)
    @fastmath p.next_jump_time = t + calc_next_jump!(p, u, params, t)
    nothing
end


######################## SSA specific helper routines ########################

# recalculate jump rates for jumps that depend on the just executed jump (p.next_jump)
function update_dependent_rates!(p, u, params, t)
    @inbounds dep_rxs = p.dep_gr[p.next_jump]
    num_majumps = get_num_majumps(p.ma_jumps)
    cur_rates   = p.cur_rates
    sum_rate    = p.sum_rate
    majumps     = p.ma_jumps
    @inbounds for rx in dep_rxs
        sum_rate -= cur_rates[rx]
        if rx <= num_majumps
            @inbounds cur_rates[rx] = evalrxrate(u, rx, majumps)
        else
            @inbounds cur_rates[rx] = p.rates[rx-num_majumps](u, params, t)
        end
        sum_rate += cur_rates[rx]
    end

    p.sum_rate = sum_rate
end


function fill_rates_and_sum!(p, u, params, t)
    sum_rate = zero(typeof(p.sum_rate))

    # mass action jumps
    majumps   = p.ma_jumps
    cur_rates = p.cur_rates
    @inbounds for i in 1:get_num_majumps(majumps)
        cur_rates[i] = evalrxrate(u, i, majumps)
        sum_rate    += cur_rates[i]
    end

    # constant rates
    rates = p.rates
    idx   = get_num_majumps(majumps) + 1
    @inbounds for rate in rates
        cur_rates[idx] = rate(u, params, t)
        sum_rate += cur_rates[idx]
        idx += 1
    end

    p.sum_rate = sum_rate
end

# searches down the rate list for the next reaction
@fastmath function calc_next_jump!(p, u, params, t)
    
    # next jump type
    cur_rates = p.cur_rates
    numjumps  = length(cur_rates)
    jso       = p.jump_search_order
    rn        = p.sum_rate * rand(p.rng)
    @inbounds for idx = 1:numjumps
        rn -= cur_rates[jso[idx]]
        if rn < zero(rn)
            p.jump_search_idx = idx
            break
        end
    end
    @inbounds p.next_jump = jso[p.jump_search_idx]

    # return time to next jump
    randexp(p.rng) / p.sum_rate

end
