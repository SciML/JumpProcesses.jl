"""
Composition-Rejection Direct Method (DirectCR), implementation combining
features from the original article:
*A constant-time kinetic Monte Carlo algorithm for simulation of large biochemical reaction networks*,
by A. Slepoy, A.P. Thompson and S.J. Plimpton, J. Chem. Phys, 128, 205101 (2008).
and
*Efficient Formulations for Exact Stochastic Simulation of Chemical Systems*,
by S. Mauch and M. Stalzer, ACM Trans. Comp. Biol. and Bioinf., 8, No. 1, 27-35 (2010).
"""

const MINJUMPRATE = 2.0^exponent(1e-12)
const MAXJUMPRATE = 2.0^exponent(1e12)

mutable struct DirectCRJumpAggregation{T,S,F1,F2,RNG,DEPGR,U<:PriorityTable,W<:Function} <: AbstractSSAJumpAggregator
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
    minrate::T   
    maxrate::T   # initial maxrate only, table can increase beyond it!
    rt::U
    ratetogroup::W
  end

function DirectCRJumpAggregation(nj::Int, njt::T, et::T, crs::Vector{T}, sr::T,
                                      maj::S, rs::F1, affs!::F2, sps::Tuple{Bool,Bool},
                                      rng::RNG; num_specs, dep_graph=nothing, 
                                      minrate=convert(T,MINJUMPRATE), maxrate=convert(T,MAXJUMPRATE),
                                      kwargs...) where {T,S,F1,F2,RNG}

    # a dependency graph is needed and must be provided if there are constant rate jumps
    if dep_graph === nothing
        if (get_num_majumps(maj) == 0) || !isempty(rs)
            error("To use ConstantRateJumps with the DirectCR algorithm a dependency graph must be supplied.")
        else
            dg = make_dependency_graph(num_specs, maj)
        end
    else
        dg = dep_graph
    end

    # make sure each jump depends on itself
    add_self_dependencies!(dg)

    # mapping from jump rate to group id
    minexponent = exponent(minrate)
    ratetogroup = rate -> priortogid(rate, minexponent)

    # construct an empty initial priority table -- we'll overwrite this in init anyways...
    rt = PriorityTable{T,Int,Int,typeof(ratetogroup)}(minrate, maxrate, 
                                                        Vector{PriorityGroup{T,Vector{Int}}}(),
                                                        Vector{T}(), zero(T), 
                                                        Vector{Tuple{Int,Int}}(), ratetogroup)

    DirectCRJumpAggregation{T,S,F1,F2,RNG,typeof(dg),typeof(rt),typeof(ratetogroup)}(
                                            nj, njt, et, crs, sr, maj, rs, affs!, sps, rng, 
                                            dg, minrate, maxrate, rt, ratetogroup)
end



########### The following routines should be templates for all SSAs ###########

# condition for jump to occur
@inline function (p::DirectCRJumpAggregation)(u, t, integrator)
    p.next_jump_time == t
end

# executing jump at the next jump time
function (p::DirectCRJumpAggregation)(integrator)
    execute_jumps!(p, integrator, integrator.u, integrator.p, integrator.t)
    generate_jumps!(p, integrator, integrator.u, integrator.p, integrator.t)
    register_next_jump_time!(integrator, p, integrator.t)
    nothing
end

# setting up a new simulation
function (p::DirectCRJumpAggregation)(dj, u, t, integrator) # initialize
    initialize!(p, integrator, u, integrator.p, t)
    register_next_jump_time!(integrator, p, t)
    nothing
end

############################# Required Functions ##############################

# creating the JumpAggregation structure (function wrapper-based constant jumps)
function aggregate(aggregator::DirectCR, u, p, t, end_time, constant_jumps,
                   ma_jumps, save_positions, rng; kwargs...)

    # handle constant jumps using function wrappers
    rates, affects! = get_jump_info_fwrappers(u, p, t, constant_jumps)

    build_jump_aggregation(DirectCRJumpAggregation, u, p, t, end_time, ma_jumps,
                           rates, affects!, save_positions, rng; num_specs=length(u), kwargs...)
end

# set up a new simulation and calculate the first jump / jump time
function initialize!(p::DirectCRJumpAggregation, integrator, u, params, t)
    
    # initialize rates 
    fill_rates_and_sum!(p, u, params, t)

    # setup PriorityTable
    rt   = PriorityTable(p.ratetogroup, p.cur_rates, p.minrate, p.maxrate)
    p.rt = rt

    generate_jumps!(p, integrator, u, params, t)
    nothing
end

# execute one jump, changing the system state
function execute_jumps!(p::DirectCRJumpAggregation, integrator, u, params, t)
    # execute jump
    num_ma_rates = get_num_majumps(p.ma_jumps)
    if p.next_jump <= num_ma_rates
        @inbounds executerx!(u, p.next_jump, p.ma_jumps)
    else
        idx = p.next_jump - num_ma_rates
        @inbounds p.affects![idx](integrator)
    end

    # update current jump rates
    update_dependent_rates!(p, u, params, t)
    nothing
end

# calculate the next jump / jump time
function generate_jumps!(p::DirectCRJumpAggregation, integrator, u, params, t)
    @fastmath p.next_jump_time = t + calc_next_jump!(p, u, params, t)
    nothing
end


######################## SSA specific helper routines #########################

# searches down the rate list for the next reaction
@fastmath function calc_next_jump!(p::DirectCRJumpAggregation, u, params, t)

    # next jump type
    p.next_jump = sample(p.rt, p.cur_rates, p.rng)

    # return time to next jump
    randexp(p.rng) / p.sum_rate
end


# recalculate jump rates for jumps that depend on the just executed jump
# requires dependency graph
function update_dependent_rates!(p::DirectCRJumpAggregation, u, params, t)
    @unpack cur_rates, rates, ma_jumps, rt = p
    @inbounds dep_rxs = p.dep_gr[p.next_jump]
    num_majumps = get_num_majumps(ma_jumps)

    @inbounds for rx in dep_rxs
        oldrate = cur_rates[rx]

        # update rate
        if rx <= num_majumps            
            newrate = evalrxrate(u, rx, ma_jumps)
        else
            newrate = rates[rx-num_majumps](u, params, t)
        end
        cur_rates[rx] = newrate

        # update table
        update!(rt, rx, oldrate, newrate)
    end
  
    p.sum_rate = groupsum(rt)
    nothing
  end
  