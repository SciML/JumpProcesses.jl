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

mutable struct DirectCRJumpAggregation{T, S, F1, F2, RNG, DEPGR, U <: PriorityTable,
                                       W <: Function} <: AbstractSSAJumpAggregator
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
    minrate::T
    maxrate::T   # initial maxrate only, table can increase beyond it!
    rt::U
    ratetogroup::W
end

function DirectCRJumpAggregation(nj::Int, njt::T, et::T, crs::Vector{T}, sr::T,
                                 maj::S, rs::F1, affs!::F2, sps::Tuple{Bool, Bool},
                                 rng::RNG; num_specs, dep_graph = nothing,
                                 minrate = convert(T, MINJUMPRATE),
                                 maxrate = convert(T, Inf),
                                 kwargs...) where {T, S, F1, F2, RNG}

    # a dependency graph is needed and must be provided if there are constant rate jumps
    if dep_graph === nothing
        if (get_num_majumps(maj) == 0) || !isempty(rs)
            error("To use ConstantRateJumps with the DirectCR algorithm a dependency graph must be supplied.")
        else
            dg = make_dependency_graph(num_specs, maj)
        end
    else
        dg = dep_graph

        # make sure each jump depends on itself
        add_self_dependencies!(dg)
    end

    # mapping from jump rate to group id
    minexponent = exponent(minrate)

    # use the largest power of two that is <= the passed in minrate
    minrate = 2.0^minexponent
    ratetogroup = rate -> priortogid(rate, minexponent)

    # construct an empty initial priority table -- we'll reset this in init
    rt = PriorityTable(ratetogroup, zeros(T, 1), minrate, 2 * minrate)

    DirectCRJumpAggregation{T, S, F1, F2, RNG, typeof(dg),
                            typeof(rt), typeof(ratetogroup)}(nj, nj, njt, et, crs, sr, maj,
                                                             rs, affs!, sps, rng, dg,
                                                             minrate, maxrate, rt,
                                                             ratetogroup)
end

############################# Required Functions ##############################

# creating the JumpAggregation structure (function wrapper-based constant jumps)
function aggregate(aggregator::DirectCR, u, p, t, end_time, constant_jumps,
                   ma_jumps, save_positions, rng; kwargs...)

    # handle constant jumps using function wrappers
    rates, affects! = get_jump_info_fwrappers(u, p, t, constant_jumps)

    build_jump_aggregation(DirectCRJumpAggregation, u, p, t, end_time, ma_jumps,
                           rates, affects!, save_positions, rng; num_specs = length(u),
                           kwargs...)
end

# set up a new simulation and calculate the first jump / jump time
function initialize!(p::DirectCRJumpAggregation, integrator, u, params, t)
    p.end_time = integrator.sol.prob.tspan[2]

    # initialize rates
    fill_rates_and_sum!(p, u, params, t)

    # setup PriorityTable
    reset!(p.rt)
    for (pid, priority) in enumerate(p.cur_rates)
        insert!(p.rt, pid, priority)
    end

    generate_jumps!(p, integrator, u, params, t)
    nothing
end

# execute one jump, changing the system state
function execute_jumps!(p::DirectCRJumpAggregation, integrator, u, params, t)
    # execute jump
    u = update_state!(p, integrator, u)

    # update current jump rates
    update_dependent_rates!(p, u, params, t)
    nothing
end

# calculate the next jump / jump time
function generate_jumps!(p::DirectCRJumpAggregation, integrator, u, params, t)
    p.next_jump_time = t + randexp(p.rng) / p.sum_rate

    if p.next_jump_time < p.end_time
        p.next_jump = sample(p.rt, p.cur_rates, p.rng)
    end
    nothing
end

######################## SSA specific helper routines #########################

# recalculate jump rates for jumps that depend on the just executed jump
# requires dependency graph
function update_dependent_rates!(p::DirectCRJumpAggregation, u, params, t)
    @unpack cur_rates, rates, ma_jumps, rt = p
    @inbounds dep_rxs = p.dep_gr[p.next_jump]
    num_majumps = get_num_majumps(ma_jumps)

    @inbounds for rx in dep_rxs
        oldrate = cur_rates[rx]

        # update rate
        cur_rates[rx] = calculate_jump_rate(ma_jumps, num_majumps, rates, u, params, t, rx)

        # update table
        update!(rt, rx, oldrate, cur_rates[rx])
    end

    p.sum_rate = groupsum(rt)
    nothing
end
