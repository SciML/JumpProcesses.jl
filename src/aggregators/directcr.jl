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

mutable struct DirectCRJumpAggregation{T, S, F1, F2, DEPS, U <: PriorityTable,
    W <: Function} <:
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
    dependencies::DEPS
    minrate::T
    maxrate::T   # initial maxrate only, table can increase beyond it!
    rt::U
    ratetogroup::W
    # bound on the rounding error in `sum_rate`; kept apart from it, since storing the two
    # together measurably slowed each jump
    sum_rate_err::T
end

function DirectCRJumpAggregation(nj::Int, njt::T, et::T, crs::Vector{T}, sr::T,
        maj::S, rs::F1, affs!::F2, sps::Tuple{Bool, Bool};
        maj_rates = Vector{T}(undef, get_num_majumps(maj)),
        dependencies,
        minrate = convert(T, MINJUMPRATE),
        maxrate = convert(T, Inf),
        kwargs...) where {T, S, F1, F2}

    # mapping from jump rate to group id
    minexponent = exponent(minrate)

    # use the largest power of two that is <= the passed in minrate
    minrate = T(2)^minexponent
    ratetogroup = rate -> priortogid(rate, minexponent)

    # construct an empty initial priority table -- we'll reset this in init
    rt = PriorityTable(ratetogroup, zeros(T, 1), minrate, 2 * minrate)

    affecttype = F2 <: Tuple ? F2 : Any
    DirectCRJumpAggregation{T, S, F1, affecttype, typeof(dependencies),
        typeof(rt), typeof(ratetogroup)}(nj, nj, njt, et, crs, sr, maj,
        maj_rates, rs, affs!, sps, dependencies,
        minrate, maxrate, rt,
        ratetogroup, zero(sr))
end

# the fields that `isolated_copy` shares; it copies the others (see `shared_fields`)
function shared_fields(::Type{<:DirectCRJumpAggregation})
    (:ma_jumps, :rates, :affects!, :save_positions, :dependencies, :ratetogroup)
end

############################# Required Functions ##############################

# creating the JumpAggregation structure (function wrapper-based constant jumps)
function aggregate(aggregator::DirectCR, u, p, t, end_time, constant_jumps,
        ma_jumps, save_positions; kwargs...)

    # handle constant jumps using function wrappers
    rates, affects! = get_jump_info_fwrappers(u, p, t, constant_jumps)

    build_jump_aggregation(DirectCRJumpAggregation, u, p, t, end_time, ma_jumps,
        rates, affects!, save_positions; kwargs...)
end

# set up a new simulation and calculate the first jump / jump time
function initialize!(p::DirectCRJumpAggregation, integrator, u, params, t::Number)
    p.end_time = integrator.sol.prob.tspan[2]
    fill_scaled_rates!(p.maj_rates, p.ma_jumps, params)

    # initialize rates
    fill_rates_and_sum!(p, u, params, t)

    # setup PriorityTable
    reset!(p.rt)
    for (pid, priority) in enumerate(p.cur_rates)
        insert!(p.rt, pid, priority)
    end
    p.sum_rate_err = sum_error_bound(p.sum_rate, numterms(p.rt))

    generate_jumps!(p, integrator, u, params, t)
    nothing
end

# execute one jump, changing the system state
function execute_jumps!(p::DirectCRJumpAggregation, integrator, u, params, t, affects!)
    # execute jump
    u = update_state!(p, integrator, u, affects!)

    # update current jump rates
    @inbounds dep_rxs = dependents(p.dependencies.dep_graph, p.next_jump)
    update_dependent_rates!(p, dep_rxs, u, params, t)
    nothing
end

# calculate the next jump / jump time; @inline keeps it inlined into the per-jump affect, as
# the compiler chose before the checks of the rate sum made it longer
@inline function generate_jumps!(p::DirectCRJumpAggregation, integrator, u, params, t)
    # recompute the sums of the rates if rounding may have made them inaccurate
    if sum_rate_inaccurate(p.sum_rate, p.sum_rate_err, numterms(p.rt))
        resum_rates!(p)
    end
    nomorejumps!(p, p.sum_rate) && return nothing

    rng = get_rng(integrator)
    p.next_jump_time = t + typed_randexp(rng, t) / p.sum_rate

    if p.next_jump_time < p.end_time
        p.next_jump = sample(p.rt, p.cur_rates, rng)

        # sampling finds no jump only if the sums exceed the sum of the rates
        iszero(p.next_jump) && (p.next_jump = resum_and_sample!(p, rng))
    end
    nothing
end

# Recompute the sums of the rates after sampling found no jump, which happens only if the sums
# exceed the sum of the rates, and sample again. The recomputed sum is positive, since
# `sum_rate_inaccurate` ruled out that the exact sum is zero, and the next jump time, drawn
# with the previous sum, is off by at most its relative tolerance. A sample can still miss
# by rounding, since `sum` adds the group sums in another order than `sample` subtracts
# them, but only with probability around `eps` per try, so it repeats until it finds a jump.
@noinline function resum_and_sample!(p::DirectCRJumpAggregation, rng)
    resum_rates!(p)
    jidx = 0
    while iszero(jidx)
        jidx = sample(p.rt, p.cur_rates, rng)
    end
    return jidx
end

# recompute the priority table's sums from the rates
@noinline function resum_rates!(p::DirectCRJumpAggregation)
    resum!(p.rt, p.cur_rates)
    p.sum_rate = groupsum(p.rt)
    p.sum_rate_err = sum_error_bound(p.sum_rate, numterms(p.rt))
    nothing
end

######################## SSA specific helper routines #########################

# recalculate the rates of the jumps in dep_rxs, e.g. those depending on the executed jump
# requires dependency graph
function update_dependent_rates!(p::DirectCRJumpAggregation, dep_rxs, u, params, t)
    (; cur_rates, rates, ma_jumps, rt) = p
    num_majumps = get_num_majumps(ma_jumps)

    @inbounds for rx in dep_rxs
        oldrate = cur_rates[rx]

        # update rate
        cur_rates[rx] = calculate_jump_rate(ma_jumps, num_majumps, rates, u, params, t, rx,
            p.maj_rates)

        # update table
        update!(rt, rx, oldrate, cur_rates[rx])
    end

    # each rate change takes up to four additions or subtractions in the table's sums
    sum_rate = groupsum(rt)
    p.sum_rate_err += update_error_bound(p.sum_rate, sum_rate, 4 * length(dep_rxs))
    p.sum_rate = sum_rate
    nothing
end
