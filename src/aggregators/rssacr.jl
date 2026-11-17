"""
Composition-Rejection with Rejection sampling method (RSSA-CR)
"""

const MINJUMPRATE = 2.0^exponent(1e-12)

mutable struct RSSACRJumpAggregation{F, S, F1, F2, CB, U, DEPS, BD,
    P <: PriorityTable, W <: Function} <:
               AbstractSSAJumpAggregator{F, S, F1, F2}
    next_jump::Int
    prev_jump::Int
    next_jump_time::F
    end_time::F
    cur_rate_low::Vector{F}
    cur_rate_high::Vector{F}
    sum_rate::F
    ma_jumps::S
    maj_rates::Vector{F}
    rates::F1
    affects!::F2
    brackets::CB
    save_positions::Tuple{Bool, Bool}
    dependencies::DEPS
    bracket_data::BD
    ulow::U
    uhigh::U
    minrate::F
    maxrate::F   # initial maxrate only, table can increase beyond it!
    rt::P #rate table
    ratetogroup::W
    # bound on the rounding error in `sum_rate`; kept apart from it, since storing the two
    # together measurably slowed each jump
    sum_rate_err::F
end

function RSSACRJumpAggregation(nj::Int, njt::F, et::F, crs::Vector{F}, sum_rate::F, maj::S,
        rs::F1, affs!::F2, sps::Tuple{Bool, Bool};
        maj_rates = Vector{F}(undef, get_num_majumps(maj)),
        u::U, brackets, dependencies,
        bracket_data = nothing, minrate = convert(F, MINJUMPRATE),
        maxrate = convert(F, Inf),
        kwargs...) where {F, S, F1, F2, U}
    # vectors to store bracketing intervals for jump rates
    crl_bnds = similar(crs)
    crh_bnds = similar(crs)

    # a bracket data structure is needed for updating species populations
    bd = (bracket_data === nothing) ? BracketData{F, eltype(U)}() : bracket_data

    # matrix to store bracketing interval for species and the relative interval width
    # first row is Xlow, second is Xhigh
    ulow = similar(u)
    uhigh = similar(u)

    # mapping from jump rate to group id, in the rates' type. Rates at or below eps(F) all go
    # in the table's first group, so a smaller minrate only adds empty groups, one per power
    # of two (about 10^9 for floatmin(BigFloat)): it is raised to eps(F)
    minrate = max(convert(F, minrate), eps(F))
    minexponent = exponent(minrate)

    # use the largest power of two that is <= the passed in minrate
    minrate = F(2)^minexponent
    ratetogroup = rate -> priortogid(rate, minexponent)

    # construct an empty initial priority table -- we'll reset this in init
    rt = PriorityTable(ratetogroup, zeros(F, 1), minrate, 2 * minrate)

    affecttype = F2 <: Tuple ? F2 : Any
    RSSACRJumpAggregation{typeof(njt), S, F1, affecttype, typeof(brackets), U,
        typeof(dependencies), typeof(bd), typeof(rt),
        typeof(ratetogroup)}(nj, nj, njt, et, crl_bnds, crh_bnds,
        sum_rate, maj, maj_rates, rs, affs!, brackets, sps, dependencies, bd, ulow,
        uhigh, minrate, maxrate,
        rt, ratetogroup, zero(sum_rate))
end

# the fields that `isolated_copy` shares; it copies the others (see `shared_fields`)
function shared_fields(::Type{<:RSSACRJumpAggregation})
    (:ma_jumps, :rates, :affects!, :brackets, :save_positions, :dependencies,
        :bracket_data, :ratetogroup)
end

############################# Required Functions ##############################

# creating the JumpAggregation structure (function wrapper-based constant jumps)
function aggregate(aggregator::RSSACR, u, p, t, end_time, constant_jumps,
        ma_jumps, save_positions; kwargs...)

    # handle constant jumps using function wrappers
    rates, affects! = get_jump_info_fwrappers(u, p, t, constant_jumps)
    brackets = get_jump_bracket_fwrappers(u, p, t, constant_jumps, aggregator)

    build_jump_aggregation(RSSACRJumpAggregation, u, p, t, end_time, ma_jumps,
        rates, affects!, save_positions; u, brackets, kwargs...)
end

# set up a new simulation and calculate the first jump / jump time
function initialize!(p::RSSACRJumpAggregation, integrator, u, params, t::Number)
    p.end_time = integrator.sol.prob.tspan[2]
    fill_scaled_rates!(p.maj_rates, p.ma_jumps, params)
    set_bracketing!(p, u, params, t)

    # setup PriorityTable
    reset!(p.rt)
    for (pid, priority) in enumerate(p.cur_rate_high)
        insert!(p.rt, pid, priority)
    end
    p.sum_rate_err = sum_error_bound(p.sum_rate, numterms(p.rt))

    generate_jumps!(p, integrator, u, params, t)
    nothing
end

# execute one jump, changing the system state
function execute_jumps!(p::RSSACRJumpAggregation, integrator, u, params, t, affects!)
    # execute jump
    u = update_state!(p, integrator, u, affects!)

    # update rates
    @inbounds changed_vars = dependents(p.dependencies.jumptovars_map, p.next_jump)
    update_dependent_rates!(p, changed_vars, u, params, t)
    nothing
end

# calculate the next jump / jump time
function generate_jumps!(p::RSSACRJumpAggregation, integrator, u, params, t)
    # recompute the sums of the rate bounds if rounding may have made them inaccurate
    if sum_rate_inaccurate(p.sum_rate, p.sum_rate_err, numterms(p.rt))
        resum_rates!(p)
    end

    sum_rate = p.sum_rate
    # if no more events possible there is nothing to do
    if nomorejumps!(p, sum_rate)
        return nothing
    end

    (; rt, ma_jumps, rates, cur_rate_high, cur_rate_low) = p
    rng = get_rng(integrator)
    num_majumps = get_num_majumps(ma_jumps)
    rerl = zero(sum_rate)

    jidx = sample(rt, cur_rate_high, rng)
    # sampling finds no jump only if the sums exceed the sum of the rate bounds
    iszero(jidx) && ((sum_rate, jidx) = resum_and_sample!(p, rng))
    rerl += typed_randexp(rng, t)
    nrejected = 0
    while rejectrx(ma_jumps, num_majumps, rates, cur_rate_high, cur_rate_low, rng, u, jidx,
        params, t, p.maj_rates)
        # after a long run of rejections, check if a jump can still come before the end time
        nrejected += 1
        if nrejected == max(16, length(cur_rate_high))
            nrejected = 0
            no_jump_before_end!(p, u, params, t, rerl / sum_rate) && return nothing
        end

        # sample candidate reaction
        jidx = sample(rt, cur_rate_high, rng)
        iszero(jidx) && ((sum_rate, jidx) = resum_and_sample!(p, rng))
        rerl += typed_randexp(rng, t)
    end
    p.next_jump = jidx

    # update time to next jump
    p.next_jump_time = t + rerl / sum_rate
    nothing
end

# Recompute the sums of the rate bounds after sampling found no jump, which happens only if the
# sums exceed the sum of the bounds, and sample again. The recomputed sum is positive, since
# `sum_rate_inaccurate` ruled out that the exact sum is zero. A sample can still miss by
# rounding, since `sum` adds the group sums in another order than `sample` subtracts them,
# but only with probability around `eps` per try, so it repeats until it finds a jump.
@noinline function resum_and_sample!(p::RSSACRJumpAggregation, rng)
    resum_rates!(p)
    jidx = 0
    while iszero(jidx)
        jidx = sample(p.rt, p.cur_rate_high, rng)
    end
    return p.sum_rate, jidx
end

# recompute the priority table's sums from the rate bounds
@noinline function resum_rates!(p::RSSACRJumpAggregation)
    resum!(p.rt, p.cur_rate_high)
    p.sum_rate = groupsum(p.rt)
    p.sum_rate_err = sum_error_bound(p.sum_rate, numterms(p.rt))
    nothing
end

######################## SSA specific helper routines #########################
"""
update bracketing for the species in `changed_vars`, e.g. those the executed jump changed
"""
@inline function update_dependent_rates!(p::RSSACRJumpAggregation, changed_vars,
        u::AbstractVector, params, t)
    # update bracketing intervals
    (; ulow, uhigh) = p
    crhigh = p.cur_rate_high
    sum_rate = p.sum_rate
    sum_rate_err = p.sum_rate_err

    @inbounds for uidx in changed_vars
        uval = u[uidx]
        # if new u value is outside the bracketing interval
        if uval == zero(uval) || uval < ulow[uidx] || uval > uhigh[uidx]
            # update u bracketing interval
            ulow[uidx], uhigh[uidx] = get_spec_brackets(p.bracket_data, uidx, uval)

            # for each dependent jump, update jump rate brackets
            deps = dependents(p.dependencies.vartojumps_map, uidx)
            for jidx in deps
                oldrate = crhigh[jidx]
                p.cur_rate_low[jidx], crhigh[jidx] = get_jump_brackets(
                    jidx, p, u, params, t)

                # update the priority table
                update!(p.rt, jidx, oldrate, crhigh[jidx])
            end
            # bound the rounding error per species: a jump reading several changed species
            # is updated once per species, so its intermediate bounds can far exceed the
            # sums before and after the whole update
            # (each update takes up to four additions or subtractions in the table's sums)
            sum_after = groupsum(p.rt)
            sum_rate_err += update_error_bound(sum_rate, sum_after, 4 * length(deps))
            sum_rate = sum_after
        end
    end

    p.sum_rate_err = sum_rate_err
    p.sum_rate = sum_rate
    nothing
end

@inline function update_dependent_rates!(p::RSSACRJumpAggregation, changed_vars,
        u::SVector, params, t)
    # update bracketing intervals
    crhigh = p.cur_rate_high
    sum_rate = p.sum_rate
    sum_rate_err = p.sum_rate_err

    @inbounds for uidx in changed_vars
        uval = u[uidx]
        # if new u value is outside the bracketing interval
        if uval == zero(uval) || uval < p.ulow[uidx] || uval > p.uhigh[uidx]
            # update u bracketing interval
            ulow, uhigh = get_spec_brackets(p.bracket_data, uidx, uval)
            p.ulow = setindex(p.ulow, ulow, uidx)
            p.uhigh = setindex(p.uhigh, uhigh, uidx)

            # for each dependent jump, update jump rate brackets
            deps = dependents(p.dependencies.vartojumps_map, uidx)
            for jidx in deps
                oldrate = crhigh[jidx]
                p.cur_rate_low[jidx], crhigh[jidx] = get_jump_brackets(
                    jidx, p, u, params, t)

                # update the priority table
                update!(p.rt, jidx, oldrate, crhigh[jidx])
            end
            # bound the rounding error per species: a jump reading several changed species
            # is updated once per species, so its intermediate bounds can far exceed the
            # sums before and after the whole update
            # (each update takes up to four additions or subtractions in the table's sums)
            sum_after = groupsum(p.rt)
            sum_rate_err += update_error_bound(sum_rate, sum_after, 4 * length(deps))
            sum_rate = sum_after
        end
    end

    p.sum_rate_err = sum_rate_err
    p.sum_rate = sum_rate
    nothing
end
