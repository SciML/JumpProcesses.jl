# Rejection SSA Method (RSSA), implementation following:
# Marchetti, Priami and Thanh - Simulation Algorithms for Computational Systems Biology
# Note, this implementation **assumes** jump rate functions are monotone
# functions of the current population sizes (i.e. u)
# requires vartojumps_map and fluct_rates as JumpProblem keywords

mutable struct RSSAJumpAggregation{T, S, F1, F2, CB, DEPS, BD, U} <:
               AbstractSSAJumpAggregator{T, S, F1, F2}
    next_jump::Int
    prev_jump::Int
    next_jump_time::T
    end_time::T
    cur_rate_low::Vector{T}
    cur_rate_high::Vector{T}
    sum_rate::T
    ma_jumps::S
    maj_rates::Vector{T}
    rates::F1
    affects!::F2
    brackets::CB
    save_positions::Tuple{Bool, Bool}
    dependencies::DEPS
    bracket_data::BD
    ulow::U
    uhigh::U
    # bound on the rounding error in `sum_rate`; kept apart from it, since storing the two
    # together measurably slowed each jump
    sum_rate_err::T
end

function RSSAJumpAggregation(nj::Int, njt::T, et::T, crs::Vector{T}, sr::T,
        maj::S, rs::F1, affs!::F2, sps::Tuple{Bool, Bool};
        maj_rates = Vector{T}(undef, get_num_majumps(maj)),
        u::U, brackets, dependencies,
        bracket_data = nothing, kwargs...) where {T, S, F1, F2, U}
    # vectors to store bracketing intervals for jump rates
    crl_bnds = similar(crs)
    crh_bnds = similar(crs)

    # a bracket data structure is needed for updating species populations
    bd = (bracket_data === nothing) ? BracketData{T, eltype(U)}() : bracket_data

    # current bounds on solution
    ulow = similar(u)
    uhigh = similar(u)

    affecttype = F2 <: Tuple ? F2 : Any
    RSSAJumpAggregation{T, S, F1, affecttype, typeof(brackets), typeof(dependencies),
        typeof(bd), U}(nj, nj, njt, et, crl_bnds,
        crh_bnds, sr, maj, maj_rates, rs, affs!, brackets, sps,
        dependencies, bd, ulow,
        uhigh, zero(sr))
end

# the fields that `isolated_copy` shares; it copies the others (see `shared_fields`)
function shared_fields(::Type{<:RSSAJumpAggregation})
    (:ma_jumps, :rates, :affects!, :brackets, :save_positions, :dependencies,
        :bracket_data)
end

############################# Required Functions ##############################

# creating the JumpAggregation structure (function wrapper-based constant jumps)
function aggregate(aggregator::RSSA, u, p, t, end_time, constant_jumps,
        ma_jumps, save_positions; kwargs...)

    # handle constant jumps using function wrappers
    rates, affects! = get_jump_info_fwrappers(u, p, t, constant_jumps)
    brackets = get_jump_bracket_fwrappers(u, p, t, constant_jumps, aggregator)

    build_jump_aggregation(RSSAJumpAggregation, u, p, t, end_time, ma_jumps,
        rates, affects!, save_positions; u, brackets,
        kwargs...)
end

# set up a new simulation and calculate the first jump / jump time
function initialize!(p::RSSAJumpAggregation, integrator, u, params, t::Number)
    p.end_time = integrator.sol.prob.tspan[2]
    fill_scaled_rates!(p.maj_rates, p.ma_jumps, params)
    set_bracketing!(p, u, params, t)
    p.sum_rate_err = sum_error_bound(p.sum_rate, length(p.cur_rate_high))
    generate_jumps!(p, integrator, u, params, t)
    nothing
end

# execute one jump, changing the system state
function execute_jumps!(p::RSSAJumpAggregation, integrator, u, params, t, affects!)
    # execute jump
    u = update_state!(p, integrator, u, affects!)
    @inbounds changed_vars = dependents(p.dependencies.jumptovars_map, p.next_jump)
    update_rates!(p, changed_vars, u, params, t)
    nothing
end

# an outside change: `deps` are the changed variables, whose brackets may need updating
function update_external_rates!(p::RSSAJumpAggregation, deps, u, params, t, integrator)
    update_rates!(p, deps, u, params, t)
end

# calculate the next jump / jump time
function generate_jumps!(p::RSSAJumpAggregation, integrator, u, params, t)
    # recompute the sum of the rate bounds if rounding may have made it inaccurate
    if sum_rate_inaccurate(p.sum_rate, p.sum_rate_err, length(p.cur_rate_high))
        resum_rates!(p, p.cur_rate_high)
    end

    sum_rate = p.sum_rate
    # if no more events possible there is nothing to do
    if nomorejumps!(p, sum_rate)
        return nothing
    end
    # next jump type
    (; ma_jumps, rates, cur_rate_high, cur_rate_low) = p
    rng = get_rng(integrator)
    num_majumps = get_num_majumps(ma_jumps)
    rerl = zero(sum_rate)

    r = rand(rng) * sum_rate
    jidx = linear_search(cur_rate_high, r)
    # the search finds no jump only if the sum exceeds the sum of the rate bounds
    iszero(jidx) && ((sum_rate, jidx) = resum_and_search!(p, rng))
    rerl += randexp(rng)
    nrejected = 0
    @inbounds while rejectrx(ma_jumps, num_majumps, rates, cur_rate_high,
        cur_rate_low, rng, u, jidx, params, t, p.maj_rates)
        # after a long run of rejections, check if a jump can still come before the end time
        nrejected += 1
        if nrejected == max(16, length(cur_rate_high))
            nrejected = 0
            no_jump_before_end!(p, u, params, t, rerl / sum_rate) && return nothing
        end

        # sample candidate reaction
        r = rand(rng) * sum_rate
        jidx = linear_search(cur_rate_high, r)
        iszero(jidx) && ((sum_rate, jidx) = resum_and_search!(p, rng))
        rerl += randexp(rng)
    end
    p.next_jump = jidx

    p.next_jump_time = t + rerl / sum_rate
    nothing
end

######################## SSA specific helper routines #########################

# Recompute the sum of the rate bounds after a search found no jump, which happens only if the
# sum exceeds the sum of the bounds, and search again. The recomputed sum is positive, since
# `sum_rate_inaccurate` ruled out that the exact sum is zero. A search can still miss by
# rounding, since `sum` adds the bounds in another order than the search subtracts them, but
# only with probability around `eps` per try, so it repeats until it finds a jump.
@noinline function resum_and_search!(p::RSSAJumpAggregation, rng)
    resum_rates!(p, p.cur_rate_high)
    sum_rate = p.sum_rate
    jidx = 0
    while iszero(jidx)
        jidx = linear_search(p.cur_rate_high, rand(rng) * sum_rate)
    end
    return sum_rate, jidx
end

"""
Update rates after the species in `changed_vars` changed
"""
@inline function update_rates!(p::RSSAJumpAggregation, changed_vars, u::AbstractVector,
        params, t)
    # update bracketing intervals
    (; ulow, uhigh) = p
    sum_rate = p.sum_rate
    crhigh = p.cur_rate_high
    sum_rate_err = p.sum_rate_err

    @inbounds for uidx in changed_vars
        uval = u[uidx]

        # if new u value is outside the bracketing interval
        if uval == zero(uval) || uval < ulow[uidx] || uval > uhigh[uidx]
            # update u bracketing interval
            ulow[uidx], uhigh[uidx] = get_spec_brackets(p.bracket_data, uidx, uval)

            # for each dependent jump, update jump rate brackets
            sum_before = sum_rate
            deps = dependents(p.dependencies.vartojumps_map, uidx)
            for jidx in deps
                sum_rate -= crhigh[jidx]
                p.cur_rate_low[jidx], crhigh[jidx] = get_jump_brackets(
                    jidx, p, u, params, t)
                sum_rate += crhigh[jidx]
            end
            # bound the rounding error per species: a jump reading several changed species
            # is updated once per species, so its intermediate bounds can far exceed the
            # sums before and after the whole update
            sum_rate_err += update_error_bound(sum_before, sum_rate, 2 * length(deps))
        end
    end
    p.sum_rate_err = sum_rate_err
    p.sum_rate = sum_rate
end

@inline function update_rates!(p::RSSAJumpAggregation, changed_vars, u::SVector, params, t)
    # update bracketing intervals
    sum_rate = p.sum_rate
    crhigh = p.cur_rate_high
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
            sum_before = sum_rate
            deps = dependents(p.dependencies.vartojumps_map, uidx)
            for jidx in deps
                sum_rate -= crhigh[jidx]
                p.cur_rate_low[jidx], crhigh[jidx] = get_jump_brackets(
                    jidx, p, u, params, t)
                sum_rate += crhigh[jidx]
            end
            # bound the rounding error per species: a jump reading several changed species
            # is updated once per species, so its intermediate bounds can far exceed the
            # sums before and after the whole update
            sum_rate_err += update_error_bound(sum_before, sum_rate, 2 * length(deps))
        end
    end
    p.sum_rate_err = sum_rate_err
    p.sum_rate = sum_rate
end
