# Rejection SSA Method (RSSA), implementation following:
# Marchetti, Priami and Thanh - Simulation Algorithms for Computational Systems Biology
# Note, this implementation **assumes** jump rate functions are monotone
# functions of the current population sizes (i.e. u)
# requires vartojumps_map and fluct_rates as JumpProblem keywords

mutable struct RSSAJumpAggregation{T, S, F1, F2, VJMAP, JVMAP, BD, U} <:
               AbstractSSAJumpAggregator{T, S, F1, F2}
    next_jump::Int
    prev_jump::Int
    next_jump_time::T
    end_time::T
    cur_rate_low::Vector{T}
    cur_rate_high::Vector{T}
    sum_rate::T
    ma_jumps::S
    rates::F1
    affects!::F2
    save_positions::Tuple{Bool, Bool}
    vartojumps_map::VJMAP
    jumptovars_map::JVMAP
    bracket_data::BD
    ulow::U
    uhigh::U
end

function RSSAJumpAggregation(nj::Int, njt::T, et::T, crs::Vector{T}, sr::T,
        maj::S, rs::F1, affs!::F2, sps::Tuple{Bool, Bool};
        u::U, vartojumps_map = nothing,
        jumptovars_map = nothing,
        bracket_data = nothing, kwargs...) where {T, S, F1, F2, U}
    # a dependency graph is needed and must be provided if there are constant rate jumps
    if vartojumps_map === nothing
        if (get_num_majumps(maj) == 0) || !isempty(rs)
            error("To use the RSSA algorithm a map from variables to dependent jumps must be supplied.")
        else
            vtoj_map = var_to_jumps_map(length(u), maj)
        end
    else
        vtoj_map = vartojumps_map
    end

    if jumptovars_map === nothing
        if (get_num_majumps(maj) == 0) || !isempty(rs)
            error("To use the RSSA algorithm a map from jumps to dependent variables must be supplied.")
        else
            jtov_map = jump_to_vars_map(maj)
        end
    else
        jtov_map = jumptovars_map
    end

    # vectors to store bracketing intervals for jump rates
    crl_bnds = similar(crs)
    crh_bnds = similar(crs)

    # a bracket data structure is needed for updating species populations
    bd = (bracket_data === nothing) ? BracketData{T, eltype(U)}() : bracket_data

    # current bounds on solution
    ulow = similar(u)
    uhigh = similar(u)

    affecttype = F2 <: Tuple ? F2 : Any
    RSSAJumpAggregation{T, S, F1, affecttype, typeof(vtoj_map),
        typeof(jtov_map), typeof(bd), U}(nj, nj, njt, et, crl_bnds,
        crh_bnds, sr, maj, rs, affs!, sps,
        vtoj_map, jtov_map, bd, ulow,
        uhigh)
end

############################# Required Functions ##############################

# creating the JumpAggregation structure (function wrapper-based constant jumps)
function aggregate(aggregator::RSSA, u, p, t, end_time, constant_jumps,
        ma_jumps, save_positions; kwargs...)

    # handle constant jumps using function wrappers
    rates, affects! = get_jump_info_fwrappers(u, p, t, constant_jumps)

    build_jump_aggregation(RSSAJumpAggregation, u, p, t, end_time, ma_jumps,
        rates, affects!, save_positions; u,
        kwargs...)
end

# set up a new simulation and calculate the first jump / jump time
function initialize!(p::RSSAJumpAggregation, integrator, u, params, t)
    p.end_time = integrator.sol.prob.tspan[2]
    set_bracketing!(p, u, params, t)
    generate_jumps!(p, integrator, u, params, t)
    nothing
end

# execute one jump, changing the system state
function execute_jumps!(p::RSSAJumpAggregation, integrator, u, params, t, affects!)
    # execute jump
    u = update_state!(p, integrator, u, affects!)
    update_rates!(p, u, params, t)
    nothing
end

# calculate the next jump / jump time
function generate_jumps!(p::RSSAJumpAggregation, integrator, u, params, t)
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
    if iszero(jidx)
        p.next_jump_time = Inf
        return nothing
    end
    rerl += randexp(rng)
    @inbounds while rejectrx(ma_jumps, num_majumps, rates, cur_rate_high,
        cur_rate_low, rng, u, jidx, params, t)
        # sample candidate reaction
        r = rand(rng) * sum_rate
        jidx = linear_search(cur_rate_high, r)
        rerl += randexp(rng)
    end
    p.next_jump = jidx

    p.next_jump_time = t + rerl / sum_rate
    nothing
end

# alt erlang sampling above
#rerl = one(sum_rate)
#rerl *= rand(p.rng)
#p.next_jump_time = t + (-one(sum_rate) / sum_rate) * log(rerl)

######################## SSA specific helper routines #########################

"""
Update rates
"""
@inline function update_rates!(p::RSSAJumpAggregation, u::AbstractVector, params, t)
    # update bracketing intervals
    (; ulow, uhigh) = p
    sum_rate = p.sum_rate
    crhigh = p.cur_rate_high

    @inbounds for uidx in p.jumptovars_map[p.next_jump]
        uval = u[uidx]

        # if new u value is outside the bracketing interval
        if uval == zero(uval) || uval < ulow[uidx] || uval > uhigh[uidx]
            # update u bracketing interval
            ulow[uidx], uhigh[uidx] = get_spec_brackets(p.bracket_data, uidx, uval)

            # for each dependent jump, update jump rate brackets
            for jidx in p.vartojumps_map[uidx]
                sum_rate -= crhigh[jidx]
                p.cur_rate_low[jidx], crhigh[jidx] = get_jump_brackets(jidx, p, params, t)
                sum_rate += crhigh[jidx]
            end
        end
    end
    p.sum_rate = sum_rate
end

@inline function update_rates!(p::RSSAJumpAggregation, u::SVector, params, t)
    # update bracketing intervals
    sum_rate = p.sum_rate
    crhigh = p.cur_rate_high

    @inbounds for uidx in p.jumptovars_map[p.next_jump]
        uval = u[uidx]

        # if new u value is outside the bracketing interval
        if uval == zero(uval) || uval < p.ulow[uidx] || uval > p.uhigh[uidx]
            # update u bracketing interval
            ulow, uhigh = get_spec_brackets(p.bracket_data, uidx, uval)
            p.ulow = setindex(p.ulow, ulow, uidx)
            p.uhigh = setindex(p.uhigh, uhigh, uidx)

            # for each dependent jump, update jump rate brackets
            for jidx in p.vartojumps_map[uidx]
                sum_rate -= crhigh[jidx]
                p.cur_rate_low[jidx], crhigh[jidx] = get_jump_brackets(jidx, p, params, t)
                sum_rate += crhigh[jidx]
            end
        end
    end
    p.sum_rate = sum_rate
end
