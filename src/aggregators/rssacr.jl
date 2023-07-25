"""
Composition-Rejection with Rejection sampling method (RSSA-CR)
"""

const MINJUMPRATE = 2.0^exponent(1e-12)

mutable struct RSSACRJumpAggregation{F, S, F1, F2, RNG, U, VJMAP, JVMAP, BD,
                                     P <: PriorityTable, W <: Function} <:
               AbstractSSAJumpAggregator{F, S, F1, F2, RNG}
    next_jump::Int
    prev_jump::Int
    next_jump_time::F
    end_time::F
    cur_rate_low::Vector{F}
    cur_rate_high::Vector{F}
    sum_rate::F
    ma_jumps::S
    rates::F1
    affects!::F2
    save_positions::Tuple{Bool, Bool}
    rng::RNG
    vartojumps_map::VJMAP
    jumptovars_map::JVMAP
    bracket_data::BD
    ulow::U
    uhigh::U
    minrate::F
    maxrate::F   # initial maxrate only, table can increase beyond it!
    rt::P #rate table
    ratetogroup::W
end

function RSSACRJumpAggregation(nj::Int, njt::F, et::F, crs::Vector{F}, sum_rate::F, maj::S,
                               rs::F1, affs!::F2, sps::Tuple{Bool, Bool}, rng::RNG; u::U,
                               vartojumps_map = nothing, jumptovars_map = nothing,
                               bracket_data = nothing, minrate = convert(F, MINJUMPRATE),
                               maxrate = convert(F, Inf),
                               kwargs...) where {F, S, F1, F2, RNG, U}
    # a dependency graph is needed and must be provided if there are constant rate jumps
    if vartojumps_map === nothing
        if (get_num_majumps(maj) == 0) || !isempty(rs)
            error("To use the RSSACR algorithm a map from variables to dependent jumps must be supplied.")
        else
            vtoj_map = var_to_jumps_map(length(u), maj)
        end
    else
        vtoj_map = vartojumps_map
    end

    if jumptovars_map === nothing
        if (get_num_majumps(maj) == 0) || !isempty(rs)
            error("To use the RSSACR algorithm a map from jumps to dependent variables must be supplied.")
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
    bd = (bracket_data === nothing) ? BracketData{F, eltype(U)}() : bracket_data

    # matrix to store bracketing interval for species and the relative interval width
    # first row is Xlow, second is Xhigh
    ulow = similar(u)
    uhigh = similar(u)

    # mapping from jump rate to group id
    minexponent = exponent(minrate)

    # use the largest power of two that is <= the passed in minrate
    minrate = 2.0^minexponent
    ratetogroup = rate -> priortogid(rate, minexponent)

    # construct an empty initial priority table -- we'll reset this in init
    rt = PriorityTable(ratetogroup, zeros(F, 1), minrate, 2 * minrate)

    affecttype = F2 <: Tuple ? F2 : Any
    RSSACRJumpAggregation{typeof(njt), S, F1, affecttype, RNG, U, typeof(vtoj_map),
                          typeof(jtov_map), typeof(bd), typeof(rt),
                          typeof(ratetogroup)}(nj, nj, njt, et, crl_bnds, crh_bnds,
                                               sum_rate, maj, rs, affs!, sps, rng, vtoj_map,
                                               jtov_map, bd, ulow, uhigh, minrate, maxrate,
                                               rt, ratetogroup)
end

############################# Required Functions ##############################

# creating the JumpAggregation structure (function wrapper-based constant jumps)
function aggregate(aggregator::RSSACR, u, p, t, end_time, constant_jumps,
                   ma_jumps, save_positions, rng; kwargs...)

    # handle constant jumps using function wrappers
    rates, affects! = get_jump_info_fwrappers(u, p, t, constant_jumps)

    build_jump_aggregation(RSSACRJumpAggregation, u, p, t, end_time, ma_jumps,
                           rates, affects!, save_positions, rng; u = u, kwargs...)
end

# set up a new simulation and calculate the first jump / jump time
function initialize!(p::RSSACRJumpAggregation, integrator, u, params, t)
    p.end_time = integrator.sol.prob.tspan[2]
    set_bracketing!(p, u, params, t)

    # setup PriorityTable
    reset!(p.rt)
    for (pid, priority) in enumerate(p.cur_rate_high)
        insert!(p.rt, pid, priority)
    end

    generate_jumps!(p, integrator, u, params, t)
    nothing
end

# execute one jump, changing the system state
function execute_jumps!(p::RSSACRJumpAggregation, integrator, u, params, t, affects!)
    # execute jump
    u = update_state!(p, integrator, u, affects!)

    # update rates
    update_dependent_rates!(p, u, params, t)
    nothing
end

# calculate the next jump / jump time
function generate_jumps!(p::RSSACRJumpAggregation, integrator, u, params, t)
    sum_rate = p.sum_rate
    # if no more events possible there is nothing to do
    if nomorejumps!(p, sum_rate)
        return nothing
    end

    @unpack rt, ma_jumps, rates, cur_rate_high, cur_rate_low, rng = p
    num_majumps = get_num_majumps(ma_jumps)
    rerl = zero(sum_rate)

    jidx = sample(rt, cur_rate_high, rng)
    if iszero(jidx)
        p.next_jump_time = Inf
        return nothing
    end
    rerl += randexp(rng)
    while rejectrx(ma_jumps, num_majumps, rates, cur_rate_high, cur_rate_low, rng, u, jidx,
                   params, t)
        # sample candidate reaction
        jidx = sample(rt, cur_rate_high, rng)
        rerl += randexp(rng)
    end
    p.next_jump = jidx

    # update time to next jump
    p.next_jump_time = t + rerl / sum_rate
    nothing
end

######################## SSA specific helper routines #########################
"""
update bracketing for species that depend on the just executed jump
"""
@inline function update_dependent_rates!(p::RSSACRJumpAggregation, u::AbstractVector,
                                         params, t)
    # update bracketing intervals
    @unpack ulow, uhigh = p
    crhigh = p.cur_rate_high

    @inbounds for uidx in p.jumptovars_map[p.next_jump]
        uval = u[uidx]
        # if new u value is outside the bracketing interval
        if uval == zero(uval) || uval < ulow[uidx] || uval > uhigh[uidx]
            # update u bracketing interval
            ulow[uidx], uhigh[uidx] = get_spec_brackets(p.bracket_data, uidx, uval)

            # for each dependent jump, update jump rate brackets
            for jidx in p.vartojumps_map[uidx]
                oldrate = crhigh[jidx]
                p.cur_rate_low[jidx], crhigh[jidx] = get_jump_brackets(jidx, p, params, t)

                # update the priority table
                update!(p.rt, jidx, oldrate, crhigh[jidx])
            end
        end
    end

    p.sum_rate = groupsum(p.rt)
    nothing
end

@inline function update_dependent_rates!(p::RSSACRJumpAggregation, u::SVector, params, t)
    # update bracketing intervals
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
                oldrate = crhigh[jidx]
                p.cur_rate_low[jidx], crhigh[jidx] = get_jump_brackets(jidx, p, params, t)

                # update the priority table
                update!(p.rt, jidx, oldrate, crhigh[jidx])
            end
        end
    end

    p.sum_rate = groupsum(p.rt)
    nothing
end
