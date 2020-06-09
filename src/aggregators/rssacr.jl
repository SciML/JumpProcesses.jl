"""
Composition-Rejection with Rejection sampling method (RSSA-CR)
"""

const MINJUMPRATE = 2.0^exponent(1e-12)

mutable struct RSSACRJumpAggregation{F,U,S,F1,F2,RNG,VJMAP,JVMAP,BD,T2V,P<:PriorityTable,W<:Function} <: AbstractSSAJumpAggregator
    next_jump::Integer
    next_jump_time::F
    end_time::F
    cur_rate_low::Vector{F}
    cur_rate_high::Vector{F}
    sum_rate::F
    ma_jumps::S
    rates::F1
    affects!::F2
    save_positions::Tuple{Bool,Bool}
    rng::RNG
    vartojumps_map::VJMAP
    jumptovars_map::JVMAP
    bracket_data::BD
    ulow::T2V
    uhigh::T2V
    minrate::F
    maxrate::F   # initial maxrate only, table can increase beyond it!
    rt::P #rate table
    ratetogroup::W
    cur_u_bnds::Matrix{U} # current bounds on state u
  end

function RSSACRJumpAggregation(nj::Int, njt::F, et::F, crs::Vector{F}, sum_rate::F, maj::S, rs::F1, affs!::F2, sps::Tuple{Bool,Bool}, rng::RNG; u::U, vartojumps_map=nothing, jumptovars_map=nothing, bracket_data=nothing, minrate=convert(F,MINJUMPRATE), maxrate=convert(F,Inf), kwargs...) where {F,S,F1,F2,RNG,U}
    # a dependency graph is needed and must be provided if there are constant rate jumps
    if vartojumps_map === nothing
        error("To use the RSSA algorithm a map from variables to depedent jumps must be supplied.")
    else
        vtoj_map = vartojumps_map
    end
    if jumptovars_map === nothing
        error("To use the RSSA algorithm a map from jumps to dependent variables must be supplied.")
    else
        jtov_map = jumptovars_map
    end

    # vectors to store bracketing intervals for jump rates
    crl_bnds = similar(crs)
    crh_bnds = similar(crs)

    # a bracket data structure is needed for updating species populations
    bd = (bracket_data === nothing) ? BracketData{F,eltype(U)}() : bracket_data

    # matrix to store bracketing interval for species and the relative interval width
    # first row is Xlow, second is Xhigh
    cs_bnds = Matrix{eltype(U)}(undef, 2, length(u))
    ulow    = @view cs_bnds[1,:]
    uhigh   = @view cs_bnds[2,:]

    # mapping from jump rate to group id
    minexponent = exponent(minrate)

    # use the largest power of two that is <= the passed in minrate
    minrate = 2.0^minexponent
    ratetogroup = rate -> priortogid(rate, minexponent)

    # construct an empty initial priority table -- we'll overwrite this in init anyways...
    rt = PriorityTable{F,Int,Int,typeof(ratetogroup)}(minrate, 2*minrate, Vector{PriorityGroup{F,Vector{Int}}}(), Vector{F}(), zero(F), Vector{Tuple{Int,Int}}(), ratetogroup)

    RSSACRJumpAggregation{typeof(njt),eltype(U),S,F1,F2,RNG,typeof(vtoj_map),typeof(jtov_map),typeof(bd),typeof(ulow),typeof(rt),typeof(ratetogroup)}(nj, njt, et, crl_bnds, crh_bnds, sum_rate, maj, rs, affs!, sps, rng, vtoj_map, jtov_map, bd, ulow, uhigh, minrate, maxrate, rt, ratetogroup, cs_bnds)
end

########### The following routines should be templates for all SSAs ###########

# condition for jump to occur
@inline function (p::RSSACRJumpAggregation)(u, t, integrator)
    p.next_jump_time == t
end

# executing jump at the next jump time
function (p::RSSACRJumpAggregation)(integrator)
    execute_jumps!(p, integrator, integrator.u, integrator.p, integrator.t)
    generate_jumps!(p, integrator.u, integrator.p, integrator.t)
    register_next_jump_time!(integrator, p, integrator.t)
    nothing
end

# setting up a new simulation
function (p::RSSACRJumpAggregation)(dj, u, t, integrator) # initialize
    initialize!(p, integrator, u, integrator.p, t)
    register_next_jump_time!(integrator, p, t)
    nothing
end

############################# Required Functions ##############################

# creating the JumpAggregation structure (function wrapper-based constant jumps)
function aggregate(aggregator::RSSACR, u, p, t, end_time, constant_jumps,
                   ma_jumps, save_positions, rng; kwargs...)

    # handle constant jumps using function wrappers
    rates, affects! = get_jump_info_fwrappers(u, p, t, constant_jumps)

    build_jump_aggregation(RSSACRJumpAggregation, u, p, t, end_time, ma_jumps,
                           rates, affects!, save_positions, rng; u=u, kwargs...)
end

# set up a new simulation and calculate the first jump / jump time
function initialize!(p::RSSACRJumpAggregation, integrator, u, params, t)
    set_bracketing!(p,u,params,t)

    # if no maxrate was set, use largest initial rate (pad by 2 for an extra group)
    isinf(p.maxrate) && (p.maxrate = 2*maximum(p.cur_rate_high))

    # setup PriorityTable
    p.rt = PriorityTable(p.ratetogroup, p.cur_rate_high, p.minrate, p.maxrate)

    generate_jumps!(p, u, params, t)
    nothing
end

# execute one jump, changing the system state
function execute_jumps!(p::RSSACRJumpAggregation, integrator, u, params, t)
    # execute jump
    u = update_state!(p, integrator, u)

    # update rates
    update_dependent_rates!(p, u, params, t)
    nothing
end

# calculate the next jump / jump time
function generate_jumps!(p::RSSACRJumpAggregation, u, params, t)
    sum_rate = p.sum_rate
    # if no more events possible there is nothing to do
    if nomorejumps!(p, sum_rate)
        return nothing
    end

    @unpack rt, ma_jumps, rates, cur_rate_high, cur_rate_low, rng = p
    rerl = zero(sum_rate)

    jidx    = sample(rt, cur_rate_high, rng)
    rerl   += randexp(rng)
    while rejectrx(ma_jumps, rates, cur_rate_high, cur_rate_low, rng, u, jidx, params, t)
        # sample candidate reaction
        jidx    = sample(rt, cur_rate_high, rng)
        rerl   += randexp(rng)
    end
    p.next_jump = jidx

    # update time to next jump
    p.next_jump_time = t + rerl / sum_rate
    nothing
end


######################## SSA specific helper routines #########################
"update bracketing for species that depend on the just executed jump"
@inline function update_dependent_rates!(p::RSSACRJumpAggregation, u, params, t)
    # update bracketing intervals
    ubnds       = p.cur_u_bnds
    crhigh      = p.cur_rate_high

    @inbounds for uidx in p.jumptovars_map[p.next_jump]
        uval = u[uidx]
        # if new u value is outside the bracketing interval
        if uval == zero(uval) || uval < ubnds[1,uidx] || uval > ubnds[2,uidx]
            # update u bracketing interval
            ubnds[1,uidx], ubnds[2,uidx] = get_spec_brackets(p.bracket_data, uidx, uval)

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
