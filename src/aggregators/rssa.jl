# Rejection SSA Method (RSSA), implementation following:
# Marchetti, Priami and Thanh - Simulation Algorithms for Computational Systems Biology
# Note, this implementation **assumes** jump rate functions are monotone
# functions of the current population sizes (i.e. u)
# requires vartojumps_map and fluct_rates as JumpProblem keywords

mutable struct RSSAJumpAggregation{T,T2,S,F1,F2,RNG,VJMAP,JVMAP,BD,T2V} <: AbstractSSAJumpAggregator
    next_jump::Int
    next_jump_time::T
    end_time::T
    cur_rate_low::Vector{T}
    cur_rate_high::Vector{T}
    sum_rate::T
    cur_u_bnds::Matrix{T2}
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
  end

function RSSAJumpAggregation(nj::Int, njt::T, et::T, crs::Vector{T}, sr::T,
                                      maj::S, rs::F1, affs!::F2, sps::Tuple{Bool,Bool},
                                      rng::RNG; u::U, vartojumps_map=nothing, jumptovars_map=nothing,
                                      bracket_data=nothing, kwargs...) where {T,S,F1,F2,RNG,U}

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
    bd = (bracket_data === nothing) ? BracketData{T,eltype(U)}() : bracket_data

    # matrix to store bracketing interval for species and the relative interval width
    # first row is Xlow, second is Xhigh
    cs_bnds = Matrix{eltype(U)}(undef, 2, length(u))
    ulow    = @view cs_bnds[1,:]
    uhigh   = @view cs_bnds[2,:]

    RSSAJumpAggregation{T,eltype(U),S,F1,F2,RNG,typeof(vtoj_map),typeof(jtov_map),typeof(bd),typeof(ulow)}(
                        nj, njt, et, crl_bnds, crh_bnds, sr, cs_bnds, maj, rs,
                        affs!, sps, rng, vtoj_map, jtov_map, bd, ulow, uhigh)
end


########### The following routines should be templates for all SSAs ###########

# condition for jump to occur
@inline function (p::RSSAJumpAggregation)(u, t, integrator)
    p.next_jump_time == t
end

# executing jump at the next jump time
function (p::RSSAJumpAggregation)(integrator)
    execute_jumps!(p, integrator, integrator.u, integrator.p, integrator.t)
    generate_jumps!(p, integrator, integrator.u, integrator.p, integrator.t)
    register_next_jump_time!(integrator, p, integrator.t)
    nothing
end

# setting up a new simulation
function (p::RSSAJumpAggregation)(dj, u, t, integrator) # initialize
    initialize!(p, integrator, u, integrator.p, t)
    register_next_jump_time!(integrator, p, t)
    nothing
end

############################# Required Functions ##############################

# creating the JumpAggregation structure (function wrapper-based constant jumps)
function aggregate(aggregator::RSSA, u, p, t, end_time, constant_jumps,
                   ma_jumps, save_positions, rng; kwargs...)

    # handle constant jumps using function wrappers
    rates, affects! = get_jump_info_fwrappers(u, p, t, constant_jumps)

    build_jump_aggregation(RSSAJumpAggregation, u, p, t, end_time, ma_jumps,
                           rates, affects!, save_positions, rng; u=u,
                           kwargs...)
end

# set up a new simulation and calculate the first jump / jump time
function initialize!(p::RSSAJumpAggregation, integrator, u, params, t)
    set_bracketing!(p, u, params, t)
    generate_jumps!(p, integrator, u, params, t)
    nothing
end

# execute one jump, changing the system state
function execute_jumps!(p::RSSAJumpAggregation, integrator, u, params, t)
    # execute jump
    u = update_state!(p, integrator, u)
    update_rates!(p, u, params, t)
    nothing
end

# calculate the next jump / jump time
@fastmath function generate_jumps!(p::RSSAJumpAggregation, integrator, u, params, t)
    sum_rate = p.sum_rate
    # if no more events possible there is nothing to do
    if nomorejumps!(p, sum_rate)
        return nothing
    end

    # next jump type
    @unpack ma_jumps, rates, cur_rate_high, cur_rate_low, rng = p
    #rerl        = one(sum_rate)
    rerl        = zero(sum_rate)

    r      = rand(rng) * sum_rate
    jidx   = linear_search(cur_rate_high, r)
    rerl  += randexp(rng)
    @inbounds while rejectrx(ma_jumps, rates, cur_rate_high, cur_rate_low, rng, u, jidx, params, t)
        # sample candidate reaction
        r      = rand(rng) * sum_rate
        jidx   = linear_search(cur_rate_high, r)
        #rerl *= rand(p.rng)
        rerl += randexp(rng)
    end
    p.next_jump = jidx

    # update time to next jump
    #p.next_jump_time = t + (-one(sum_rate) / sum_rate) * log(rerl)
    p.next_jump_time = t + rerl / sum_rate

    nothing
end

######################## SSA specific helper routines #########################

"Update rates"
@inline function update_rates!(p::RSSAJumpAggregation, u, params, t)
    # update bracketing intervals
    ubnds       = p.cur_u_bnds
    sum_rate    = p.sum_rate
    crhigh      = p.cur_rate_high

    @inbounds for uidx in p.jumptovars_map[p.next_jump]
        uval = u[uidx]

        # if new u value is outside the bracketing interval
        if uval == zero(uval) || uval < ubnds[1,uidx] || uval > ubnds[2,uidx]
            # update u bracketing interval
            ubnds[1,uidx], ubnds[2,uidx] = get_spec_brackets(p.bracket_data, uidx, uval)

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
