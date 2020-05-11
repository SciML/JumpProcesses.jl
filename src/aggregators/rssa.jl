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

    # species bracketing interval
    ubnds = p.cur_u_bnds
    @inbounds for (i,uval) in enumerate(u)
        ubnds[1,i], ubnds[2,i] = get_spec_brackets(p.bracket_data, i, uval)
    end

    # reaction rate bracketing interval
    # mass action jumps
    sum_rate = zero(p.sum_rate)
    majumps  = p.ma_jumps
    crlow    = p.cur_rate_low
    crhigh   = p.cur_rate_high
    @inbounds for k = 1:get_num_majumps(majumps)
        crlow[k], crhigh[k] = get_majump_brackets(p.ulow, p.uhigh, k, majumps)
        sum_rate += crhigh[k]
    end

    # constant rate jumps
    k = get_num_majumps(majumps) + 1
    @inbounds for rate in p.rates
        crlow[k], crhigh[k] = get_cjump_brackets(p.ulow, p.uhigh, rate, params, t)
        sum_rate += crhigh[k]
        k += 1
    end
    p.sum_rate = sum_rate

    generate_jumps!(p, integrator, u, params, t)
    nothing
end

# execute one jump, changing the system state
function execute_jumps!(p::RSSAJumpAggregation, integrator, u, params, t)
    # execute jump
    majumps     = p.ma_jumps
    num_majumps = get_num_majumps(majumps)
    if p.next_jump <= num_majumps
        if u isa SVector
          integrator.u = executerx(u, p.next_jump, p.ma_jumps)
        else
          @inbounds executerx!(u, p.next_jump, p.ma_jumps)
        end
    else
        idx = p.next_jump - num_majumps
        @inbounds p.affects![idx](integrator)
    end

    # update bracketing intervals
    ubnds       = p.cur_u_bnds
    sum_rate    = p.sum_rate
    crlow       = p.cur_rate_low
    crhigh      = p.cur_rate_high
    bd          = p.bracket_data
    ulow        = p.ulow
    uhigh       = p.uhigh
    @inbounds for uidx in p.jumptovars_map[p.next_jump]
        uval = u[uidx]

        # if new u value is outside the bracketing interval
        if uval == 0 || uval < ubnds[1,uidx] || uval > ubnds[2,uidx]
            # update u bracketing interval
            ubnds[1,uidx], ubnds[2,uidx] = get_spec_brackets(bd, uidx, uval)

            # for each dependent jump, update jump rate brackets
            for jidx in p.vartojumps_map[uidx]
                sum_rate -= crhigh[jidx]
                if jidx <= num_majumps
                    crlow[jidx], crhigh[jidx] = get_majump_brackets(ulow, uhigh, jidx, majumps)
                else
                    j = jidx - num_majumps
                    crlow[jidx], crhigh[jidx] = get_cjump_brackets(ulow, uhigh, p.rates[j], params, t)
                end
                sum_rate += crhigh[jidx]
            end
        end
    end
    p.sum_rate = sum_rate

    nothing
end

# calculate the next jump / jump time
@fastmath function generate_jumps!(p::RSSAJumpAggregation, integrator, u, params, t)

    # next jump type
    sum_rate    = p.sum_rate

    # if no more events possible there is nothing to do
    if sum_rate < eps(sum_rate)
        p.next_jump = 0
        p.next_jump_time = convert(typeof(sum_rate), Inf)
        return
    end

    crlow       = p.cur_rate_low
    crhigh      = p.cur_rate_high
    majumps     = p.ma_jumps
    num_majumps = get_num_majumps(majumps)
    #rerl        = one(sum_rate)
    rerl        = zero(sum_rate)
    notdone     = true
    jidx        = 0
    @inbounds while notdone
        # sample candidate reaction
        r      = rand(p.rng) * sum_rate
        jidx   = 1
        parsum = crhigh[jidx]
        while parsum < r
            jidx   += 1
            parsum += crhigh[jidx]
        end

        # rejection test
        @inbounds r2 = rand(p.rng) * crhigh[jidx]
        @inbounds if crlow[jidx] > zero(crlow[jidx]) && r2 <= crlow[jidx]
            notdone = false
        else
            # calculate actual propensity, split up for type stability
            if jidx <= num_majumps
                @inbounds crate = evalrxrate(u, jidx, majumps)
                if crate > zero(crate) && r2 <= crate
                    notdone = false
                end
            else
                @inbounds crate = p.rates[jidx - num_majumps](u, params, t)
                if crate > zero(crate) && r2 <= crate
                    notdone = false
                end
            end
        end

        #rerl *= rand(p.rng)
        rerl += randexp(p.rng)
    end
    p.next_jump = jidx

    # update time to next jump
    #p.next_jump_time = t + (-one(sum_rate) / sum_rate) * log(rerl)
    p.next_jump_time = t + rerl / sum_rate

    nothing
end
