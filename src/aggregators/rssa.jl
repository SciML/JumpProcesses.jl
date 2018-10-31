# Rejection SSA Method (RSSA), implementation following:
# Marchetti, Priami and Thanh - Simulation Algorithms for Computational Systems Biology
# Note, this implementation **assumes** jump rate functions are monotone
# functions of the current population sizes (i.e. u)
# requires vartojumps_map and fluct_rates as JumpProblem keywords

mutable struct RSSAJumpAggregation{T,T2,S,F1,F2,RNG,VJMAP,JVMAP} <: AbstractSSAJumpAggregator
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
  end

function RSSAJumpAggregation(nj::Int, njt::T, et::T, crs::Vector{T}, sr::T,
                                      maj::S, rs::F1, affs!::F2, sps::Tuple{Bool,Bool},
                                      rng::RNG; vartojumps_map=nothing, jumptovars_map=nothing,
                                      fluct_rates=nothing, kwargs...) where {T,S,F1,F2,RNG}

    # a dependency graph is needed and must be provided if there are constant rate jumps
    if varstojumps_map == nothing
        error("To use the RSSA algorithm a map from variables to depedent jumps must be supplied.")
    else
        vtoj_map = vartojumps_map
    end

    if jumptovars_map == nothing
        error("To use the RSSA algorithm a map from jumps to dependent variables must be supplied.")
    else
        jtov_map = jumptovars_map
    end

    # vectors to store bracketing intervals for jump rates
    crl_bnds = similar(crs)
    crh_bnds = similar(crs)

    # matrix to store bracketing interval for species and the relative interval width
    # first row is fluct rate, δᵢ, then Xlow then Xhigh
    cs_bnds = Matrix{eltype(frs)}(undef, 3, length(u))

    # fluctuation rates for how big to take the species bracketing interval are required
    if fluct_rates == nothing
        error("To use the RSSA algorithm a vector of fluctuation rates must be supplied.")
    else
        cs_bnds[1,:] .= fluct_rates
    end

    RSSAJumpAggregation{T,S,F1,F2,RNG,typeof(dg),typeof(vtoj_map),typeof(jtov_map)}(
                        nj, njt, et, crl_bnds, crl_bnds, sr, cs_bnds, maj, rs,
                        affs!, sps, rng, vtoj_map, jtov_map)
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
                           rates, affects!, save_positions, rng; kwargs...)
end

# set up a new simulation and calculate the first jump / jump time
function initialize!(p::RSSAJumpAggregation, integrator, u, params, t)

    # species bracketing interval
    ubnds = p.cur_u_bnds
    @inbounds for i = 1:length(u)
        ubnds[2,i],ubnds[3,i] = get_spec_brackets(u[i], ubnds[1,i])
    end

    # reaction rate bracketing interval
    # mass action jumps
    sum_rate = zero(typeof(p.sum_rate))
    majumps  = p.majumps
    crlow    = p.cur_rate_low
    crhigh   = p.cur_rate_high
    ulow     = @view ubnds[2,:]
    uhigh    = @view ubnds[3,:]
    @inbounds for k = 1:get_num_majumps(majumps)
        crlow[k],crhigh[k] = get_majump_brackets(ulow, uhigh, k, majumps)
        sum_rate += crhigh[k]
    end

    # constant rate jumps
    k = get_num_majumps(majumps) + 1
    @inbounds for rate in p.rates
        crlow[k], crhigh[k] = get_cjump_brackets(ulow, uhigh, rate, params, t)
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
    num_ma_rates = get_num_majumps(p.ma_jumps)
    if p.next_jump <= num_ma_rates
        @inbounds executerx!(u, p.next_jump, p.ma_jumps)
    else
        idx = p.next_jump - num_ma_rates
        @inbounds p.affects![idx](integrator)
    end

    # update bracketing intervals
    ubnds       = p.cur_u_bnds
    sum_rate    = p.sum_rate
    crlow       = p.cur_rate_low
    crhigh      = p.cur_rate_high
    majumps     = p.majumps
    num_majumps = get_num_majumps(majumps)
    @inbounds ulow  = @view ubnds[2,:]
    @inbounds uhigh = @view ubnds[3,:]
    @inbounds for uidx in p.jumptovars_map[p.next_jump]
        uval = u[uidx]

        # if new u value is outside the bracketing interval
        if (uval < ubnds[2,uidx]) || (uval > ubnds[3,uidx])
            # update u bracketing interval
            ubnds[2,uidx],ubnds[3,uidx] = get_spec_brackets(u[uidx], ubnds[1,uidx])

            # for each dependent jump, update jump rate brackets
            for jidx in p.varstojumps_map[uidx]
                sum_rate -= crhigh[jidx]
                if jidx <= num_majumps
                    crlow[jidx],crhigh[jidx] = get_majump_brackets(ulow, uhigh, jidx, majumps)
                else
                    j = jidx - num_majumps
                    crlow[jidx],crhigh[jidx] = get_cjump_brackets(ulow, uhigh, p.rates[j], params, t)
                end
                sum_rate += crhigh[jidx]
            end
        end
    end
    p.sum_rate = sum_rate

    nothing
end

# calculate the next jump / jump time
function generate_jumps!(p::RSSAJumpAggregation, integrator, u, params, t)
    @fastmath p.next_jump_time = t + calc_next_jump!(p, u, params, t)
    nothing
end


######################## SSA specific helper routines #########################

# searches for the next reaction using rejection
@fastmath function calc_next_jump!(p, u, params, t)

    # next jump type
    #rprod       = one(typeof(sum_rate))
    rprod       = zero(typeof(sum_rate))
    crlow       = p.cur_rate_low
    crhigh      = p.cur_rate_high
    majumps     = p.majumps
    num_majumps = get_num_majumps(majumps)
    notdone     = true
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
        r2 = rand(p.rng) * crhigh[jidx]
        if r2 <= crlow[jidx]
            notdone = false
        else
            # calculate actual propensity
            crate = (jidx <= num_majumps) ? evalrxrate(u, jidx, majumps) : rate[jidx - num_majumps](u, params, t)
            if r2 <= crate
                notdone = false
            end
        end

        #rprod *= rand(p.rng)
        rprob += randexp(p.rng)
    end
     p.next_jump = jidx

    # return time to next jump
    #return (-one(typeof(p.sum_rate)) / p.sum_rate) * log(rprod)
    return rprob / p.sum_rate
end
