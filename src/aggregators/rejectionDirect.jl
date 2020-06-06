"""
Direct with rejection sampling
"""
mutable struct RDirectJumpAggregation{T,S,F1,F2,RNG,DEPGR} <: AbstractSSAJumpAggregator
  next_jump::Int
  next_jump_time::T
  end_time::T
  cur_rates::Vector{T}
  sum_rate::T
  ma_jumps::S
  rates::F1
  affects!::F2
  save_positions::Tuple{Bool,Bool}
  rng::RNG
  dep_gr::DEPGR
  max_rate::T
end


function RDirectJumpAggregation(nj::Int, njt::T, et::T, crs::Vector{T}, sr::T, maj::S, rs::F1, affs!::F2, sps::Tuple{Bool,Bool}, rng::RNG; num_specs, dep_graph=nothing, kwargs...) where {T,S,F1,F2,RNG,DEPGR}
    # a dependency graph is needed and must be provided if there are constant rate jumps
    if dep_graph === nothing
        if (get_num_majumps(maj) == 0) || !isempty(rs)
            error("To use ConstantRateJumps with the Rejection Direct (RDirect) algorithm a dependency graph must be supplied.")
        else
            dg = make_dependency_graph(num_specs, maj)
        end
    else
        dg = dep_graph
    end

    # make sure each jump depends on itself
    add_self_dependencies!(dg)

    max_rate = maximum(crs)
    return RDirectJumpAggregation{T,S,F1,F2,RNG,DEPGR}(nj, njt, et, crs, sr, maj, rs, affs!, sps, rng, dg, max_rate)


########### The following routines should be templates for all SSAs ###########

# condition for jump to occur
@inline function (p::RDirectJumpAggregation)(u, t, integrator)
  p.next_jump_time == t
end

# executing jump at the next jump time
function (p::RDirectJumpAggregation)(integrator)
  execute_jumps!(p, integrator, integrator.u, integrator.p, integrator.t)
  generate_jumps!(p, integrator, integrator.u, integrator.p, integrator.t)
  register_next_jump_time!(integrator, p, integrator.t)
  nothing
end

# setting up a new simulation
function (p::RDirectJumpAggregation)(dj, u, t, integrator) # initialize
  initialize!(p, integrator, u, integrator.p, t)
  register_next_jump_time!(integrator, p, t)
  nothing
end

############################# Required Functions #############################

# creating the JumpAggregation structure (tuple-based constant jumps)
function aggregate(aggregator::Direct, u, p, t, end_time, constant_jumps,
                    ma_jumps, save_positions, rng; kwargs...)

  # handle constant jumps using tuples
  rates, affects! = get_jump_info_tuples(constant_jumps)

  build_jump_aggregation(RDirectJumpAggregation, u, p, t, end_time, ma_jumps,
                          rates, affects!, save_positions, rng; kwargs...)
end

# creating the JumpAggregation structure (function wrapper-based constant jumps)
function aggregate(aggregator::DirectFW, u, p, t, end_time, constant_jumps,
                    ma_jumps, save_positions, rng; kwargs...)

  # handle constant jumps using function wrappers
  rates, affects! = get_jump_info_fwrappers(u, p, t, constant_jumps)

  build_jump_aggregation(RDirectJumpAggregation, u, p, t, end_time, ma_jumps,
                          rates, affects!, save_positions, rng; kwargs...)
end

# set up a new simulation and calculate the first jump / jump time
function initialize!(p::RDirectJumpAggregation, integrator, u, params, t)
  generate_jumps!(p, integrator, u, params, t)
  nothing
end

# execute one jump, changing the system state
function execute_jumps!(p::RDirectJumpAggregation, integrator, u, params, t)
    # execute jump
    u = update_state!(p, integrator, u)

    # update rates
    update_dependent_rates!(p, u, params, t)
    nothing
end

# calculate the next jump / jump time
function generate_jumps!(p::RDirectJumpAggregation, u, params, t)
    # if no more events possible there is nothing to do
    if is_total_rate_zero!(p)
        return nothing
    end

    @unpack sum_rate, rng, cur_rates, max_rate = p
    r = rand(rng) * length(cur_rates)
    rx = convert(Integer, ceil(r))

    @inbounds while true
        # pick a random element
        pididx = trunc(Int, r)
        pid    = pids[pididx+1]

        # acceptance test
        ( (r - pididx)*max_rate < priorities[pid] ) && break
    end


    jidx    = sample(rt, cur_rate_high, rng)
    notdone = rejectrx(p, u, jidx, params, t)
    rerl   += randexp(rng)
    @inbounds while rejectrx(p, u, jidx, params, t)
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
    majumps     = p.ma_jumps
    num_majumps = get_num_majumps(majumps)
    ubnds       = p.cur_u_bnds
    crlow       = p.cur_rate_low
    crhigh      = p.cur_rate_high
    bd          = p.bracket_data
    ulow        = p.ulow
    uhigh       = p.uhigh
    @inbounds for uidx in p.jumptovars_map[p.next_jump]
        uval = u[uidx]
        # if new u value is outside the bracketing interval
        if uval == zero(uval) || uval < ubnds[1,uidx] || uval > ubnds[2,uidx]
            # update u bracketing interval
            ubnds[1,uidx], ubnds[2,uidx] = get_spec_brackets(bd, uidx, uval)

            # for each dependent jump, update jump rate brackets
            for jidx in p.vartojumps_map[uidx]
                oldrate = crhigh[jidx]
                if jidx <= num_majumps
                    crlow[jidx], crhigh[jidx] = get_majump_brackets(ulow, uhigh, jidx, majumps)
                else
                    j = jidx - num_majumps
                    crlow[jidx], crhigh[jidx] = get_cjump_brackets(ulow, uhigh, p.rates[j], params, t)
                end
                # update the priority table
                update!(p.rt, jidx, oldrate, crhigh[jidx])
            end
        end
    end

    p.sum_rate = groupsum(p.rt)
    nothing
  end
