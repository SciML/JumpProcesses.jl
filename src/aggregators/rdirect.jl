"""
Direct with rejection sampling
"""

#global num_reactions = 0
#global num_rejections = 0

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
    return RDirectJumpAggregation{T,S,F1,F2,RNG,typeof(dg)}(nj, njt, et, crs, sr, maj, rs, affs!, sps, rng, dg, max_rate)
end

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
function aggregate(aggregator::RDirect, u, p, t, end_time, constant_jumps,
                    ma_jumps, save_positions, rng; kwargs...)

    # handle constant jumps using function wrappers
    rates, affects! = get_jump_info_fwrappers(u, p, t, constant_jumps)

    build_jump_aggregation(RDirectJumpAggregation, u, p, t, end_time, ma_jumps,
                          rates, affects!, save_positions, rng; num_specs = length(u), kwargs...)
end

# set up a new simulation and calculate the first jump / jump time
function initialize!(p::RDirectJumpAggregation, integrator, u, params, t)
    fill_rates_and_sum!(p, u, params, t)
    generate_jumps!(p, integrator, u, params, t)
    nothing
end

"execute one jump, changing the system state and updating rates"
function execute_jumps!(p::RDirectJumpAggregation, integrator, u, params, t)
    # execute jump
    u = update_state!(p, integrator, u)

    # update rates
    update_dependent_rates!(p, u, params, t)
    nothing
end

"calculate the next jump / jump time"
function generate_jumps!(p::RDirectJumpAggregation, integrator, u, params, t)
    # if no more events possible there is nothing to do
    sum_rate = p.sum_rate
    if nomorejumps!(p, sum_rate)
        return nothing
    end
    @unpack rng, cur_rates, max_rate = p

    num_rxs = length(cur_rates)
    rx = trunc(Integer, rand(rng) * num_rxs)+1
    while cur_rates[rx] < rand(rng) * max_rate
        # global num_rejections += 1
        rx = trunc(Integer, rand(rng) * num_rxs)+1
    end
    # global num_reactions += 1

    p.next_jump = rx

    # update time to next jump
    p.next_jump_time = t + randexp(p.rng) / sum_rate
    # if (p.next_jump_time >= p.end_time)
    #     println("Average number of rejections per reaction: $(num_rejections/num_reactions)")
    # end
    nothing
end


######################## SSA specific helper routines #########################
function update_dependent_rates!(p::RDirectJumpAggregation, u, params, t)
    @inbounds dep_rxs = p.dep_gr[p.next_jump]
    @unpack ma_jumps, rates, cur_rates, sum_rate = p
    need_to_reculculate_max_rate = false
    @inbounds for rx in dep_rxs
        sum_rate -= cur_rates[rx]
        @inbounds new_rate = calculate_jump_rate(ma_jumps, rates, u,params,t,rx)
        sum_rate += new_rate
        if new_rate > p.max_rate
            p.max_rate = new_rate
        else if abs(cur_rates[rx]-p.max_rate) < eps(typeof(p.max_rate))
            need_to_reculculate_max_rate = false
        end
        cur_rates[rx] = new_rate
    end
    if need_to_reculculate_max_rate
        p.max_rate = maximum(p.cur_rates)
    end

    p.sum_rate = sum_rate
    nothing
end
