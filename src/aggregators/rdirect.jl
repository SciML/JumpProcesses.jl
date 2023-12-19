"""
Direct with rejection sampling
"""

mutable struct RDirectJumpAggregation{T, S, F1, F2, RNG, DEPGR} <:
               AbstractSSAJumpAggregator{T, S, F1, F2, RNG}
    next_jump::Int
    prev_jump::Int
    next_jump_time::T
    end_time::T
    cur_rates::Vector{T}
    sum_rate::T
    ma_jumps::S
    rates::F1
    affects!::F2
    save_positions::Tuple{Bool, Bool}
    rng::RNG
    dep_gr::DEPGR
    max_rate::T
    counter::Int
    counter_threshold::Any
end

function RDirectJumpAggregation(nj::Int, njt::T, et::T, crs::Vector{T}, sr::T, maj::S,
        rs::F1, affs!::F2,
        sps::Tuple{Bool, Bool}, rng::RNG;
        num_specs, counter_threshold = length(crs),
        dep_graph = nothing,
        kwargs...) where {T, S, F1, F2, RNG}
    # a dependency graph is needed and must be provided if there are constant rate jumps
    if dep_graph === nothing
        if (get_num_majumps(maj) == 0) || !isempty(rs)
            error(
                "To use ConstantRateJumps with the Rejection Direct (RDirect) algorithm a dependency graph must be supplied.",
            )
        else
            dg = make_dependency_graph(num_specs, maj)
        end
    else
        dg = dep_graph

        # make sure each jump depends on itself
        add_self_dependencies!(dg)
    end

    max_rate = maximum(crs)
    affecttype = F2 <: Tuple ? F2 : Any
    return RDirectJumpAggregation{T, S, F1, affecttype, RNG, typeof(dg)}(nj, nj, njt, et,
        crs, sr, maj, rs,
        affs!, sps,
        rng,
        dg, max_rate, 0,
        counter_threshold)
end

############################# Required Functions #############################

# creating the JumpAggregation structure (tuple-based constant jumps)
function aggregate(aggregator::RDirect, u, p, t, end_time, constant_jumps,
        ma_jumps, save_positions, rng; kwargs...)

    # handle constant jumps using function wrappers
    rates, affects! = get_jump_info_fwrappers(u, p, t, constant_jumps)

    build_jump_aggregation(RDirectJumpAggregation, u, p, t, end_time, ma_jumps,
        rates,
        affects!, save_positions, rng; num_specs = length(u),
        kwargs...)
end

# set up a new simulation and calculate the first jump / jump time
function initialize!(p::RDirectJumpAggregation, integrator, u, params, t)
    p.end_time = integrator.sol.prob.tspan[2]
    fill_rates_and_sum!(p, u, params, t)
    p.max_rate = maximum(p.cur_rates)
    generate_jumps!(p, integrator, u, params, t)
    nothing
end

"""
execute one jump, changing the system state and updating rates
"""
function execute_jumps!(p::RDirectJumpAggregation, integrator, u, params, t, affects!)
    # execute jump
    u = update_state!(p, integrator, u, affects!)

    # update rates
    update_dependent_rates!(p, u, params, t)
    nothing
end

"""
calculate the next jump / jump time
"""
function generate_jumps!(p::RDirectJumpAggregation, integrator, u, params, t)
    # if no more events possible there is nothing to do
    sum_rate = p.sum_rate
    if nomorejumps!(p, sum_rate)
        return nothing
    end
    @unpack rng, cur_rates, max_rate = p

    num_rxs = length(cur_rates)
    counter = 0
    rx = trunc(Int, rand(rng) * num_rxs) + 1
    @inbounds while cur_rates[rx] < rand(rng) * max_rate
        rx = trunc(Int, rand(rng) * num_rxs) + 1
        counter += 1
    end

    p.counter = counter
    p.next_jump = rx
    p.next_jump_time = t + randexp(p.rng) / sum_rate
    nothing
end

######################## SSA specific helper routines #########################
function update_dependent_rates!(p::RDirectJumpAggregation, u, params, t)
    @inbounds dep_rxs = p.dep_gr[p.next_jump]
    @unpack ma_jumps, rates, cur_rates, sum_rate = p
    num_majumps = get_num_majumps(ma_jumps)
    max_rate_increased = false
    @inbounds for rx in dep_rxs
        @inbounds new_rate = calculate_jump_rate(ma_jumps, num_majumps,
            rates, u, params, t,
            rx)
        sum_rate += new_rate - cur_rates[rx]
        if new_rate > p.max_rate
            p.max_rate = new_rate
            max_rate_increased = true
        end
        cur_rates[rx] = new_rate
    end
    if !max_rate_increased && p.counter > p.counter_threshold
        p.max_rate = maximum(p.cur_rates)
    end

    p.sum_rate = sum_rate
    nothing
end
