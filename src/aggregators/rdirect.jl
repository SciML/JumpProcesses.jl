"""
Direct with rejection sampling
"""

mutable struct RDirectJumpAggregation{T, S, F1, F2, DEPS} <:
               AbstractSSAJumpAggregator{T, S, F1, F2}
    next_jump::Int
    prev_jump::Int
    next_jump_time::T
    end_time::T
    cur_rates::Vector{T}
    sum_rate::T
    ma_jumps::S
    maj_rates::Vector{T}
    rates::F1
    affects!::F2
    save_positions::Tuple{Bool, Bool}
    dependencies::DEPS
    max_rate::T
    counter::Int
    counter_threshold::Int
    # bound on the rounding error in `sum_rate`; kept apart from it, since storing the two
    # together measurably slowed each jump
    sum_rate_err::T
end

function RDirectJumpAggregation(nj::Int, njt::T, et::T, crs::Vector{T}, sr::T, maj::S,
        rs::F1, affs!::F2, sps::Tuple{Bool, Bool};
        maj_rates = Vector{T}(undef, get_num_majumps(maj)),
        counter_threshold = length(crs), dependencies,
        kwargs...) where {T, S, F1, F2}
    max_rate = maximum(crs)
    affecttype = F2 <: Tuple ? F2 : Any
    return RDirectJumpAggregation{T, S, F1, affecttype, typeof(dependencies)}(
        nj, nj, njt, et,
        crs, sr, maj, maj_rates, rs,
        affs!, sps,
        dependencies, max_rate, 0,
        Int(counter_threshold), zero(sr))
end

# the fields that `isolated_copy` shares; it copies the others (see `shared_fields`)
function shared_fields(::Type{<:RDirectJumpAggregation})
    (:ma_jumps, :rates, :affects!, :save_positions, :dependencies, :counter_threshold)
end

############################# Required Functions #############################

# creating the JumpAggregation structure (tuple-based constant jumps)
function aggregate(aggregator::RDirect, u, p, t, end_time, constant_jumps,
        ma_jumps, save_positions; kwargs...)

    # handle constant jumps using function wrappers
    rates, affects! = get_jump_info_fwrappers(u, p, t, constant_jumps)

    build_jump_aggregation(RDirectJumpAggregation, u, p, t, end_time, ma_jumps,
        rates, affects!, save_positions; kwargs...)
end

# set up a new simulation and calculate the first jump / jump time
function initialize!(p::RDirectJumpAggregation, integrator, u, params, t::Number)
    p.end_time = integrator.sol.prob.tspan[2]
    fill_scaled_rates!(p.maj_rates, p.ma_jumps, params)
    fill_rates_and_sum!(p, u, params, t)
    p.sum_rate_err = sum_error_bound(p.sum_rate, length(p.cur_rates))
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
    @inbounds dep_rxs = dependents(p.dependencies.dep_graph, p.next_jump)
    update_dependent_rates!(p, dep_rxs, u, params, t)
    nothing
end

"""
calculate the next jump / jump time
"""
function generate_jumps!(p::RDirectJumpAggregation, integrator, u, params, t)
    # recompute the sum of the rates if rounding may have made it inaccurate, and the largest
    # rate with it: when the rates have collapsed, a stale maximum far above every rate would
    # make the rejection loop below practically endless
    if sum_rate_inaccurate(p.sum_rate, p.sum_rate_err, length(p.cur_rates))
        resum_rates!(p, p.cur_rates)
        p.max_rate = maximum(p.cur_rates)
    end

    # if no more events possible there is nothing to do
    sum_rate = p.sum_rate
    if nomorejumps!(p, sum_rate)
        return nothing
    end
    rng = get_rng(integrator)
    (; cur_rates, max_rate) = p

    num_rxs = length(cur_rates)
    counter = 0
    rx = trunc(Int, rand(rng) * num_rxs) + 1
    @inbounds while cur_rates[rx] < rand(rng) * max_rate
        rx = trunc(Int, rand(rng) * num_rxs) + 1
        counter += 1
    end

    p.counter = counter
    p.next_jump = rx
    p.next_jump_time = t + typed_randexp(rng, t) / sum_rate
    nothing
end

######################## SSA specific helper routines #########################
function update_dependent_rates!(p::RDirectJumpAggregation, dep_rxs, u, params, t)
    (; ma_jumps, rates, cur_rates, sum_rate) = p
    num_majumps = get_num_majumps(ma_jumps)
    max_rate_increased = false
    @inbounds for rx in dep_rxs
        @inbounds new_rate = calculate_jump_rate(
            ma_jumps, num_majumps, rates, u, params, t,
            rx, p.maj_rates)
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

    p.sum_rate_err += update_error_bound(p.sum_rate, sum_rate, 2 * length(dep_rxs))
    p.sum_rate = sum_rate
    nothing
end
