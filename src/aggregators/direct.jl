mutable struct DirectJumpAggregation{T, S, F1, F2, RNG} <:
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
end
function DirectJumpAggregation(nj::Int, njt::T, et::T, crs::Vector{T}, sr::T, maj::S,
        rs::F1, affs!::F2, sps::Tuple{Bool, Bool}, rng::RNG;
        kwargs...) where {T, S, F1, F2, RNG}
    affecttype = F2 <: Tuple ? F2 : Any
    DirectJumpAggregation{T, S, F1, affecttype, RNG}(nj, nj, njt, et, crs, sr, maj, rs,
        affs!, sps, rng)
end

############################# Required Functions #############################

# creating the JumpAggregation structure (tuple-based constant jumps)
function aggregate(aggregator::Direct, u, p, t, end_time, constant_jumps,
        ma_jumps, save_positions, rng; kwargs...)

    # handle constant jumps using tuples
    rates, affects! = get_jump_info_tuples(constant_jumps)

    build_jump_aggregation(DirectJumpAggregation, u, p, t, end_time, ma_jumps,
        rates, affects!, save_positions, rng; kwargs...)
end

# creating the JumpAggregation structure (function wrapper-based constant jumps)
function aggregate(aggregator::DirectFW, u, p, t, end_time, constant_jumps,
        ma_jumps, save_positions, rng; kwargs...)

    # handle constant jumps using function wrappers
    rates, affects! = get_jump_info_fwrappers(u, p, t, constant_jumps)

    build_jump_aggregation(DirectJumpAggregation, u, p, t, end_time, ma_jumps,
        rates, affects!, save_positions, rng; kwargs...)
end

# set up a new simulation and calculate the first jump / jump time
function initialize!(p::DirectJumpAggregation, integrator, u, params, t)
    p.end_time = integrator.sol.prob.tspan[2]
    generate_jumps!(p, integrator, u, params, t)
    nothing
end

# execute one jump, changing the system state
@inline function execute_jumps!(p::DirectJumpAggregation, integrator, u, params, t,
        affects!)
    update_state!(p, integrator, u, affects!)
    nothing
end

# calculate the next jump / jump time
function generate_jumps!(p::DirectJumpAggregation, integrator, u, params, t)
    p.sum_rate, ttnj = time_to_next_jump(p, u, params, t)
    p.next_jump_time = add_fast(t, ttnj)
    @inbounds p.next_jump = searchsortedfirst(p.cur_rates, rand(p.rng) * p.sum_rate)
    nothing
end

######################## SSA specific helper routines ########################

# Fill `cur_rates` with the *raw* (non-cumulative) per-channel rates: mass-action
# rates first (indices `1:get_num_majumps(majumps)`), then constant-jump rates.
# Shared by `Direct`'s `time_to_next_jump` (which then forms the running
# cumulative sum used for channel sampling) and by the StochasticAD bounded SSA
# path (which uses the raw rates directly). Tuple rates use the type-stable
# recursive `fill_cur_rates`; function-wrapper rates use a plain loop. This is the
# one place per-channel rates are computed, so a generic (e.g. StochasticTriple)
# rate type flows through it unchanged.
@inline function fill_cur_rates!(cur_rates, u, p, t, majumps, rates::Tuple)
    nma = get_num_majumps(majumps)
    @inbounds for i in 1:nma
        cur_rates[i] = evalrxrate(u, i, majumps)
    end
    isempty(rates) || fill_cur_rates(u, p, t, cur_rates, nma + 1, rates...)
    nothing
end

@inline function fill_cur_rates!(cur_rates, u, p, t, majumps, rates::AbstractArray)
    nma = get_num_majumps(majumps)
    @inbounds for i in 1:nma
        cur_rates[i] = evalrxrate(u, i, majumps)
    end
    @inbounds for k in eachindex(rates)
        cur_rates[nma + k] = rates[k](u, p, t)
    end
    nothing
end

# tuple-based constant jumps
function time_to_next_jump(p::DirectJumpAggregation{T, S, F1}, u, params,
        t) where {T, S, F1 <: Tuple}
    cur_rates = p.cur_rates
    fill_cur_rates!(cur_rates, u, params, t, p.ma_jumps, p.rates)

    # form the running cumulative sum used by `generate_jumps!` for channel sampling
    prev_rate = zero(t)
    @inbounds for i in eachindex(cur_rates)
        cur_rates[i] = add_fast(cur_rates[i], prev_rate)
        prev_rate = cur_rates[i]
    end

    @inbounds sum_rate = cur_rates[end]
    sum_rate, randexp(p.rng) / sum_rate
end

@inline function fill_cur_rates(u, p, t, cur_rates, idx, rate, rates...)
    @inbounds cur_rates[idx] = rate(u, p, t)
    idx += 1
    fill_cur_rates(u, p, t, cur_rates, idx, rates...)
end

@inline function fill_cur_rates(u, p, t, cur_rates, idx, rate)
    @inbounds cur_rates[idx] = rate(u, p, t)
    nothing
end

# function wrapper-based constant jumps
function time_to_next_jump(p::DirectJumpAggregation{T, S, F1}, u, params,
        t) where {T, S, F1 <: AbstractArray}
    cur_rates = p.cur_rates
    fill_cur_rates!(cur_rates, u, params, t, p.ma_jumps, p.rates)

    # form the running cumulative sum used by `generate_jumps!` for channel sampling
    prev_rate = zero(t)
    @inbounds for i in eachindex(cur_rates)
        cur_rates[i] = add_fast(cur_rates[i], prev_rate)
        prev_rate = cur_rates[i]
    end

    @inbounds sum_rate = cur_rates[end]
    sum_rate, randexp(p.rng) / sum_rate
end
