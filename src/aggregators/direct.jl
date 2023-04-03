mutable struct DirectJumpAggregation{T, S, F1, F2, RNG} <: AbstractSSAJumpAggregator
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

######################### dispatches for type stablity #######################

@inline function concretize_affects!(p::DirectJumpAggregation, ::I) where {I <: DiffEqBase.DEIntegrator}
    if p.affects! isa Vector{Any}
        AffectWrapper = FunctionWrappers.FunctionWrapper{Nothing, Tuple{I}}
        p.affects! = AffectWrapper[AffectWrapper(aff) for aff in p.affects!]
    end
    nothing
end

@inline function concretize_affects!(p::DirectJumpAggregation{T, S, F1, F2}, ::I) where {T, S, F1, F2 <: Tuple, I <: DiffEqBase.DEIntegrator}
    nothing
end

# executing jump at the next jump time
function (p::DirectJumpAggregation)(integrator::I) where {I <: DiffEqBase.DEIntegrator}
    affects! = p.affects!
    if affects! isa Vector{FunctionWrappers.FunctionWrapper{Nothing, Tuple{I}}}
        execute_jumps!(p, integrator, integrator.u, integrator.p, integrator.t, affects!)
    else
        error("Error, invalid affects! type in $(typeof(p))")
    end
    generate_jumps!(p, integrator, integrator.u, integrator.p, integrator.t)
    register_next_jump_time!(integrator, p, integrator.t)
    nothing
end

function (p::DirectJumpAggregation{T,S,F1,F2})(integrator::I) where {T, S, F1, F2 <: Tuple, I <: DiffEqBase.DEIntegrator}
    execute_jumps!(p, integrator, integrator.u, integrator.p, integrator.t, p.affects!)
    generate_jumps!(p, integrator, integrator.u, integrator.p, integrator.t)
    register_next_jump_time!(integrator, p, integrator.t)
    nothing
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
    rates, affects! = get_jump_info_fwrappers_direct(u, p, t, constant_jumps)

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
@inline function execute_jumps!(p::DirectJumpAggregation, integrator, u, params, t, affects!)
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

# tuple-based constant jumps
function time_to_next_jump(p::DirectJumpAggregation{T, S, F1}, u, params,
                           t) where {T, S, F1 <: Tuple}
    prev_rate = zero(t)
    new_rate = zero(t)
    cur_rates = p.cur_rates

    # mass action rates
    majumps = p.ma_jumps
    idx = get_num_majumps(majumps)
    @inbounds for i in 1:idx
        new_rate = evalrxrate(u, i, majumps)
        cur_rates[i] = add_fast(new_rate, prev_rate)
        prev_rate = cur_rates[i]
    end

    # constant jump rates
    rates = p.rates
    if !isempty(rates)
        idx += 1
        fill_cur_rates(u, params, t, cur_rates, idx, rates...)
        @inbounds for i in idx:length(cur_rates)
            cur_rates[i] = add_fast(cur_rates[i], prev_rate)
            prev_rate = cur_rates[i]
        end
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
    prev_rate = zero(t)
    new_rate = zero(t)
    cur_rates = p.cur_rates

    # mass action rates
    majumps = p.ma_jumps
    idx = get_num_majumps(majumps)
    @inbounds for i in 1:idx
        new_rate = evalrxrate(u, i, majumps)
        cur_rates[i] = add_fast(new_rate, prev_rate)
        prev_rate = cur_rates[i]
    end

    # constant jump rates
    idx += 1
    rates = p.rates
    @inbounds for i in 1:length(p.rates)
        new_rate = rates[i](u, params, t)
        cur_rates[idx] = add_fast(new_rate, prev_rate)
        prev_rate = cur_rates[idx]
        idx += 1
    end

    @inbounds sum_rate = cur_rates[end]
    sum_rate, randexp(p.rng) / sum_rate
end
