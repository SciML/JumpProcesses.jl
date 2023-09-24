mutable struct FRMJumpAggregation{T, S, F1, F2, RNG} <:
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
function FRMJumpAggregation(nj::Int, njt::T, et::T, crs::Vector{T}, sr::T, maj::S, rs::F1,
                            affs!::F2, sps::Tuple{Bool, Bool}, rng::RNG;
                            kwargs...) where {T, S, F1, F2, RNG}
    affecttype = F2 <: Tuple ? F2 : Any
    FRMJumpAggregation{T, S, F1, affecttype, RNG}(nj, nj, njt, et, crs, sr, maj, rs,
                                                  affs!, sps, rng)
end

############################# Required Functions #############################

# creating the JumpAggregation structure (tuple-based constant jumps)
function aggregate(aggregator::FRM, u, p, t, end_time, constant_jumps,
                   ma_jumps, save_positions, rng; kwargs...)

    # handle constant jumps using tuples
    rates, affects! = get_jump_info_tuples(constant_jumps)

    build_jump_aggregation(FRMJumpAggregation, u, p, t, end_time, ma_jumps, rates, affects!,
                           save_positions, rng; kwargs...)
end

# creating the JumpAggregation structure (function wrapper-based constant jumps)
function aggregate(aggregator::FRMFW, u, p, t, end_time, constant_jumps,
                   ma_jumps, save_positions, rng; kwargs...)

    # handle constant jumps using function wrappers
    rates, affects! = get_jump_info_fwrappers(u, p, t, constant_jumps)

    build_jump_aggregation(FRMJumpAggregation, u, p, t, end_time, ma_jumps, rates, affects!,
                           save_positions, rng; kwargs...)
end

# set up a new simulation and calculate the first jump / jump time
function initialize!(p::FRMJumpAggregation, integrator, u, params, t)
    p.end_time = integrator.sol.prob.tspan[2]
    generate_jumps!(p, integrator, u, params, t)
    nothing
end

# execute one jump, changing the system state
@inline function execute_jumps!(p::FRMJumpAggregation, integrator, u, params, t, affects!)
    # execute jump
    update_state!(p, integrator, u, affects!)
    nothing
end

# calculate the next jump / jump time
function generate_jumps!(p::FRMJumpAggregation, integrator, u, params, t)
    nextmaj, ttnmaj = next_ma_jump(p, u, params, t)
    nextcrj, ttncrj = next_constant_rate_jump(p, u, params, t)

    # execute reaction with minimal time
    if ttnmaj < ttncrj
        p.next_jump = nextmaj
        p.next_jump_time = t + ttnmaj
    else
        p.next_jump = nextcrj
        p.next_jump_time = t + ttncrj
    end
    nothing
end

######################## SSA specific helper routines ########################

# mass action jumps
function next_ma_jump(p::FRMJumpAggregation, u, params, t)
    ttnj = typemax(typeof(t))
    nextrx = zero(Int)
    majumps = p.ma_jumps
    @inbounds for i in 1:get_num_majumps(majumps)
        p.cur_rates[i] = evalrxrate(u, i, majumps)
        dt = randexp(p.rng) / p.cur_rates[i]
        if dt < ttnj
            ttnj = dt
            nextrx = i
        end
    end
    nextrx, ttnj
end

# tuple-based constant jumps
function next_constant_rate_jump(p::FRMJumpAggregation{T, S, F1, F2, RNG}, u, params,
                                 t) where {T, S, F1 <: Tuple, F2 <: Tuple, RNG}
    ttnj = typemax(typeof(t))
    nextrx = zero(Int)
    if !isempty(p.rates)
        idx = get_num_majumps(p.ma_jumps) + 1
        fill_cur_rates(u, params, t, p.cur_rates, idx, p.rates...)
        @inbounds for i in idx:length(p.cur_rates)
            dt = randexp(p.rng) / p.cur_rates[i]
            if dt < ttnj
                ttnj = dt
                nextrx = i
            end
        end
    end
    nextrx, ttnj
end

# function wrapper-based constant jumps
function next_constant_rate_jump(p::FRMJumpAggregation{T, S, F1}, u, params,
                                 t) where {T, S, F1 <: AbstractArray}
    ttnj = typemax(typeof(t))
    nextrx = zero(Int)
    if !isempty(p.rates)
        idx = get_num_majumps(p.ma_jumps) + 1
        @inbounds for i in 1:length(p.rates)
            p.cur_rates[idx] = p.rates[i](u, params, t)
            dt = randexp(p.rng) / p.cur_rates[idx]
            if dt < ttnj
                ttnj = dt
                nextrx = idx
            end
            idx += 1
        end
    end
    nextrx, ttnj
end
