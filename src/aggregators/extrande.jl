# Define the aggregator. 
struct Extrande <: AbstractAggregatorAlgorithm end

"""
Extrande sampling method for jumps with defined rate bounds.
"""

nullaffect!(integrator) = nothing
const NullAffectJump = ConstantRateJump((u, p, t) -> 0.0, nullaffect!)

mutable struct ExtrandeJumpAggregation{T, S, F1, F2, F3, F4, RNG} <:
               AbstractSSAJumpAggregator
    next_jump::Int
    prev_jump::Int
    next_jump_time::T
    end_time::T
    cur_rates::Vector{T}
    sum_rate::T
    ma_jumps::S
    rate_bnds::F3
    wds::F4
    rates::F1
    affects!::F2
    save_positions::Tuple{Bool, Bool}
    rng::RNG
end

function ExtrandeJumpAggregation(nj::Int, njt::T, et::T, crs::Vector{T}, sr::T, maj::S,
                                 rs::F1, affs!::F2, sps::Tuple{Bool, Bool}, rng::RNG;
                                 rate_bounds::F3, windows::F4,
                                 kwargs...) where {T, S, F1, F2, F3, F4, RNG}
    ExtrandeJumpAggregation{T, S, F1, F2, F3, F4, RNG}(nj, nj, njt, et, crs, sr, maj,
                                                       rate_bounds, windows, rs, affs!, sps,
                                                       rng)
end

############################# Required Functions ##############################
function aggregate(aggregator::Extrande, u, p, t, end_time, constant_jumps,
                   ma_jumps, save_positions, rng; variable_jumps = (), kwargs...)
    rates, affects! = get_jump_info_fwrappers(u, p, t,
                                              (constant_jumps..., variable_jumps...,
                                               NullAffectJump))
    rbnds, wnds = get_va_jump_bound_info_fwrapper(u, p, t,
                                                  (constant_jumps..., variable_jumps...,
                                                   NullAffectJump))
    build_jump_aggregation(ExtrandeJumpAggregation, u, p, t, end_time, ma_jumps,
                           rates, affects!, save_positions, rng; u = u, rate_bounds = rbnds,
                           windows = wnds, kwargs...)
end

# set up a new simulation and calculate the first jump / jump time
function initialize!(p::ExtrandeJumpAggregation, integrator, u, params, t)
    p.end_time = integrator.sol.prob.tspan[2]
    generate_jumps!(p, integrator, u, params, t)
end

# execute one jump, changing the system state
@inline function execute_jumps!(p::ExtrandeJumpAggregation, integrator, u, params, t)
    # execute jump
    u = update_state!(p, integrator, u)
    nothing
end

@fastmath function next_extrande_jump(p::ExtrandeJumpAggregation, u, params, t)
    ttnj = typemax(typeof(t))
    Wmin = typemax(typeof(t))
    Bmax = zero(t)

    prev_rate = zero(t)
    new_rate = zero(t)
    cur_rates = p.cur_rates

    # Mass action rates
    majumps = p.ma_jumps
    idx = get_num_majumps(majumps)

    @inbounds for i in 1:idx
        new_rate = evalrxrate(u, i, majumps)
        cur_rates[i] = add_fast(new_rate, prev_rate)
        prev_rate = cur_rates[i]
        Bmax += prev_rate
    end

    # Calculate the total rate bound and the largest common validity window.
    if !isempty(p.rate_bnds)
        @inbounds for i in 1:length(p.wds)
            Wmin = min(Wmin, p.wds[i](u, params, t))
            Bmax += p.rate_bnds[i](u, params, t)
        end
    end

    # Rejection sampling.
    nextrx = length(cur_rates)
    prop_ttnj = randexp(p.rng) / Bmax
    if prop_ttnj < Wmin
        if !isempty(p.rates)
            idx += 1
            fill_cur_rates(u, params, prop_ttnj + t, p.cur_rates, idx, p.rates...)
            @inbounds for i in idx:length(cur_rates)
                cur_rates[i] = add_fast(cur_rates[i], prev_rate)
                prev_rate = cur_rates[i]
            end
        end
        UBmax = rand(p.rng) * Bmax
        ttnj = prop_ttnj
        if p.cur_rates[end] ≥ UBmax
            nextrx = searchsortedfirst(p.cur_rates, UBmax)
        end
    else
        ttnj = Wmin
    end

    return nextrx, ttnj
end

function generate_jumps!(p::ExtrandeJumpAggregation, integrator, u, params, t)
    nextexj, ttnexj = next_extrande_jump(p, u, params, t)
    p.next_jump = nextexj
    p.next_jump_time = t + ttnexj

    nothing
end
