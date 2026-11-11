# Sorting Direct Method, implementation following McCollum et al,
# "The sorting direct method for stochastic simulation of biochemical systems with varying reaction execution behavior"
# Comp. Bio. and Chem., 30, pg. 39-49 (2006).

mutable struct SortingDirectJumpAggregation{T, S, F1, F2, DEPS} <:
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
    jump_search_order::Vector{Int}
    jump_search_idx::Int
    # bound on the rounding error in `sum_rate`; kept apart from it, since storing the two
    # together measurably slowed each jump
    sum_rate_err::T
end

function SortingDirectJumpAggregation(nj::Int, njt::T, et::T, crs::Vector{T}, sr::T,
        maj::S, rs::F1, affs!::F2, sps::Tuple{Bool, Bool};
        maj_rates = Vector{T}(undef, get_num_majumps(maj)),
        dependencies,
        kwargs...) where {T, S, F1, F2}

    # map jump idx to idx in cur_rates
    jtoidx = collect(1:length(crs))
    affecttype = F2 <: Tuple ? F2 : Any
    SortingDirectJumpAggregation{T, S, F1, affecttype, typeof(dependencies)}(
        nj, nj, njt, et,
        crs, sr, maj, maj_rates, rs,
        affs!, sps,
        dependencies, jtoidx,
        zero(Int), zero(sr))
end

# the fields that `isolated_copy` shares; it copies the others (see `shared_fields`)
function shared_fields(::Type{<:SortingDirectJumpAggregation})
    (:ma_jumps, :rates, :affects!, :save_positions, :dependencies)
end

############################# Required Functions ##############################

# creating the JumpAggregation structure (function wrapper-based constant jumps)
function aggregate(aggregator::SortingDirect, u, p, t, end_time, constant_jumps,
        ma_jumps, save_positions; kwargs...)

    # handle constant jumps using function wrappers
    rates, affects! = get_jump_info_fwrappers(u, p, t, constant_jumps)

    build_jump_aggregation(SortingDirectJumpAggregation, u, p, t, end_time, ma_jumps,
        rates, affects!, save_positions; kwargs...)
end

# set up a new simulation and calculate the first jump / jump time
function initialize!(p::SortingDirectJumpAggregation, integrator, u, params, t::Number)
    p.end_time = integrator.sol.prob.tspan[2]
    fill_scaled_rates!(p.maj_rates, p.ma_jumps, params)
    fill_rates_and_sum!(p, u, params, t)
    p.sum_rate_err = sum_error_bound(p.sum_rate, length(p.cur_rates))
    generate_jumps!(p, integrator, u, params, t)
    nothing
end

# execute one jump, changing the system state
function execute_jumps!(p::SortingDirectJumpAggregation, integrator, u, params, t, affects!)
    # execute jump
    u = update_state!(p, integrator, u, affects!)

    # update search order
    jso = p.jump_search_order
    jsidx = p.jump_search_idx
    if jsidx != 1
        @inbounds tmp = jso[jsidx]
        @inbounds jso[jsidx] = jso[jsidx - 1]
        @inbounds jso[jsidx - 1] = tmp
    end

    # update current jump rates
    @inbounds dep_rxs = dependents(p.dependencies.dep_graph, p.next_jump)
    update_dependent_rates!(p, dep_rxs, u, params, t)
    nothing
end

# calculate the next jump / jump time
function generate_jumps!(p::SortingDirectJumpAggregation, integrator, u, params, t)
    # recompute the sum of the rates if rounding may have made it inaccurate
    if sum_rate_inaccurate(p.sum_rate, p.sum_rate_err, length(p.cur_rates))
        resum_rates!(p, p.cur_rates)
    end
    nomorejumps!(p, p.sum_rate) && return nothing

    rng = get_rng(integrator)
    p.next_jump_time = t + typed_randexp(rng, t) / p.sum_rate

    # search for next jump
    if p.next_jump_time < p.end_time
        cur_rates = p.cur_rates
        numjumps = length(cur_rates)
        jso = p.jump_search_order
        rn = p.sum_rate * rand(rng)
        @inbounds for idx in 1:numjumps
            rn -= cur_rates[jso[idx]]
            if rn < zero(rn)
                p.jump_search_idx = idx
                break
            end
        end

        # the search finds no jump only if the sum exceeds the sum of the rates
        rn >= zero(rn) && resum_and_search!(p, rng)
        @inbounds p.next_jump = jso[p.jump_search_idx]
    end

    nothing
end

# Recompute the sum of the rates after the search found no jump, which happens only if the sum
# exceeds the sum of the rates, and search again. The recomputed sum is positive, since
# `sum_rate_inaccurate` ruled out that the exact sum is zero, and the next jump time, drawn
# with the previous sum, is off by at most its relative tolerance. A search can still miss
# by rounding, since `sum` adds the rates in another order than the search subtracts them,
# but only with probability around `eps` per try, so it repeats until it finds a jump.
@noinline function resum_and_search!(p::SortingDirectJumpAggregation, rng)
    resum_rates!(p, p.cur_rates)
    (; cur_rates, jump_search_order) = p
    while true
        rn = p.sum_rate * rand(rng)
        @inbounds for idx in eachindex(jump_search_order)
            rn -= cur_rates[jump_search_order[idx]]
            if rn < zero(rn)
                p.jump_search_idx = idx
                return nothing
            end
        end
    end
end
