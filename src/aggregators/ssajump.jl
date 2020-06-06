"""
An aggregator interface for SSA-like algorithms.

### Required Fields
- `next_jump`
- `next_jump_time`
- `end_time`
- `cur_rates`
- `sum_rate`
- `ma_jumps`
- `rates`
- `affects!`
- `save_positions`
- `rng`
"""
abstract type AbstractSSAJumpAggregator <: AbstractJumpAggregator end

DiscreteCallback(c::AbstractSSAJumpAggregator) = DiscreteCallback(c, c, initialize = c, save_positions = c.save_positions)

########### The following routines should be templates for all SSAs ###########

# # condition for jump to occur
# @inline function (p::SSAJumpAggregator)(u, t, integrator)
#   p.next_jump_time == t
# end

# # executing jump at the next jump time
# function (p::SSAJumpAggregator)(integrator)
#   execute_jumps!(p, integrator, integrator.u, integrator.p, integrator.t)
#   generate_jumps!(p, integrator, integrator.u, integrator.p, integrator.t)
#   register_next_jump_time!(integrator, p, integrator.t)
#   nothing
# end

# # setting up a new simulation
# function (p::SSAJumpAggregator)(dj, u, t, integrator) # initialize
#   initialize!(p, integrator, u, integrator.p, t)
#   register_next_jump_time!(integrator, p, integrator.t)
#   nothing
# end

############################## Generic Routines ###############################

@inline function register_next_jump_time!(integrator, p::AbstractSSAJumpAggregator, t)
    if p.next_jump_time < p.end_time
        add_tstop!(integrator, p.next_jump_time)
    end
    nothing
end

# helper routine for setting up standard fields of SSA jump aggregations
function build_jump_aggregation(jump_agg_type, u, p, t, end_time, ma_jumps, rates,
                                affects!, save_positions, rng; kwargs...)

  # mass action jumps
    majumps = ma_jumps
    if majumps === nothing
        majumps = MassActionJump(Vector{typeof(t)}(),
                             Vector{Vector{Pair{Int,eltype(u)}}}(),
                             Vector{Vector{Pair{Int,eltype(u)}}}())
    end

  # current jump rates, allows mass action rates and constant jumps
    cur_rates = Vector{typeof(t)}(undef, get_num_majumps(majumps) + length(rates))

    sum_rate = zero(typeof(t))
    next_jump = 0
    next_jump_time = typemax(typeof(t))
    jump_agg_type(next_jump, next_jump_time, end_time, cur_rates, sum_rate,
                majumps, rates, affects!, save_positions, rng; kwargs...)
end

# reevaluate all rates and total rate
function fill_rates_and_sum!(p::AbstractSSAJumpAggregator, u, params, t)
    sum_rate = zero(typeof(p.sum_rate))

  # mass action jumps
    majumps   = p.ma_jumps
    cur_rates = p.cur_rates
    @inbounds for i in 1:get_num_majumps(majumps)
        cur_rates[i] = evalrxrate(u, i, majumps)
        sum_rate    += cur_rates[i]
    end

  # constant rates
    rates = p.rates
    idx   = get_num_majumps(majumps) + 1
    @inbounds for rate in rates
        cur_rates[idx] = rate(u, params, t)
        sum_rate += cur_rates[idx]
        idx += 1
    end

    p.sum_rate = sum_rate
    nothing
end

# recalculate jump rates for jumps that depend on the just executed jump
# requires dependency graph
function update_dependent_rates!(p::AbstractSSAJumpAggregator, u, params, t)
    @inbounds dep_rxs = p.dep_gr[p.next_jump]
    num_majumps = get_num_majumps(p.ma_jumps)
    cur_rates   = p.cur_rates
    sum_rate    = p.sum_rate
    majumps     = p.ma_jumps
    @inbounds for rx in dep_rxs
        sum_rate -= cur_rates[rx]
        @inbounds cur_rates[rx] = calculate_jump_rate(p,u,params,t,rx)
        sum_rate += cur_rates[rx]
    end

    p.sum_rate = sum_rate
    nothing
end

# Update state based on the p.next_jump
@inline function update_state!(p::AbstractSSAJumpAggregator, integrator, u)
    num_ma_rates = get_num_majumps(p.ma_jumps)
    if p.next_jump <= num_ma_rates # is next jump a mass action jump
        if u isa SVector
            integrator.u = executerx(u, p.next_jump, p.ma_jumps)
        else
            @inbounds executerx!(u, p.next_jump, p.ma_jumps)
        end
    else
        idx = p.next_jump - num_ma_rates
        @inbounds p.affects![idx](integrator)
    end
    return integrator.u
end

"check if the rate is 0 and if it is, make the next jump time Inf"
@inline function is_total_rate_zero!(p) :: Bool
    if abs(p.sum_rate < eps(typeof(p.sum_rate)))
        p.next_jump = 0
        p.next_jump_time = convert(typeof(p.sum_rate), Inf)
        return true
    end
    return false
end

"perform linear search of r over array. Output element j s.t. array[j-1] < r <= array[j]. Will error if no such r exists"
@inline function linear_search(array, r)
    jidx = 1
    parsum = array[jidx]
    while parsum < r
        jidx   += 1
        parsum += array[jidx]
    end
    return jidx
end

"perform rejection sampling test"
@inline function rejectrx(p, u, jidx, params, t)
    # rejection test
    @inbounds r2     = rand(p.rng) * p.cur_rate_high[jidx]
    @inbounds crlow  = p.cur_rate_low[jidx]

    @inbounds if crlow > zero(crlow) && r2 <= crlow
        return false
    else
        # calculate actual propensity, split up for type stability
        @inbounds crate = calculate_jump_rate(p,u,params,t,jidx)
        if crate > zero(crate) && r2 <= crate
            return false
        end
    end
    return true
end

"update the jump rate, assuming p.rates is a vector of functions"
@inline function calculate_jump_rate(p, u, params, t, rx)
    ma_jumps = p.ma_jumps
    num_majumps = get_num_majumps(ma_jumps)
    if rx <= num_majumps
        return evalrxrate(u, rx, p.ma_jumps)
    else
        return p.rates[rx-num_majumps](u, params, t)
    end
end
