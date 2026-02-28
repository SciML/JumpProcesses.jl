"""
An aggregator interface for SSA-like algorithms.

### Required Fields

  - `next_jump`          # the next jump to execute
  - `prev_jump`          # the previous jump that was executed
  - `next_jump_time`     # the time of the next jump
  - `end_time`           # the time to stop a simulation
  - `cur_rates`          # vector of current propensity values
  - `sum_rate`           # sum of current propensity values
  - `ma_jumps`           # any MassActionJumps for the system (scalar form)
  - `rates`              # vector of rate functions for ConstantRateJumps
  - `affects!`           # vector of affect functions for ConstantRateJumps
  - `save_positions`     # tuple for whether to save the jumps before and/or after event
### Optional fields:

  - `dep_gr`             # dependency graph, dep_gr[i] = indices of reactions that should be updated when rx i occurs.
"""
abstract type AbstractSSAJumpAggregator{T, S, F1, F2} <: AbstractJumpAggregator end

function DiscreteCallback(c::AbstractSSAJumpAggregator)
    DiscreteCallback(c, c, initialize = c, save_positions = c.save_positions)
end

########### The following routines are templates for all SSAs ###########
########### Generally they should not need to be overloaded.  ###########

## Users will normally define (see direct.jl for examples):
# aggregate
# initialize!
# execute_jumps!
# generate_jumps!

@inline function makewrapper(::Type{T}, aff) where {T}
    # rewrap existing wrappers
    if aff isa FunctionWrappers.FunctionWrapper
        T(aff.obj[])
    elseif aff isa Function
        T(aff)
    else
        error("Invalid type of affect function, $(typeof(aff)), expected a Function or FunctionWrapper.")
    end
end

@inline function concretize_affects!(p::AbstractSSAJumpAggregator,
        ::I) where {I <: SciMLBase.DEIntegrator}
    if (p.affects! isa Vector) &&
       !(p.affects! isa Vector{FunctionWrappers.FunctionWrapper{Nothing, Tuple{I}}})
        AffectWrapper = FunctionWrappers.FunctionWrapper{Nothing, Tuple{I}}
        p.affects! = AffectWrapper[makewrapper(AffectWrapper, aff) for aff in p.affects!]
    end
    nothing
end

@inline function concretize_affects!(p::AbstractSSAJumpAggregator{T, S, F1, F2},
        ::I) where {T, S, F1, F2 <: Tuple, I <: SciMLBase.DEIntegrator}
    nothing
end

# setting up a new simulation
function (p::AbstractSSAJumpAggregator)(dj, u, t, integrator) # initialize
    concretize_affects!(p, integrator)
    initialize!(p, integrator, u, integrator.p, t)
    register_next_jump_time!(integrator, p, integrator.t)
    u_modified!(integrator, false)
    nothing
end

# condition for jump to occur
@inline function (p::AbstractSSAJumpAggregator)(u, t, integrator)
    p.next_jump_time == t
end

# executing jump at the next jump time
function (p::AbstractSSAJumpAggregator)(integrator::I) where {I <: SciMLBase.DEIntegrator}
    affects! = p.affects!
    if affects! isa Vector{FunctionWrappers.FunctionWrapper{Nothing, Tuple{I}}}
        execute_jumps!(p, integrator, integrator.u, integrator.p, integrator.t, affects!)
    else
        error("Error, invalid affects! type. Expected a vector of function wrappers and got $(typeof(affects!))")
    end
    generate_jumps!(p, integrator, integrator.u, integrator.p, integrator.t)
    register_next_jump_time!(integrator, p, integrator.t)
    nothing
end

function (p::AbstractSSAJumpAggregator{
        T, S, F1, F2})(integrator::SciMLBase.DEIntegrator) where
        {T, S, F1, F2 <: Union{Tuple, Nothing}}
    execute_jumps!(p, integrator, integrator.u, integrator.p, integrator.t, p.affects!)
    generate_jumps!(p, integrator, integrator.u, integrator.p, integrator.t)
    register_next_jump_time!(integrator, p, integrator.t)
    nothing
end

############################## Generic Routines ###############################

"""
    register_next_jump_time!(integrator, p::AbstractSSAJumpAggregator, t)

Adds a `tstop` to the integrator at the next jump time.
"""
@inline function register_next_jump_time!(integrator, p::AbstractSSAJumpAggregator, t)
    if p.next_jump_time < p.end_time
        add_tstop!(integrator, p.next_jump_time)
    end
    nothing
end

"""
    build_jump_aggregation(jump_agg_type, u, p, t, end_time, ma_jumps, rates,
                           affects!, save_positions; kwargs...)

Helper routine for setting up standard fields of SSA jump aggregations.
"""
function build_jump_aggregation(jump_agg_type, u, p, t, end_time, ma_jumps, rates,
        affects!, save_positions; kwargs...)

    # mass action jumps
    majumps = ma_jumps
    if majumps === nothing
        majumps = MassActionJump(Vector{typeof(t)}(),
            Vector{Vector{Pair{Int, eltype(u)}}}(),
            Vector{Vector{Pair{Int, eltype(u)}}}())
    end

    # current jump rates, allows mass action rates and constant jumps
    cur_rates = Vector{typeof(t)}(undef, get_num_majumps(majumps) + length(rates))

    sum_rate = zero(typeof(t))
    next_jump = 0
    next_jump_time = typemax(typeof(t))
    jump_agg_type(next_jump, next_jump_time, end_time, cur_rates, sum_rate,
        majumps, rates, affects!, save_positions; kwargs...)
end

"""
    fill_rates_and_sum!(p::AbstractSSAJumpAggregator, u, params, t)

Reevaluate all rates and their sum.
"""
function fill_rates_and_sum!(p::AbstractSSAJumpAggregator, u, params, t)
    sum_rate = zero(typeof(p.sum_rate))

    # mass action jumps
    majumps = p.ma_jumps
    cur_rates = p.cur_rates
    @inbounds for i in 1:get_num_majumps(majumps)
        cur_rates[i] = evalrxrate(u, i, majumps)
        sum_rate += cur_rates[i]
    end

    # constant rates
    rates = p.rates
    idx = get_num_majumps(majumps) + 1
    @inbounds for rate in rates
        cur_rates[idx] = rate(u, params, t)
        sum_rate += cur_rates[idx]
        idx += 1
    end

    p.sum_rate = sum_rate
    nothing
end

"""
    calculate_jump_rate(ma_jumps, rates, u, params, t, rx)

Recalculate the rate for the jump with index `rx`.
"""
@inline function calculate_jump_rate(ma_jumps, num_majumps, rates, u, params, t, rx)
    if rx <= num_majumps
        return evalrxrate(u, rx, ma_jumps)
    else
        @inbounds return rates[rx - num_majumps](u, params, t)
    end
end

"""
    update_dependent_rates!(p::AbstractSSAJumpAggregator, u, params, t)

Recalculate jump rates for jumps that depend on the just executed jump.

Notes:

  - Intended for methods that have a dependency graph, i.e. define `p.dep_gr`.
"""
function update_dependent_rates!(p::AbstractSSAJumpAggregator, u, params, t)
    @inbounds dep_rxs = p.dep_gr[p.next_jump]
    cur_rates = p.cur_rates
    sum_rate = p.sum_rate
    num_majumps = get_num_majumps(p.ma_jumps)
    @inbounds for rx in dep_rxs
        sum_rate -= cur_rates[rx]
        @inbounds cur_rates[rx] = calculate_jump_rate(p.ma_jumps, num_majumps, p.rates, u,
            params, t, rx)
        sum_rate += cur_rates[rx]
    end

    p.sum_rate = sum_rate
    nothing
end

"""
    update_state!(p::AbstractSSAJumpAggregator, integrator, u)

Execute `p.next_jump`.
"""
@inline function update_state!(p::AbstractSSAJumpAggregator, integrator, u, affects!)
    (; ma_jumps, next_jump) = p
    num_ma_rates = get_num_majumps(ma_jumps)
    if next_jump <= num_ma_rates # is next jump a mass action jump
        if u isa SVector
            integrator.u = executerx(u, next_jump, ma_jumps)
        else
            @inbounds executerx!(u, next_jump, ma_jumps)
        end
    else
        idx = next_jump - num_ma_rates
        @inbounds affects![idx](integrator)
    end

    # save jump that was just executed
    p.prev_jump = next_jump
    return integrator.u
end

@generated function update_state!(p::AbstractSSAJumpAggregator, integrator, u,
        affects!::T) where {T <: Tuple}
    quote
        (; ma_jumps, next_jump) = p
        num_ma_rates = get_num_majumps(ma_jumps)
        if next_jump <= num_ma_rates # is next jump a mass action jump
            if u isa SVector
                integrator.u = executerx(u, next_jump, ma_jumps)
            else
                @inbounds executerx!(u, next_jump, ma_jumps)
            end
        else
            idx = next_jump - num_ma_rates
            Base.Cartesian.@nif $(fieldcount(T)) i->(i == idx) i->(@inbounds affects![i](integrator)) i->(@inbounds affects![fieldcount(T)](integrator))
        end

        # save jump that was just executed
        p.prev_jump = next_jump
        return integrator.u
    end
end

"""
    nomorejumps!(p, sum_rate) :: Bool

Check if the total rate is zero, and if it is, make the next jump time Inf.
"""
@inline function nomorejumps!(p, sum_rate)::Bool
    if sum_rate < eps(typeof(sum_rate))
        p.next_jump = zero(p.next_jump)
        p.next_jump_time = convert(typeof(sum_rate), Inf)
        return true
    end
    return false
end

"""
    linear_search(iterator, r)

Perform linear search for `r` over iterator. Output index j s.t. sum(array[1:j-1])
< r <= sum(array[1:j]), where array=collect(iterator)

Notes:

  - The iterator must have all positive numbers
  - Returns index zero if the search is unsuccessful. Assumes this corresponds to
    the case of an infinite next reaction time and so the jump index does not
    matter.
"""
@inline function linear_search(iterator, r)
    jidx = 0
    idx = 1
    for element in iterator
        r -= element
        if r < zero(r)
            jidx = idx
            break
        end
        idx += 1
    end
    return jidx
end

"""
    rejectrx(ma_jumps, rates, cur_rate_high, cur_rate_low, rng, u, jidx, params, t)

Perform rejection sampling test (used in RSSA methods).
"""
@inline function rejectrx(
        ma_jumps, num_majumps, rates, cur_rate_high, cur_rate_low, rng, u,
        jidx, params, t)
    # rejection test
    @inbounds r2 = rand(rng) * cur_rate_high[jidx]
    @inbounds crlow = cur_rate_low[jidx]

    if crlow > zero(crlow) && r2 <= crlow
        return false
    else
        # calculate actual propensity, split up for type stability
        crate = calculate_jump_rate(ma_jumps, num_majumps, rates, u, params, t, jidx)
        if crate > zero(crate) && r2 <= crate
            return false
        end
    end
    return true
end
