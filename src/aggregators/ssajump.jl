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

  - `dependencies`       # `JumpDependencies` holding the maps used, e.g. `dep_graph`
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

const DESPECIALIZED_PARAMETERS_MSG = """
    The integrator passes the jumps a `SciMLBase.DespecializedParameters`, which this \
    aggregator cannot use: it stores the jump rate functions in `FunctionWrapper`s typed on \
    the problem's parameter type. Integrators do this when the problem's function uses \
    `SciMLBase.AutoDespecialize`, the default for ModelingToolkit `ODEProblem`s. Build the \
    problem with `SciMLBase.FullSpecialize` instead, e.g. \
    `ODEProblem{true, SciMLBase.FullSpecialize}(...)`."""

# `FunctionWrapper`-based aggregators type their rate wrappers on `typeof(prob.p)`, so they
# cannot be called with the `SciMLBase.DespecializedParameters` that supporting integrators
# pass under `SciMLBase.AutoDespecialize`. Without a check, the first rate evaluation throws
# a `MethodError` from inside the wrapper.
holds_rate_wrappers(rates) = false
function holds_rate_wrappers(rates::AbstractVector{<:FunctionWrappers.FunctionWrapper})
    !isempty(rates)
end
uses_rate_wrappers(p::AbstractSSAJumpAggregator) = holds_rate_wrappers(p.rates)

check_parameters(agg, params) = nothing
function check_parameters(agg, ::SciMLBase.DespecializedParameters)
    uses_rate_wrappers(agg) && throw(ArgumentError(DESPECIALIZED_PARAMETERS_MSG))
    nothing
end

# setting up a new simulation
function (p::AbstractSSAJumpAggregator)(dj, u, t, integrator) # initialize
    check_parameters(p, integrator.p)
    concretize_affects!(p, integrator)
    initialize!(p, integrator, u, integrator.p, t)
    register_next_jump_time!(integrator, p, integrator.t)
    derivative_discontinuity!(integrator, false)
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
        execute_tied_jumps!(p, integrator, affects!)
    else
        error("Error, invalid affects! type. Expected a vector of function wrappers and got $(typeof(affects!))")
    end
    nothing
end

function (p::AbstractSSAJumpAggregator{
        T, S, F1, F2})(integrator::SciMLBase.DEIntegrator) where
        {T, S, F1, F2 <: Union{Tuple, Nothing}}
    execute_tied_jumps!(p, integrator, p.affects!)
    nothing
end

# Executes the jump at the current time, then any next jumps whose times round to it: their
# stops would be at the current time, which an ODE or SDE integrator never reaches, so no jump
# would fire again. `SSAStepper` executes such jumps through its own loop, and has its own
# method. The time is read after each affect, which may change it.
#
# After an affect stops the solve, nothing more is executed or scheduled: a new stop would
# restart an ODE or SDE solve. Ties are rare, so the loop over them is kept out of this
# function, which otherwise does what the affect did without it.
function execute_tied_jumps!(p::AbstractSSAJumpAggregator, integrator, affects!)
    execute_jumps!(p, integrator, integrator.u, integrator.p, integrator.t, affects!)
    solve_stopped(integrator) && return nothing
    generate_jumps!(p, integrator, integrator.u, integrator.p, integrator.t)
    if next_jump_tied!(p, integrator)
        execute_more_tied_jumps!(p, integrator, affects!) && return nothing
    end
    register_next_jump_time!(integrator, p, integrator.t)
    nothing
end

# Executes the tied jumps after the first. Each affect starts with a derivative
# discontinuity, as a separate event's would, and the integrator restarts if any affect,
# including the first, leaves it set. Returns whether an affect stopped the solve.
@noinline function execute_more_tied_jumps!(p::AbstractSSAJumpAggregator, integrator,
        affects!)
    discontinuity = integrator.derivative_discontinuity
    while true
        save_between_tied_jumps!(p, integrator)
        derivative_discontinuity!(integrator, true)
        execute_jumps!(p, integrator, integrator.u, integrator.p, integrator.t, affects!)
        discontinuity |= integrator.derivative_discontinuity
        if solve_stopped(integrator)
            derivative_discontinuity!(integrator, discontinuity)
            return true
        end
        generate_jumps!(p, integrator, integrator.u, integrator.p, integrator.t)
        if !next_jump_tied!(p, integrator)
            derivative_discontinuity!(integrator, discontinuity)
            return false
        end
    end
end

# Whether the next jump is also at the current time, and is executed now. Not at the end
# time, where a tied jump's time is really past the end and some aggregators don't select a
# next jump.
function next_jump_tied!(p::AbstractSSAJumpAggregator, integrator)
    t = integrator.t
    p.next_jump_time == t && t < p.end_time
end

# Whether an affect stopped the solve. `terminate!` can set any retcode; this fallback only
# recognizes the default, and integrators that can say more have their own methods.
solve_stopped(integrator) = integrator.sol.retcode === ReturnCode.Terminated

# Saves between tied jumps as the integrator saves around separate events, which it does
# before the first jump and after the last as `save_positions` asks
function save_between_tied_jumps!(p::AbstractSSAJumpAggregator, integrator)
    p.save_positions[2] && savevalues!(integrator, true)
    p.save_positions[1] && savevalues!(integrator, true)
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
    maj_rates = Vector{typeof(t)}(undef, get_num_majumps(majumps))

    sum_rate = zero(typeof(t))
    next_jump = 0
    next_jump_time = typemax(typeof(t))
    jump_agg_type(next_jump, next_jump_time, end_time, cur_rates, sum_rate,
        majumps, rates, affects!, save_positions; maj_rates, kwargs...)
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
    maj_rates = p.maj_rates
    @inbounds for i in 1:get_num_majumps(majumps)
        cur_rates[i] = evalrxrate(u, i, majumps, maj_rates)
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
@inline function calculate_jump_rate(ma_jumps, num_majumps, rates, u, params, t, rx,
        maj_rates)
    if rx <= num_majumps
        return evalrxrate(u, rx, ma_jumps, maj_rates)
    else
        @inbounds return rates[rx - num_majumps](u, params, t)
    end
end

"""
    update_dependent_rates!(p::AbstractSSAJumpAggregator, dep_rxs, u, params, t)

Recalculate the rates of the jumps in `dep_rxs`, the sum of all rates, and the bound on the
sum's rounding error, `p.sum_rate_err`.

Notes:

  - Intended for methods that have a dependency graph, i.e. define `p.dependencies` with a
    `dep_graph`. After a jump executes, `dep_rxs` is that graph's entry for
    `p.next_jump`, the jumps that depend on it.
"""
function update_dependent_rates!(p::AbstractSSAJumpAggregator, dep_rxs, u, params, t)
    cur_rates = p.cur_rates
    sum_rate = p.sum_rate
    num_majumps = get_num_majumps(p.ma_jumps)
    @inbounds for rx in dep_rxs
        sum_rate -= cur_rates[rx]
        @inbounds cur_rates[rx] = calculate_jump_rate(p.ma_jumps, num_majumps, p.rates, u,
            params, t, rx, p.maj_rates)
        sum_rate += cur_rates[rx]
    end

    p.sum_rate_err += update_error_bound(p.sum_rate, sum_rate, 2 * length(dep_rxs))
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

Notes:

  - A running sum of rates must first be recomputed if `sum_rate_inaccurate` says so, since
    rounding can leave it slightly positive or negative when every rate is zero.
"""
@inline function nomorejumps!(p, sum_rate)::Bool
    if sum_rate <= zero(sum_rate)
        p.next_jump = zero(p.next_jump)
        p.next_jump_time = convert(typeof(sum_rate), Inf)
        return true
    end
    return false
end

"""
    update_error_bound(sum_before, sum_after, nops)

Bound on the rounding error that `nops` additions and subtractions add to a running sum of
nonnegative rates when they update some of the rates, taking the sum from `sum_before` to
`sum_after`. As the rates are nonnegative, every intermediate sum is at most
`sum_before + sum_after`, and each operation errs by at most `eps / 2` times its result. Using
`eps` covers the rounding errors already in `sum_before` and `sum_after`, which
`sum_rate_inaccurate` keeps small relative to them.
"""
@inline function update_error_bound(sum_before, sum_after, nops)
    return nops * eps(typeof(sum_after)) * (abs(sum_before) + abs(sum_after))
end

"""
    sum_error_bound(sum_rate, n)

Bound on the rounding error of a sum of `n` nonnegative terms, `sum_rate`, computed in any
order.
"""
@inline sum_error_bound(sum_rate, n) = n * eps(typeof(sum_rate)) * abs(sum_rate)

"""
    sum_rate_inaccurate(sum_rate, sum_rate_err, n) :: Bool

Check if a running sum of `n` nonnegative rates, `sum_rate`, must be recomputed from the rates:
if the bound on its rounding error, `sum_rate_err`, doesn't rule out that the exact sum is
zero, or allows a relative error above `max(sqrt(eps), 4n eps)`. The tolerance stays above the
relative error of a recomputed sum, `n eps`.
"""
@inline function sum_rate_inaccurate(sum_rate, sum_rate_err, n)
    T = typeof(sum_rate)
    reltol = max(sqrt(eps(T)), 4 * n * eps(T))
    return sum_rate_err > reltol * sum_rate
end

"""
    resum_rates!(p, rates)

Recompute `p.sum_rate`, the sum of `rates`, and the bound on its rounding error,
`p.sum_rate_err`.
"""
@noinline function resum_rates!(p::AbstractSSAJumpAggregator, rates)
    sum_rate = sum(rates)
    p.sum_rate = sum_rate
    p.sum_rate_err = sum_error_bound(sum_rate, length(rates))
    nothing
end

"""
    linear_search(iterator, r)

Perform linear search for `r` over iterator. Output index j s.t. sum(array[1:j-1])
< r <= sum(array[1:j]), where array=collect(iterator)

Notes:

  - The iterator must have all positive numbers
  - Returns index zero if the search is unsuccessful, i.e. if `r` is at least the sum of the
    iterator's elements.
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
        jidx, params, t, maj_rates)
    # rejection test
    @inbounds r2 = rand(rng) * cur_rate_high[jidx]
    @inbounds crlow = cur_rate_low[jidx]

    if crlow > zero(crlow) && r2 <= crlow
        return false
    else
        # calculate actual propensity, split up for type stability
        crate = calculate_jump_rate(ma_jumps, num_majumps, rates, u, params, t, jidx,
            maj_rates)
        if crate > zero(crate) && r2 <= crate
            return false
        end
    end
    return true
end

"""
    all_rates_zero(p, u, params, t) :: Bool

Check if every jump's rate is zero at `(u, params, t)`.
"""
function all_rates_zero(p::AbstractSSAJumpAggregator, u, params, t)
    (; ma_jumps, rates, maj_rates) = p
    num_majumps = get_num_majumps(ma_jumps)
    for rx in 1:(num_majumps + length(rates))
        rate = calculate_jump_rate(ma_jumps, num_majumps, rates, u, params, t, rx,
            maj_rates)
        rate > zero(rate) && return false
    end
    return true
end

"""
    no_jump_before_end!(p, u, params, t, Δt) :: Bool

Check, after a long run of rejected candidates in a rejection-based aggregator, the last at
time `t + Δt`, if no jump can come before the end time; if so, set the next jump time to
`Inf` and return `true`. Once a rejected candidate comes at or after the end time, the next
jump comes later still. If every rate is zero no jump comes, but the bounds on the rates can
be positive, and then no candidate is ever accepted.

The next jump time is not the candidate's time, which can be the end time: an ODE or SDE
integrator executes the next jump when it reaches its time, the end time included.
"""
@noinline function no_jump_before_end!(p::AbstractSSAJumpAggregator, u, params, t, Δt)
    tnext = t + Δt
    if tnext >= p.end_time || all_rates_zero(p, u, params, t)
        p.next_jump_time = typemax(tnext)
        return true
    end
    return false
end
