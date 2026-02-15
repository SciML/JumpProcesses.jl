"""
$(TYPEDEF)

An abstract type for aggregators that manage the simulation of `VariableRateJump`s in jump processes.

## Notes

  - In hybrid ODE/SDE systems with general `VariableRateJump`s, `integrator.u` may be an
    `ExtendedJumpArray` for some aggregators.
"""
abstract type VariableRateAggregator end

################################### VR_FRM ####################################

"""
$(TYPEDEF)

A concrete `VariableRateAggregator` implementing a first-reaction method variant for
simulating `VariableRateJump`s. `VR_FRM` (Variable Rate First Reaction Method with
Ordinary Differential Equation) uses a user-selected ODE solver to handle integrating each
jump's intensity / propensity. A callback is also used for each jump to determine when its
integrated intensity reaches a level corresponding to a firing time, and to then execute the
affect associated with the jump at that time.

## Examples

Simulating a birth-death process with `VR_FRM`:

```julia
using JumpProcesses, OrdinaryDiffEq
u0 = [1.0]           # Initial population  
p = [10.0, 0.5]      # [birth rate, death rate]  
tspan = (0.0, 10.0)

# Birth jump: ∅ → X  
birth_rate(u, p, t) = p[1]
birth_affect!(integrator) = (integrator.u[1] += 1; nothing)
birth_jump = VariableRateJump(birth_rate, birth_affect!)

# Death jump: X → ∅  
death_rate(u, p, t) = p[2] * u[1]
death_affect!(integrator) = (integrator.u[1] -= 1; nothing)
death_jump = VariableRateJump(death_rate, death_affect!)

# Problem setup  
oprob = ODEProblem((du, u, p, t) -> du .= 0, u0, tspan, p)
jprob = JumpProblem(oprob, birth_jump, death_jump; vr_aggregator = VR_FRM())
sol = solve(jprob, Tsit5())
```

## Notes

  - Specify `VR_FRM` in a `JumpProblem` via the `vr_aggregator` keyword argument to select
    its use for handling `VariableRateJump`s.
  - While robust, it may be less performant than `VR_Direct` due to its integration of each
    individual jump's intensity, and use of one continuous callback per jump to handle
    detection of jump times and implementation of state changes from that jump.
"""
struct VR_FRM <: VariableRateAggregator end

function configure_jump_problem(prob, vr_aggregator::VR_FRM, jumps, cvrjs;
        rng = DEFAULT_RNG)
    new_prob = extend_problem(prob, cvrjs; rng)
    variable_jump_callback = build_variable_callback(CallbackSet(), 0, cvrjs...; rng)
    return new_prob, variable_jump_callback
end

# extends prob.u0 to an ExtendedJumpArray with Njumps integrated intensity values,
# of type prob.tspan
function extend_u0(prob, Njumps, rng)
    ttype = eltype(prob.tspan)
    u0 = ExtendedJumpArray(prob.u0, [-randexp(rng, ttype) for i in 1:Njumps])
    return u0
end

function extend_problem(prob::DiffEqBase.AbstractDiscreteProblem, jumps; rng = DEFAULT_RNG)
    error("General `VariableRateJump`s require a continuous problem, like an ODE/SDE/DDE/DAE problem. To use a `DiscreteProblem` bounded `VariableRateJump`s must be used. See the JumpProcesses docs.")
end

function extend_problem(prob::DiffEqBase.AbstractODEProblem, jumps; rng = DEFAULT_RNG)
    _f = SciMLBase.unwrapped_f(prob.f)

    if isinplace(prob)
        jump_f = let _f = _f
            function (du::ExtendedJumpArray, u::ExtendedJumpArray, p, t)
                _f(du.u, u.u, p, t)
                update_jumps!(du, u, p, t, length(u.u), jumps...)
            end
        end
    else
        jump_f = let _f = _f
            function (u::ExtendedJumpArray, p, t)
                du = ExtendedJumpArray(_f(u.u, p, t), u.jump_u)
                update_jumps!(du, u, p, t, length(u.u), jumps...)
                return du
            end
        end
    end

    u0 = extend_u0(prob, length(jumps), rng)
    f = ODEFunction{isinplace(prob)}(jump_f; sys = prob.f.sys,
        observed = prob.f.observed)
    remake(prob; f, u0)
end

function extend_problem(prob::DiffEqBase.AbstractSDEProblem, jumps; rng = DEFAULT_RNG)
    _f = SciMLBase.unwrapped_f(prob.f)

    if isinplace(prob)
        jump_f = let _f = _f
            function (du::ExtendedJumpArray, u::ExtendedJumpArray, p, t)
                _f(du.u, u.u, p, t)
                update_jumps!(du, u, p, t, length(u.u), jumps...)
            end
        end
    else
        jump_f = let _f = _f
            function (u::ExtendedJumpArray, p, t)
                du = ExtendedJumpArray(_f(u.u, p, t), u.jump_u)
                update_jumps!(du, u, p, t, length(u.u), jumps...)
                return du
            end
        end
    end

    if prob.noise_rate_prototype === nothing
        jump_g = function (du, u, p, t)
            prob.g(du.u, u.u, p, t)
        end
    else
        jump_g = function (du, u, p, t)
            prob.g(du, u.u, p, t)
        end
    end

    u0 = extend_u0(prob, length(jumps), rng)
    f = SDEFunction{isinplace(prob)}(jump_f, jump_g; sys = prob.f.sys,
        observed = prob.f.observed)
    remake(prob; f, g = jump_g, u0)
end

function extend_problem(prob::DiffEqBase.AbstractDDEProblem, jumps; rng = DEFAULT_RNG)
    _f = SciMLBase.unwrapped_f(prob.f)

    if isinplace(prob)
        jump_f = let _f = _f
            function (du::ExtendedJumpArray, u::ExtendedJumpArray, h, p, t)
                _f(du.u, u.u, h, p, t)
                update_jumps!(du, u, p, t, length(u.u), jumps...)
            end
        end
    else
        jump_f = let _f = _f
            function (u::ExtendedJumpArray, h, p, t)
                du = ExtendedJumpArray(_f(u.u, h, p, t), u.jump_u)
                update_jumps!(du, u, p, t, length(u.u), jumps...)
                return du
            end
        end
    end

    u0 = extend_u0(prob, length(jumps), rng)
    f = DDEFunction{isinplace(prob)}(jump_f; sys = prob.f.sys,
        observed = prob.f.observed)
    remake(prob; f, u0)
end

# Not sure if the DAE one is correct: Should be a residual of sorts
function extend_problem(prob::DiffEqBase.AbstractDAEProblem, jumps; rng = DEFAULT_RNG)
    _f = SciMLBase.unwrapped_f(prob.f)

    if isinplace(prob)
        jump_f = let _f = _f
            function (out, du::ExtendedJumpArray, u::ExtendedJumpArray, h, p, t)
                _f(out, du.u, u.u, h, p, t)
                update_jumps!(out, u, p, t, length(u.u), jumps...)
            end
        end
    else
        jump_f = let _f = _f
            function (du, u::ExtendedJumpArray, h, p, t)
                out = ExtendedJumpArray(_f(du.u, u.u, h, p, t), u.jump_u)
                update_jumps!(du, u, p, t, length(u.u), jumps...)
                return du
            end
        end
    end

    u0 = extend_u0(prob, length(jumps), rng)
    f = DAEFunction{isinplace(prob)}(jump_f, sys = prob.f.sys,
        observed = prob.f.observed)
    remake(prob; f, u0)
end

struct VR_FRMEventCallback{F, RNG}
    idx::Int
    affect!::F
    rng::RNG
end

# condition: (u, t, integrator)
@inline (c::VR_FRMEventCallback)(u, t, integrator) = u.jump_u[c.idx]

# affect: (integrator)
function (c::VR_FRMEventCallback)(integrator)
    c.affect!(integrator)
    integrator.u.jump_u[c.idx] = -randexp(c.rng, typeof(integrator.t))
    nothing
end

# initialize: (cb, u, t, integrator)
function (c::VR_FRMEventCallback)(cb, u, t, integrator)
    integrator.u.jump_u[c.idx] = -randexp(c.rng, typeof(integrator.t))
    integrator.uprev.jump_u[c.idx] = integrator.u.jump_u[c.idx]
    u_modified!(integrator, false)
    nothing
end

function wrap_jump_in_callback(idx, jump; rng = DEFAULT_RNG)
    cb_functor = VR_FRMEventCallback(idx, jump.affect!, rng)
    ContinuousCallback(cb_functor, cb_functor;
        initialize = cb_functor,
        idxs = jump.idxs,
        rootfind = jump.rootfind,
        interp_points = jump.interp_points,
        save_positions = jump.save_positions,
        abstol = jump.abstol,
        reltol = jump.reltol)
end

function build_variable_callback(cb, idx, jump, jumps...; rng = DEFAULT_RNG)
    idx += 1
    new_cb = wrap_jump_in_callback(idx, jump; rng)
    build_variable_callback(CallbackSet(cb, new_cb), idx, jumps...; rng = DEFAULT_RNG)
end

function build_variable_callback(cb, idx, jump; rng = DEFAULT_RNG)
    idx += 1
    CallbackSet(cb, wrap_jump_in_callback(idx, jump; rng))
end

@inline function update_jumps!(du, u, p, t, idx, jump)
    idx += 1
    du[idx] = jump.rate(u.u, p, t)
end

@inline function update_jumps!(du, u, p, t, idx, jump, jumps...)
    idx += 1
    du[idx] = jump.rate(u.u, p, t)
    update_jumps!(du, u, p, t, idx, jumps...)
end

################################### VR_Direct and VR_DirectFW ####################################

"""
$(TYPEDEF)

A concrete `VariableRateAggregator` implementing a direct method-based approach for
simulating `VariableRateJump`s. `VR_Direct` (Variable Rate Direct Callback) efficiently
samples jump times using one continuous callback to integrate the total intensity /
propensity for all `VariableRateJump`s, sample when the next jump occurs, and then sample
which jump occurs at this time.  `VR_DirectFW` a separate FunctionWrapper mode, which
wraps things in FunctionWrappers in cases with large numbers of jumps

## Examples

Simulating a birth-death process with `VR_Direct` (default) and VR_DirectFW:

```julia
using JumpProcesses, OrdinaryDiffEq
u0 = [1.0]           # Initial population  
p = [10.0, 0.5]      # [birth rate, death rate coefficient]  
tspan = (0.0, 10.0)

# Birth jump: ∅ → X  
birth_rate(u, p, t) = p[1]
birth_affect!(integrator) = (integrator.u[1] += 1; nothing)
birth_jump = VariableRateJump(birth_rate, birth_affect!)

# Death jump: X → ∅  
death_rate(u, p, t) = p[2] * u[1]
death_affect!(integrator) = (integrator.u[1] -= 1; nothing)
death_jump = VariableRateJump(death_rate, death_affect!)

# Problem setup  
oprob = ODEProblem((du, u, p, t) -> du .= 0, u0, tspan, p)
jprob = JumpProblem(oprob, birth_jump, death_jump; vr_aggregator = VR_Direct())
sol = solve(jprob, Tsit5())

jprob = JumpProblem(oprob, birth_jump, death_jump; vr_aggregator = VR_DirectFW())
sol = solve(jprob, Tsit5())
```

## Notes

  - `VR_Direct` and `VR_DirectFW` are expected to generally be more performant than `VR_FRM`.
"""
struct VR_Direct <: VariableRateAggregator end
struct VR_DirectFW <: VariableRateAggregator end

mutable struct VR_DirectEventCache{T, RNG, F1, F2}
    prev_time::T
    prev_threshold::T
    current_time::T
    current_threshold::T
    total_rate::T
    rng::RNG
    rate_funcs::F1
    affect_funcs::F2
    cum_rate_sum::Vector{T}
end

function VR_DirectEventCache(
        jumps::JumpSet, ::VR_Direct, prob, ::Type{T}; rng = DEFAULT_RNG) where {T}
    initial_threshold = randexp(rng, T)
    vjumps = jumps.variable_jumps

    # handle vjumps using tuples
    rate_funcs, affect_funcs = get_jump_info_tuples(vjumps)

    cum_rate_sum = Vector{T}(undef, length(vjumps))

    VR_DirectEventCache{T, typeof(rng), typeof(rate_funcs), typeof(affect_funcs)}(zero(T),
        initial_threshold, zero(T), initial_threshold, zero(T), rng, rate_funcs,
        affect_funcs, cum_rate_sum)
end

function VR_DirectEventCache(
        jumps::JumpSet, ::VR_DirectFW, prob, ::Type{T}; rng = DEFAULT_RNG) where {T}
    initial_threshold = randexp(rng, T)
    vjumps = jumps.variable_jumps

    t, u = prob.tspan[1], prob.u0

    # handle vjumps using tuples
    rate_funcs, affect_funcs = get_jump_info_fwrappers(u, prob.p, t, vjumps)

    cum_rate_sum = Vector{T}(undef, length(vjumps))

    VR_DirectEventCache{T, typeof(rng), typeof(rate_funcs), Any}(zero(T),
        initial_threshold, zero(T), initial_threshold, zero(T), rng, rate_funcs,
        affect_funcs, cum_rate_sum)
end

# Initialization function for VR_DirectEventCache
function initialize_vr_direct_cache!(cache::VR_DirectEventCache, u, t, integrator)
    cache.prev_time = zero(integrator.t)
    cache.current_time = zero(integrator.t)
    cache.prev_threshold = randexp(cache.rng, eltype(integrator.t))
    cache.current_threshold = cache.prev_threshold
    cache.total_rate = zero(integrator.t)
    cache.cum_rate_sum .= 0
    nothing
end

@inline function concretize_vr_direct_affects!(cache::VR_DirectEventCache,
        ::I) where {I <: SciMLBase.DEIntegrator}
    if (cache.affect_funcs isa Vector) &&
       !(cache.affect_funcs isa Vector{FunctionWrappers.FunctionWrapper{Nothing, Tuple{I}}})
        AffectWrapper = FunctionWrappers.FunctionWrapper{Nothing, Tuple{I}}
        cache.affect_funcs = AffectWrapper[makewrapper(AffectWrapper, aff)
                                           for aff in cache.affect_funcs]
    end
    nothing
end

@inline function concretize_vr_direct_affects!(cache::VR_DirectEventCache{T, RNG, F1, F2},
        ::I) where {T, RNG, F1, F2 <: Tuple, I <: SciMLBase.DEIntegrator}
    nothing
end

# Wrapper for initialize to match ContinuousCallback signature
function initialize_vr_direct_wrapper(cb::ContinuousCallback, u, t, integrator)
    concretize_vr_direct_affects!(cb.condition, integrator)
    initialize_vr_direct_cache!(cb.condition, u, t, integrator)
    u_modified!(integrator, false)
    nothing
end

# Merge callback parameters across all jumps for VR_Direct
function build_variable_integcallback(cache::VR_DirectEventCache, jumps)
    save_positions = (false, false)
    abstol = jumps[1].abstol
    reltol = jumps[1].reltol

    for jump in jumps
        save_positions = save_positions .|| jump.save_positions
        abstol = min(abstol, jump.abstol)
        reltol = min(reltol, jump.reltol)
    end

    return ContinuousCallback(cache, cache; initialize = initialize_vr_direct_wrapper,
        save_positions, abstol, reltol)
end

function configure_jump_problem(prob, ::VR_Direct, jumps, cvrjs; rng = DEFAULT_RNG)
    new_prob = prob
    cache = VR_DirectEventCache(jumps, VR_Direct(), prob, eltype(prob.tspan); rng)
    variable_jump_callback = build_variable_integcallback(cache, cvrjs)
    return new_prob, variable_jump_callback
end

function configure_jump_problem(prob, ::VR_DirectFW, jumps, cvrjs; rng = DEFAULT_RNG)
    new_prob = prob
    cache = VR_DirectEventCache(jumps, VR_DirectFW(), prob, eltype(prob.tspan); rng)
    variable_jump_callback = build_variable_integcallback(cache, cvrjs)
    return new_prob, variable_jump_callback
end

# recursively evaluate the cumulative sum of the rates for type stability 
@inline function cumsum_rates!(cum_rate_sum, u, p, t, rates)
    cur_sum = zero(eltype(cum_rate_sum))
    cumsum_rates!(cum_rate_sum, u, p, t, 1, cur_sum, rates...)
end

@inline function cumsum_rates!(cum_rate_sum, u, p, t, idx, cur_sum, rate, rates...)
    new_sum = cur_sum + rate(u, p, t)
    @inbounds cum_rate_sum[idx] = new_sum
    idx += 1
    cumsum_rates!(cum_rate_sum, u, p, t, idx, new_sum, rates...)
end

@inline function cumsum_rates!(cum_rate_sum, u, p, t, idx, cur_sum, rate)
    @inbounds cum_rate_sum[idx] = cur_sum + rate(u, p, t)
end

function total_variable_rate(
        cache::VR_DirectEventCache{
            T, RNG, F1, F2}, u, p, t) where {T, RNG, F1, F2}
    (; cum_rate_sum, rate_funcs) = cache
    sum_rate = cumsum_rates!(cum_rate_sum, u, p, t, rate_funcs)
    return sum_rate
end

# how many quadrature points to use (i.e. determines the degree of the quadrature rule)
const NUM_GAUSS_QUAD_NODES = 4

# Condition functor defined directly on the cache
function (cache::VR_DirectEventCache)(u, t, integrator)
    if integrator.t < cache.current_time
        error("integrator.t < cache.current_time. $(integrator.t) < $(cache.current_time). This is not supported in the `VR_Direct` handling")
    end

    if integrator.t != cache.current_time
        cache.prev_time = cache.current_time
        cache.prev_threshold = cache.current_threshold
        cache.current_time = integrator.t
    end

    dt = t - cache.prev_time
    if dt == 0
        return cache.prev_threshold
    end

    p = integrator.p
    rate_increment = zero(t)
    gps = gauss_points[NUM_GAUSS_QUAD_NODES]
    weights = gauss_weights[NUM_GAUSS_QUAD_NODES]
    tmid = (t + cache.prev_time) / 2
    halfdt = dt / 2
    u_τ = first(SciMLBase.get_tmp_cache(integrator))
    for (i, τᵢ) in enumerate(gps)
        τ = halfdt * τᵢ + tmid
        integrator(u_τ, τ)
        total_variable_rate_τ = total_variable_rate(cache, u_τ, p, τ)
        rate_increment += weights[i] * total_variable_rate_τ
    end
    rate_increment *= halfdt

    cache.current_threshold = cache.prev_threshold - rate_increment
    return cache.current_threshold
end

@generated function execute_affect!(cache::VR_DirectEventCache{T, RNG, F1, F2},
        integrator::I, idx) where {T, RNG, F1, F2 <: Tuple, I <: SciMLBase.DEIntegrator}
    quote
        (; affect_funcs) = cache
        Base.Cartesian.@nif $(fieldcount(F2)) i -> (i == idx) i -> (@inbounds affect_funcs[i](integrator)) i -> (@inbounds affect_funcs[fieldcount(F2)](integrator))
    end
end

@inline function execute_affect!(cache::VR_DirectEventCache,
        integrator::I, idx) where {I <: SciMLBase.DEIntegrator}
    (; affect_funcs) = cache
    if affect_funcs isa Vector{FunctionWrappers.FunctionWrapper{Nothing, Tuple{I}}}
        @inbounds affect_funcs[idx](integrator)
    else
        error("Error, invalid affect_funcs type. Expected a vector of function wrappers and got $(typeof(affect_funcs))")
    end
end

# Affect functor defined directly on the cache
function (cache::VR_DirectEventCache)(integrator)
    (; t, u, p) = integrator
    total_variable_rate_sum = total_variable_rate(cache, u, p, t)
    if total_variable_rate_sum <= 0
        cache.total_rate = 0
        return nothing
    end
    cache.total_rate = total_variable_rate_sum

    rng = cache.rng
    r = rand(rng) * total_variable_rate_sum
    @inbounds jump_idx = searchsortedfirst(cache.cum_rate_sum, r)
    execute_affect!(cache, integrator, jump_idx)

    cache.prev_time = t
    cache.current_threshold = randexp(rng)
    cache.prev_threshold = cache.current_threshold
    cache.current_time = t
    return nothing
end
