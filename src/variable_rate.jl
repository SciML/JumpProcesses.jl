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
jump's intensity / propensity. Continuous callbacks are also used to determine when each
jump's integrated intensity reaches a level corresponding to a firing time, and to then
execute the affect associated with the jump at that time.

## Constructor

```julia
VR_FRM(; vector_callback = nothing)
```

  - `vector_callback = true`: one `VectorContinuousCallback` handles all the jumps. Jumps
    whose callback options (`rootfind`, `interp_points`, `save_positions`, `abstol` and
    `reltol`) differ are handled by one `VectorContinuousCallback` per set of options.
  - `vector_callback = false`: one `ContinuousCallback` handles each jump. This is faster
    with up to about three jumps, but with more it is slower and takes longer to compile.
  - `vector_callback = nothing` (the default): `JumpProblem` chooses `false` if `VR_FRM`
    handles at most three jumps, and `true` otherwise.

The choice is stored as the type parameter, `VR_FRM{vector_callback}`.

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
    individual jump's intensity, and its detection of each jump's firing time.
"""
struct VR_FRM{V} <: VariableRateAggregator end

VR_FRM(; vector_callback::Union{Nothing, Bool} = nothing) = VR_FRM{vector_callback}()

# per-jump callbacks were faster in benchmarks with up to three jumps
function configure_jump_problem(prob, ::VR_FRM{nothing}, jumps, cvrjs)
    vr_aggregator = VR_FRM{length(cvrjs) > 3}()
    configure_jump_problem(prob, vr_aggregator, jumps, cvrjs)
end

function configure_jump_problem(prob, vr_aggregator::VR_FRM, jumps, cvrjs)
    # a Vector of jumps is converted here, once, so that the extended right-hand side
    # evaluates the rates through the type-stable, unrolled `update_jumps!`
    vrjs = Tuple(cvrjs)
    new_prob = extend_problem(prob, vrjs)
    variable_jump_callback = vr_frm_callbacks(vr_aggregator, length(prob.u0), vrjs)
    return new_prob, variable_jump_callback
end

# extends prob.u0 to an ExtendedJumpArray with Njumps integrated intensity values,
# of type prob.tspan
function extend_u0(prob, Njumps)
    ttype = eltype(prob.tspan)
    u0 = ExtendedJumpArray(prob.u0, zeros(ttype, Njumps))
    return u0
end

# The function that `extend_problem` rebuilds keeps the specialization of `prob.f` if it is
# `FullSpecialize` or `NoSpecialize`. Other policies become the default, `AutoSpecialize`, as
# before: under `AutoDespecialize` and `AutoRespecialize` the integrator would pass the jump
# callbacks a parameter container instead of `prob.p`, and `FunctionWrapperSpecialize` can only
# be set by a problem constructor.
function extended_specialization(f)
    spec = SciMLBase.specialization(f)
    if spec === SciMLBase.FullSpecialize || spec === SciMLBase.NoSpecialize
        return spec
    end
    return SciMLBase.AutoSpecialize
end

function extend_problem(prob::SciMLBase.AbstractDiscreteProblem, jumps)
    error("General `VariableRateJump`s require a continuous problem, like an ODE/SDE/DDE/DAE problem. To use a `DiscreteProblem` bounded `VariableRateJump`s must be used. See the JumpProcesses docs.")
end

function extend_problem(prob::SciMLBase.AbstractODEProblem, jumps)
    _f = SciMLBase.unwrapped_f(prob.f)

    if isinplace(prob)
        jump_f = let _f = _f
            function (du::ExtendedJumpArray, u::ExtendedJumpArray, p, t)
                _f(du.u, u.u, p, t)
                update_jumps!(du, u, p, t, length(u.u), jumps)
            end
        end
    else
        jump_f = let _f = _f
            function (u::ExtendedJumpArray, p, t)
                du = ExtendedJumpArray(_f(u.u, p, t), u.jump_u)
                update_jumps!(du, u, p, t, length(u.u), jumps)
                return du
            end
        end
    end

    u0 = extend_u0(prob, length(jumps))
    spec = extended_specialization(prob.f)
    f = ODEFunction{isinplace(prob), spec}(jump_f; sys = prob.f.sys,
        observed = prob.f.observed)
    remake(prob; f, u0)
end

function extend_problem(prob::SciMLBase.AbstractSDEProblem, jumps)
    _f = SciMLBase.unwrapped_f(prob.f)

    if isinplace(prob)
        jump_f = let _f = _f
            function (du::ExtendedJumpArray, u::ExtendedJumpArray, p, t)
                _f(du.u, u.u, p, t)
                update_jumps!(du, u, p, t, length(u.u), jumps)
            end
        end
    else
        jump_f = let _f = _f
            function (u::ExtendedJumpArray, p, t)
                du = ExtendedJumpArray(_f(u.u, p, t), u.jump_u)
                update_jumps!(du, u, p, t, length(u.u), jumps)
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

    u0 = extend_u0(prob, length(jumps))
    spec = extended_specialization(prob.f)
    f = SDEFunction{isinplace(prob), spec}(jump_f, jump_g; sys = prob.f.sys,
        observed = prob.f.observed)
    remake(prob; f, g = jump_g, u0)
end

function extend_problem(prob::SciMLBase.AbstractDDEProblem, jumps)
    _f = SciMLBase.unwrapped_f(prob.f)

    if isinplace(prob)
        jump_f = let _f = _f
            function (du::ExtendedJumpArray, u::ExtendedJumpArray, h, p, t)
                _f(du.u, u.u, h, p, t)
                update_jumps!(du, u, p, t, length(u.u), jumps)
            end
        end
    else
        jump_f = let _f = _f
            function (u::ExtendedJumpArray, h, p, t)
                du = ExtendedJumpArray(_f(u.u, h, p, t), u.jump_u)
                update_jumps!(du, u, p, t, length(u.u), jumps)
                return du
            end
        end
    end

    u0 = extend_u0(prob, length(jumps))
    spec = extended_specialization(prob.f)
    f = DDEFunction{isinplace(prob), spec}(jump_f; sys = prob.f.sys,
        observed = prob.f.observed)
    remake(prob; f, u0)
end

# Not sure if the DAE one is correct: Should be a residual of sorts
function extend_problem(prob::SciMLBase.AbstractDAEProblem, jumps)
    _f = SciMLBase.unwrapped_f(prob.f)

    if isinplace(prob)
        jump_f = let _f = _f
            function (out, du::ExtendedJumpArray, u::ExtendedJumpArray, h, p, t)
                _f(out, du.u, u.u, h, p, t)
                update_jumps!(out, u, p, t, length(u.u), jumps)
            end
        end
    else
        jump_f = let _f = _f
            function (du, u::ExtendedJumpArray, h, p, t)
                out = ExtendedJumpArray(_f(du.u, u.u, h, p, t), u.jump_u)
                update_jumps!(du, u, p, t, length(u.u), jumps)
                return du
            end
        end
    end

    u0 = extend_u0(prob, length(jumps))
    spec = extended_specialization(prob.f)
    f = DAEFunction{isinplace(prob), spec}(jump_f, sys = prob.f.sys,
        observed = prob.f.observed)
    remake(prob; f, u0)
end

# The `VR_FRM` callbacks set `idxs` so that their conditions interpolate only their jumps'
# components of the extended state, `nu + i` for `jump_u[i]`, where `nu` is the length of
# the state. These indices are wrong if the state's length changes (for example through
# `remake`), so the callbacks check it when they are initialized.
function check_vr_frm_state_length(integrator, nu)
    if length(integrator.u.u) != nu
        throw(ArgumentError("The `VR_FRM` callbacks were built for a state of length $nu, but the state has length $(length(integrator.u.u)). Create a new `JumpProblem` to change the length of the state."))
    end
    nothing
end

############################ VR_FRM{false}: one callback per jump ############################

struct VR_FRMEventCallback{F}
    idx::Int
    affect!::F
end

# condition: (u, t, integrator); with the callback's `idxs`, `u` is the value of `jump_u[idx]`
@inline (c::VR_FRMEventCallback)(u, t, integrator) = u

# affect: (integrator)
function (c::VR_FRMEventCallback)(integrator)
    c.affect!(integrator)
    rng = get_rng(integrator)
    integrator.u.jump_u[c.idx] = -randexp(rng, typeof(integrator.t))
    nothing
end

# initialize: (cb, u, t, integrator)
function (c::VR_FRMEventCallback)(cb, u, t, integrator)
    check_vr_frm_state_length(integrator, cb.idxs - c.idx)
    rng = get_rng(integrator)
    integrator.u.jump_u[c.idx] = -randexp(rng, typeof(integrator.t))
    derivative_discontinuity!(integrator, true)
    nothing
end

function wrap_jump_in_callback(nu, idx, jump)
    cb_functor = VR_FRMEventCallback(idx, jump.affect!)
    ContinuousCallback(cb_functor, cb_functor;
        initialize = cb_functor,
        idxs = nu + idx,
        rootfind = jump.rootfind,
        interp_points = jump.interp_points,
        save_positions = jump.save_positions,
        abstol = jump.abstol,
        reltol = jump.reltol)
end

function build_variable_callback(cb, nu, idx, jump, jumps...)
    idx += 1
    new_cb = wrap_jump_in_callback(nu, idx, jump)
    build_variable_callback(CallbackSet(cb, new_cb), nu, idx, jumps...)
end

function build_variable_callback(cb, nu, idx, jump)
    idx += 1
    CallbackSet(cb, wrap_jump_in_callback(nu, idx, jump))
end

function vr_frm_callbacks(::VR_FRM{false}, nu, vrjs)
    build_variable_callback(CallbackSet(), nu, 0, vrjs...)
end

###################### VR_FRM{true}: one callback per set of jump options ######################

# the options a jump gives its callback; a `VectorContinuousCallback` has one set of them
function vr_frm_callback_options(jump)
    (; jump.rootfind, jump.interp_points, jump.save_positions, jump.abstol, jump.reltol)
end

# condition: (out, u, t, integrator); with the callback's `idxs`, `u` holds the values of its
# jumps' components of `jump_u`
struct VR_FRMVectorCondition end
@inline (::VR_FRMVectorCondition)(out, u, t, integrator) = (out .= u; nothing)

struct VR_FRMVectorCallback{A}
    jump_idxs::Vector{Int}  # indices in `jump_u` of the callback's jumps
    affects::A              # Tuple of their `affect!` functions
    init_all::Bool          # whether `initialize` draws the thresholds of all the jumps
end

# calls `affects[k](integrator)` through one branch per affect, so each call is static; `k` is
# always in `1:length(affects)`, so the last affect needs no test
@generated function vr_frm_affect!(affects::Tuple, k, integrator)
    quote
        Base.Cartesian.@nif $(fieldcount(affects)) i -> (i == k) i -> (@inbounds affects[i](integrator)) i -> (@inbounds affects[$(fieldcount(affects))](integrator))
        return nothing
    end
end

# affect: (integrator, events); `events[k]` is nonzero if the `k`th jump of the callback
# fired. DiffEqBase may pass a longer buffer, so only the callback's entries are read.
function (c::VR_FRMVectorCallback)(integrator, events)
    rng = get_rng(integrator)
    for (k, i) in enumerate(c.jump_idxs)
        iszero(events[k]) && continue
        vr_frm_affect!(c.affects, k, integrator)
        integrator.u.jump_u[i] = -randexp(rng, typeof(integrator.t))
    end
    nothing
end

# initialize: (cb, u, t, integrator); the first callback draws the thresholds of all the
# jumps, in order, so that they are drawn as by `VR_FRM{false}`
function (c::VR_FRMVectorCallback)(cb, u, t, integrator)
    check_vr_frm_state_length(integrator, first(cb.idxs) - first(c.jump_idxs))
    if c.init_all
        rng = get_rng(integrator)
        jump_u = integrator.u.jump_u
        for i in eachindex(jump_u)
            jump_u[i] = -randexp(rng, typeof(integrator.t))
        end
        derivative_discontinuity!(integrator, true)
    end
    nothing
end

function vr_frm_callbacks(::VR_FRM{true}, nu, vrjs)
    opts = map(vr_frm_callback_options, vrjs)
    cbs = map(enumerate(unique(opts))) do (n, jump_opts)
        jump_idxs = findall(isequal(jump_opts), opts)
        # a range when the jumps are contiguous, as when they all share their options
        idxs = if jump_idxs == first(jump_idxs):last(jump_idxs)
            (nu + first(jump_idxs)):(nu + last(jump_idxs))
        else
            nu .+ jump_idxs
        end
        affects = Tuple(vrjs[i].affect! for i in jump_idxs)
        cb_functor = VR_FRMVectorCallback(jump_idxs, affects, n == 1)
        VectorContinuousCallback(VR_FRMVectorCondition(), cb_functor, length(jump_idxs);
            initialize = cb_functor, idxs, jump_opts...)
    end
    CallbackSet(cbs...)
end

# Unrolled at compile time: recursive splatting over `jumps...` is only inferred for
# tuples of up to 32 elements, beyond which every rate evaluation is dynamically dispatched.
@generated function update_jumps!(du, u, p, t, idx, jumps::Tuple)
    body = Expr(:block)
    for i in 1:fieldcount(jumps)
        push!(body.args, :(du[idx + $i] = jumps[$i].rate(u.u, p, t)))
    end
    push!(body.args, :(return nothing))
    return body
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
"""
$(TYPEDEF)

Function-wrapper variant of [`VR_Direct`](@ref) for simulations with many
[`VariableRateJump`](@ref)s.

`VR_DirectFW` uses the same direct-method callback strategy as `VR_Direct`, but stores the
rate and affect functions in `FunctionWrappers`-based containers.

## Returns

  - A stateless [`VariableRateAggregator`](@ref) value for the `vr_aggregator` keyword of
    [`JumpProblem`](@ref).

## Examples

```julia
using JumpProcesses, OrdinaryDiffEq

u0 = [1.0]
p = [10.0, 0.5]
tspan = (0.0, 10.0)

birth_rate(u, p, t) = p[1]
birth_affect!(integrator) = (integrator.u[1] += 1; nothing)
birth_jump = VariableRateJump(birth_rate, birth_affect!)

death_rate(u, p, t) = p[2] * u[1]
death_affect!(integrator) = (integrator.u[1] -= 1; nothing)
death_jump = VariableRateJump(death_rate, death_affect!)

oprob = ODEProblem((du, u, p, t) -> du .= 0, u0, tspan, p)
jprob = JumpProblem(oprob, birth_jump, death_jump; vr_aggregator = VR_DirectFW())
sol = solve(jprob, Tsit5())
```
"""
struct VR_DirectFW <: VariableRateAggregator end

mutable struct VR_DirectEventCache{T, F1, F2}
    prev_time::T
    prev_threshold::T
    current_time::T
    current_threshold::T
    total_rate::T
    rate_funcs::F1
    affect_funcs::F2
    cum_rate_sum::Vector{T}
end

function VR_DirectEventCache(
        jumps::JumpSet, ::VR_Direct, prob, ::Type{T}) where {T}
    vjumps = jumps.variable_jumps

    # handle vjumps using tuples
    rate_funcs, affect_funcs = get_jump_info_tuples(vjumps)

    cum_rate_sum = Vector{T}(undef, length(vjumps))

    VR_DirectEventCache{T, typeof(rate_funcs), typeof(affect_funcs)}(zero(T),
        zero(T), zero(T), zero(T), zero(T), rate_funcs,
        affect_funcs, cum_rate_sum)
end

function VR_DirectEventCache(
        jumps::JumpSet, ::VR_DirectFW, prob, ::Type{T}) where {T}
    vjumps = jumps.variable_jumps

    t, u = prob.tspan[1], prob.u0

    # handle vjumps using tuples
    rate_funcs, affect_funcs = get_jump_info_fwrappers(u, prob.p, t, vjumps)

    cum_rate_sum = Vector{T}(undef, length(vjumps))

    VR_DirectEventCache{T, typeof(rate_funcs), Any}(zero(T),
        zero(T), zero(T), zero(T), zero(T), rate_funcs,
        affect_funcs, cum_rate_sum)
end

# Initialization function for VR_DirectEventCache
function initialize_vr_direct_cache!(cache::VR_DirectEventCache, u, t, integrator)
    rng = get_rng(integrator)
    cache.prev_time = zero(integrator.t)
    cache.current_time = zero(integrator.t)
    cache.prev_threshold = randexp(rng, eltype(integrator.t))
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

@inline function concretize_vr_direct_affects!(cache::VR_DirectEventCache{T, F1, F2},
        ::I) where {T, F1, F2 <: Tuple, I <: SciMLBase.DEIntegrator}
    nothing
end

uses_rate_wrappers(cache::VR_DirectEventCache) = holds_rate_wrappers(cache.rate_funcs)

# Wrapper for initialize to match ContinuousCallback signature
function initialize_vr_direct_wrapper(cb::ContinuousCallback, u, t, integrator)
    check_parameters(cb.condition, integrator.p)
    concretize_vr_direct_affects!(cb.condition, integrator)
    initialize_vr_direct_cache!(cb.condition, u, t, integrator)
    derivative_discontinuity!(integrator, false)
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

function configure_jump_problem(prob, ::VR_Direct, jumps, cvrjs)
    new_prob = prob
    cache = VR_DirectEventCache(jumps, VR_Direct(), prob, eltype(prob.tspan))
    variable_jump_callback = build_variable_integcallback(cache, cvrjs)
    return new_prob, variable_jump_callback
end

function configure_jump_problem(prob, ::VR_DirectFW, jumps, cvrjs)
    new_prob = prob
    cache = VR_DirectEventCache(jumps, VR_DirectFW(), prob, eltype(prob.tspan))
    variable_jump_callback = build_variable_integcallback(cache, cvrjs)
    return new_prob, variable_jump_callback
end

# Unrolled at compile time: recursive splatting over `rates...` is only inferred for
# tuples of up to 32 elements, beyond which every rate evaluation is dynamically dispatched.
@generated function cumsum_rates!(cum_rate_sum, u, p, t, rates::Tuple)
    body = Expr(:block, :(cur_sum = zero(eltype(cum_rate_sum))))
    for i in 1:fieldcount(rates)
        push!(body.args, :(cur_sum += rates[$i](u, p, t)))
        push!(body.args, :(@inbounds cum_rate_sum[$i] = cur_sum))
    end
    push!(body.args, :(return cur_sum))
    return body
end

# loop-based version for Vector rate_funcs (avoids dynamic splatting)
@inline function cumsum_rates!(cum_rate_sum, u, p, t, rates::AbstractVector)
    cur_sum = zero(eltype(cum_rate_sum))
    @inbounds for (idx, rate) in enumerate(rates)
        cur_sum += rate(u, p, t)
        cum_rate_sum[idx] = cur_sum
    end
    return cur_sum
end

function total_variable_rate(
        cache::VR_DirectEventCache{T, F1, F2}, u, p, t) where {T, F1, F2}
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
    gps = _GAUSS_POINTS
    weights = _GAUSS_WEIGHTS
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

@generated function execute_affect!(cache::VR_DirectEventCache{T, F1, F2},
        integrator::I, idx) where {T, F1, F2 <: Tuple, I <: SciMLBase.DEIntegrator}
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

    rng = get_rng(integrator)
    r = rand(rng) * total_variable_rate_sum
    @inbounds jump_idx = searchsortedfirst(cache.cum_rate_sum, r)
    execute_affect!(cache, integrator, jump_idx)

    cache.prev_time = t
    cache.current_threshold = randexp(rng)
    cache.prev_threshold = cache.current_threshold
    cache.current_time = t
    return nothing
end
