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
  - The total rate is integrated over each solver step with a piecewise polynomial
    (Chebyshev) interpolant of it, refined until the integral's estimated error is within
    the solver's `abstol` and `reltol`. Rates that vary within long solver steps, for
    example when the ODE part is constant, are resolved without limiting the step size.
  - The rates are evaluated along the solver's interpolant between steps. In SDE problems
    that doesn't resolve the noise within a step, so a rate that depends on a component
    driven by noise needs steps that are short compared with its variation, for example
    by setting `dtmax`. `VR_FRM` integrates each rate as a component of the state instead,
    so the solver's error control covers it.
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

# The condition integrates the total rate over each solver step with a piecewise Chebyshev
# interpolant of it, fit the first time the condition sees the step. Each panel's interpolant
# uses the rates at NUM_CHEB_NODES Chebyshev points, and panels are halved, at most
# MAX_CHEB_DEPTH times, until the estimated error of their integral is within the solver's
# tolerances. Fitting stops at the panel where the threshold is reached, since the step then
# ends at the jump; that panel may be halved up to MAX_CHEB_CROSSING_DEPTH times. Root
# finding then evaluates the interpolant's antiderivative, with no further rate evaluations.
const NUM_CHEB_NODES = 8
const MAX_CHEB_DEPTH = 12
const MAX_CHEB_CROSSING_DEPTH = 40
# Chebyshev points of the first kind on [-1, 1]
const CHEB_NODES = [cos(π * (j - 1 / 2) / NUM_CHEB_NODES) for j in 1:NUM_CHEB_NODES]
# maps the rates at CHEB_NODES to the coefficients cₖ of their interpolant
# c₀ / 2 + ∑ₖ cₖ Tₖ(x), k = 1:(NUM_CHEB_NODES - 1)
const CHEB_TRANSFORM = [2 * cos(π * k * (j - 1 / 2) / NUM_CHEB_NODES) / NUM_CHEB_NODES
                        for k in 0:(NUM_CHEB_NODES - 1), j in 1:NUM_CHEB_NODES]
# the interpolant's integral over [-1, 1] is ∑ₖ CHEB_INTEGRAL_WEIGHTS[k + 1] cₖ
const CHEB_INTEGRAL_WEIGHTS = [k == 0 ? 1.0 : isodd(k) ? 0.0 : 2 / (1 - k^2)
                               for k in 0:(NUM_CHEB_NODES - 1)]

# The total rate's interpolant over the current step, and buffers for fitting it
struct TotalRateFit{T}
    panel_starts::Vector{T}
    panel_ends::Vector{T}
    # integral of the interpolant from the start of the step to the start of each panel
    panel_integrals::Vector{T}
    # coefficients of the interpolant's antiderivative on each panel, NUM_CHEB_NODES + 1
    # per panel
    panel_coefs::Vector{T}
    rate_samples::Vector{T}
    rate_coefs::Vector{T}
end

function TotalRateFit{T}() where {T}
    TotalRateFit{T}(T[], T[], T[], T[], Vector{T}(undef, NUM_CHEB_NODES),
        Vector{T}(undef, NUM_CHEB_NODES))
end

function empty_panels!(fit::TotalRateFit)
    empty!(fit.panel_starts)
    empty!(fit.panel_ends)
    empty!(fit.panel_integrals)
    empty!(fit.panel_coefs)
    nothing
end

mutable struct VR_DirectEventCache{T, F1, F2}
    prev_time::T
    prev_threshold::T
    current_time::T
    current_threshold::T
    total_rate::T
    rate_funcs::F1
    affect_funcs::F2
    cum_rate_sum::Vector{T}
    rate_fit::TotalRateFit{T}
end

function VR_DirectEventCache(
        jumps::JumpSet, ::VR_Direct, prob, ::Type{T}) where {T}
    vjumps = jumps.variable_jumps

    # handle vjumps using tuples
    rate_funcs, affect_funcs = get_jump_info_tuples(vjumps)

    cum_rate_sum = Vector{T}(undef, length(vjumps))

    VR_DirectEventCache{T, typeof(rate_funcs), typeof(affect_funcs)}(zero(T),
        zero(T), zero(T), zero(T), zero(T), rate_funcs,
        affect_funcs, cum_rate_sum, TotalRateFit{T}())
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
        affect_funcs, cum_rate_sum, TotalRateFit{T}())
end

# Initialization function for VR_DirectEventCache
function initialize_vr_direct_cache!(cache::VR_DirectEventCache, u, t, integrator)
    rng = get_rng(integrator)
    cache.prev_time = t
    cache.current_time = t
    cache.prev_threshold = randexp(rng, eltype(integrator.t))
    cache.current_threshold = cache.prev_threshold
    cache.total_rate = zero(integrator.t)
    cache.cum_rate_sum .= 0
    empty_panels!(cache.rate_fit)
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

# Interpolates the total rate on [a, b], leaving the coefficients in `rate_fit.rate_coefs`.
# Returns the interpolant's integral over [a, b], and an estimate of its error from the last
# two coefficients.
function fit_total_rate!(cache::VR_DirectEventCache, integrator, a, b)
    (; rate_samples, rate_coefs) = cache.rate_fit
    p = integrator.p
    u_τ = first(SciMLBase.get_tmp_cache(integrator))
    tmid = (a + b) / 2
    halfdt = (b - a) / 2
    for j in 1:NUM_CHEB_NODES
        τ = halfdt * CHEB_NODES[j] + tmid
        integrator(u_τ, τ)
        rate_samples[j] = total_variable_rate(cache, u_τ, p, τ)
    end
    integral = zero(eltype(rate_coefs))
    for k in 1:NUM_CHEB_NODES
        c = zero(eltype(rate_coefs))
        for j in 1:NUM_CHEB_NODES
            c += CHEB_TRANSFORM[k, j] * rate_samples[j]
        end
        rate_coefs[k] = c
        integral += CHEB_INTEGRAL_WEIGHTS[k] * c
    end
    err = (b - a) * (abs(rate_coefs[end]) + abs(rate_coefs[end - 1]))
    return halfdt * integral, err
end

# Adds the panel [a, b], whose interpolant is in `fit.rate_coefs`, with the coefficients of
# the interpolant's antiderivative ∑ⱼ Cⱼ Tⱼ(x), Cⱼ = (cⱼ₋₁ - cⱼ₊₁) / 2j, which is zero at
# x = -1. `integral` is the integral from the start of the step to `a`.
function push_panel!(fit::TotalRateFit, a, b, integral)
    (; rate_coefs, panel_coefs) = fit
    push!(fit.panel_starts, a)
    push!(fit.panel_ends, b)
    push!(fit.panel_integrals, integral)
    i0 = length(panel_coefs) + 1
    resize!(panel_coefs, length(panel_coefs) + NUM_CHEB_NODES + 1)
    C0 = zero(eltype(panel_coefs))
    for j in 1:NUM_CHEB_NODES
        cnext = j + 2 <= NUM_CHEB_NODES ? rate_coefs[j + 2] : zero(C0)
        C = (rate_coefs[j] - cnext) / (2j)
        panel_coefs[i0 + j] = C
        C0 += isodd(j) ? C : -C
    end
    panel_coefs[i0] = C0
    nothing
end

# Bound on the rounding error of evaluating the antiderivative of the interpolant in
# `fit.rate_coefs` on [a, b]: its coefficients scale with the rates on the panel, so a small
# integral near the panel's start is a difference of much larger terms
function antiderivative_rounding(fit::TotalRateFit, a, b)
    (b - a) * (NUM_CHEB_NODES + 1) * eps(eltype(fit.rate_coefs)) * sum(abs, fit.rate_coefs)
end

# Fits the total rate on [a, b], halving the interval until the estimated error of each
# panel's integral is within the tolerance the solver would use if the remaining threshold
# were a state component. `integral` is the integral from the start of the step to `a`.
# Returns the integral to the end of the last panel fitted, and whether the threshold was
# reached there. Fitting stops at that panel, since the step ends at the jump. The jump time
# is where the remaining threshold minus the integral crosses zero, so that panel must also
# be accurate relative to the remaining threshold, including the antiderivative's rounding.
function fit_panels!(cache::VR_DirectEventCache, integrator, a, b, integral, depth,
        abstol, reltol)
    I, err = fit_total_rate!(cache, integrator, a, b)
    remaining = cache.prev_threshold - integral
    tol = abstol + reltol * abs(remaining)
    crossed = I >= remaining
    if crossed
        err = max(err, antiderivative_rounding(cache.rate_fit, a, b))
    end
    tmid = (a + b) / 2
    maxdepth = crossed ? MAX_CHEB_CROSSING_DEPTH : MAX_CHEB_DEPTH
    if err > tol && depth < maxdepth && a < tmid < b
        integral, crossed = fit_panels!(
            cache, integrator, a, tmid, integral, depth + 1, abstol, reltol)
        crossed && return integral, crossed
        return fit_panels!(cache, integrator, tmid, b, integral, depth + 1, abstol, reltol)
    end
    push_panel!(cache.rate_fit, a, b, integral)
    return integral + I, crossed
end

# the solver's tolerances can be given per state component
scalar_tolerance(tol::Number) = tol
scalar_tolerance(tol) = minimum(tol)

# Fits the total rate over the step just taken, from `cache.prev_time` to `integrator.t`
function fit_step!(cache::VR_DirectEventCache, integrator)
    empty_panels!(cache.rate_fit)
    a, b = cache.prev_time, integrator.t
    if b > a
        abstol = scalar_tolerance(integrator.opts.abstol)
        reltol = scalar_tolerance(integrator.opts.reltol)
        fit_panels!(cache, integrator, a, b, zero(cache.prev_threshold), 0, abstol, reltol)
    end
    nothing
end

# ∑ⱼ C[offset + j + 1] Tⱼ(x) for j = 0:NUM_CHEB_NODES, by Clenshaw's recurrence
function chebyshev_sum(C, offset, x)
    b1 = b2 = zero(x)
    for j in (NUM_CHEB_NODES + 1):-1:2
        b1, b2 = C[offset + j] + 2x * b1 - b2, b1
    end
    return C[offset + 1] + x * b1 - b2
end

# the interpolant's integral from the start of the step to `t`; past the last panel, which
# ends before the step does if the threshold was reached, it is the integral to that panel's
# end, so the condition keeps its sign there without extrapolating the interpolant
function fitted_rate_integral(fit::TotalRateFit, t)
    (; panel_starts, panel_ends, panel_integrals, panel_coefs) = fit
    isempty(panel_ends) && return zero(t)
    i = min(searchsortedfirst(panel_ends, t), length(panel_ends))
    a, b = panel_starts[i], panel_ends[i]
    x = min((2t - a - b) / (b - a), one(t))
    return panel_integrals[i] +
           (b - a) / 2 * chebyshev_sum(panel_coefs, (i - 1) * (NUM_CHEB_NODES + 1), x)
end

# Condition functor defined directly on the cache
function (cache::VR_DirectEventCache)(u, t, integrator)
    if integrator.t != cache.current_time
        # The step being checked starts at `integrator.tprev`. That is before the end of the
        # previous fit if another callback's event cut the previous step short, so the
        # threshold there comes from that fit, and the rest of it is discarded.
        tprev = integrator.tprev
        if tprev < cache.prev_time
            error("The solver step starts at $tprev, before $(cache.prev_time), where `VR_Direct` last integrated the total rate from. This is not supported.")
        end
        cache.prev_threshold -= fitted_rate_integral(cache.rate_fit, tprev)
        cache.prev_time = tprev
        cache.current_time = integrator.t
        fit_step!(cache, integrator)
    end

    dt = t - cache.prev_time
    if dt == 0
        return cache.prev_threshold
    end

    cache.current_threshold = cache.prev_threshold - fitted_rate_integral(cache.rate_fit, t)
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
    # the integral up to the jump is used up, and the next step starts at the jump
    empty_panels!(cache.rate_fit)
    return nothing
end
