"""
$(TYPEDEF)

Highly efficient integrator for pure jump problems that involve only `ConstantRateJump`s,
`MassActionJump`s, and/or `VariableRateJump`s *with rate bounds*.

## Constructor

```julia
SSAStepper(; save_uprev = false)
```

  - `save_uprev`: whether the integrator keeps the state at the start of each step in
    `integrator.uprev`, so that it can be evaluated anywhere in the current step (see
    "Evaluating the integrator" below). This copies the state on every step, at a cost
    proportional to the size of the state, which can dominate the per-step cost of
    efficient aggregators on large or spatial systems. `SSAStepper()` keeps no copy.

## Supported state types

The state `u` can be a `Vector` or an `SVector` of integers or floats; a species × sites
`Matrix{Int}` for spatial problems, which `NSM` and `DirectCRDirect` keep and other
aggregators flatten into a `Vector`; or, for models without `MassActionJump`s, an integer
or float scalar (not with `RSSA` or `RSSACR`). Bounded
`VariableRateJump`s require `Coevolve`, and `affect!` functions for `SVector` and scalar
states must assign a new value to `integrator.u`. Other state types are not supported. See
the [supported state
types](https://docs.sciml.ai/JumpProcesses/stable/jump_solve/#ssa_state_types) for
details.

## Evaluating the integrator

Callbacks, such as DiffEqCallbacks' `SavingCallback` and `FunctionCallingCallback` with
save times, may evaluate the integrator at times before the current one with
`integrator(t)` or `integrator(out, t)`. The sampled path is piecewise constant, and
jumps and callbacks only change the state at the end of a step, so:

  - Evaluating at the current time, `integrator.t`, returns the current state.
  - With `SSAStepper(; save_uprev = true)`, evaluating at any `t` in
    `[integrator.tprev, integrator.t)` returns the state at the start of the step,
    `integrator.uprev`. `get_tmp_cache(integrator)` then also provides scratch space for
    callbacks that evaluate the integrator in place.
  - Any other time raises an `ArgumentError`: a path cannot be extrapolated, and earlier
    states are only available from the saved solution. With the default,
    `SSAStepper()`, every time other than `integrator.t` raises this error, as does
    `get_tmp_cache`.

`integrator(t)` returns a copy for array states, and `integrator(out, t)` writes into
`out` and returns it. Instead of `save_uprev = true`, the times at which callbacks
evaluate the integrator can be passed as `tstops`, so that it stops exactly at them. This
avoids the per-step copy, but adds a stop and a pass through the callbacks at each time,
and, when every jump is saved, a saved point at each stop, so pair it with
`save_positions = (false, false)`. To save the full state, `saveat` is exact and needs
neither. DiffEqCallbacks' integrating callbacks (`IntegratingCallback`,
`IntegratingSumCallback`) are not yet supported with `SSAStepper`.

## Notes

  - Only works with `JumpProblem`s defined from `DiscreteProblem`s.
  - Only works with collections of `ConstantRateJump`s, `MassActionJump`s, and
    `VariableRateJump`s with rate bounds.
  - Only supports `DiscreteCallback`s for events, which are checked after every step taken by
    `SSAStepper`.
  - Only supports a limited subset of the output controls from the common solver interface,
    specifically `save_start`, `save_end`, and `saveat`.
  - Supports `rng` and `seed` keyword arguments in `solve`/`init` to control the random
    number generator used for jump sampling. `rng` accepts any `AbstractRNG`, while `seed`
    creates a `Xoshiro` generator. `rng` takes priority over `seed`.
  - Reuses the `JumpProblem`'s jump state (aggregator and jump callback), re-initializing
    it at every `init`; it is never copied. One `JumpProblem` therefore supports one active
    solve or integrator at a time. For concurrent solves use independent copies, for
    example `deepcopy(jprob)`, or an `EnsembleProblem`. See the `JumpProblem` docstring.
  - Supports the common `alias` keyword argument for its `u0`, `p`, and `tstops` inputs,
    following SciMLBase's alias-specifier convention: pass `nothing` (the default), a
    `Bool` that applies to all three, a `SciMLBase.DiscreteAliasSpecifier` (`alias_u0`,
    `alias_p`), or an `ODEAliasSpecifier` (which also has `alias_tstops`). `true` permits
    aliasing and `false` requests a copy. The defaults copy `u0` (with `recursivecopy`),
    reuse `p`, and never modify a caller's `tstops` array. With `alias_u0 = true` the
    integrator mutates the problem's `u0`, so a later solve starts from its final state.
    `alias_f` and `alias_du0` have no effect, and `alias` never controls jump state.
  - `tstops` may be a number, a tuple, or an `AbstractVector` of times (such as a range or
    view), or a callable `(p, tspan) -> times`. Only a `Vector` of the time type can be
    aliased; other supported inputs are copied into a new `Vector` at `init`.
  - Saved states are independent snapshots of the integrator state made with
    `recursivecopy`, as in OrdinaryDiffEq.
  - As when using jumps with ODEs and SDEs, saving controls for whether to save each time a
    jump occurs are via the `save_positions` keyword argument to `JumpProblem`. Note that when
    choosing `SSAStepper` as the timestepper, `save_positions = (true,true)`, `(true,false)`,
    or `(false,true)` are all equivalent. `SSAStepper` will save only the post-jump state in
    the solution object in each of these cases. This is because solution objects generated via
    `SSAStepper` use piecewise-constant interpolation, and can therefore exactly reconstruct
    the sampled jump process path with knowing just the post-jump state. That is, `sol(t)`
    for any `0 <= t <= tstop` will give the exact value of the sampled solution path at `t`
    provided at least one component of `save_positions` is `true`.

## Examples

SIR model:

```julia
using JumpProcesses
β = 0.1 / 1000.0;
ν = 0.01;
p = (β, ν)
rate1(u, p, t) = p[1]*u[1]*u[2]  # β*S*I
function affect1!(integrator)
    integrator.u[1] -= 1         # S -> S - 1
    integrator.u[2] += 1         # I -> I + 1
end
jump = ConstantRateJump(rate1, affect1!)

rate2(u, p, t) = p[2]*u[2]      # ν*I
function affect2!(integrator)
    integrator.u[2] -= 1        # I -> I - 1
    integrator.u[3] += 1        # R -> R + 1
end
jump2 = ConstantRateJump(rate2, affect2!)
u₀ = [999, 1, 0]
tspan = (0.0, 250.0)
prob = DiscreteProblem(u₀, tspan, p)
jump_prob = JumpProblem(prob, Direct(), jump, jump2)
sol = solve(jump_prob, SSAStepper())
```

see the
[tutorial](https://docs.sciml.ai/JumpProcesses/stable/tutorials/discrete_stochastic_example/)
for details.
"""
struct SSAStepper{SaveUprev} <: SciMLBase.AbstractDEAlgorithm
    function SSAStepper{S}() where {S}
        S isa Bool ||
            throw(ArgumentError("The `SSAStepper` type parameter must be a `Bool`, got $S."))
        new{S}()
    end
end
SSAStepper(; save_uprev::Bool = false) = SSAStepper{save_uprev}()

# Whether the integrator keeps the state at the start of each step (`save_uprev`).
ssa_save_uprev(::SSAStepper{S}) where {S} = S

SciMLBase.allows_late_binding_tstops(::SSAStepper) = true
SciMLBase.supports_solve_rng(::JumpProblem, ::SSAStepper) = true

"""
$(TYPEDEF)

Integrator for pure jump problems solved via `SSAStepper`.

## Fields

$(FIELDS)
"""
mutable struct SSAIntegrator{F, uType, tType, tdirType, P, S, CB, SA, OPT, TS, R, UP, TC} <:
               AbstractSSAIntegrator{SSAStepper, Nothing, uType, tType}
    """
    The underlying `prob.f` function. Not currently used.
    """
    f::F
    """
    The current solution values.
    """
    u::uType
    """
    The current solution time.
    """
    t::tType
    """
    The time at the start of the current step. A step ends at the next jump or `tstop`,
    and `solve!`'s final advance to the end time also counts as a step.
    """
    tprev::tType
    """
    The state at the start of the current step, at time `tprev`, when solving with
    `SSAStepper(; save_uprev = true)`; otherwise `nothing`.
    """
    uprev::UP
    """
    The direction time is changing in (must be positive, indicating time is increasing)
    """
    tdir::tdirType
    """
    The current parameters.
    """
    p::P
    """
    The current solution object.
    """
    sol::S
    i::Int
    """
    The next jump time.
    """
    tstop::tType
    """
    The jump aggregator callback.
    """
    cb::CB
    """
    Times to save the solution at.
    """
    saveat::SA
    """
    Whether to save every time a jump occurs.
    """
    save_everystep::Bool
    """
    Whether to save at the final step.
    """
    save_end::Bool
    """
    Index of the next `saveat` time.
    """
    cur_saveat::Int
    """
    Tuple storing callbacks.
    """
    opts::OPT
    """
    User supplied times to step to, useful with callbacks.
    """
    tstops::TS
    tstops_idx::Int
    u_modified::Bool
    keep_stepping::Bool          # false if should terminate a simulation
    """
    If true, will write tstops into the user-passed array
    """
    alias_tstops::Bool
    """
    If true indicates we have already allocated the tstops array
    """
    copied_tstops::Bool
    """
    The random number generator.
    """
    rng::R
    """
    Scratch space returned by `get_tmp_cache` for `save_uprev = true` and array states
    other than `SVector`s; otherwise `nothing`.
    """
    tmp_cache::TC
end

SciMLBase.has_rng(::SSAIntegrator) = true
SciMLBase.get_rng(integrator::SSAIntegrator) = integrator.rng
function SciMLBase.set_rng!(integrator::SSAIntegrator, rng)
    R = typeof(integrator.rng)
    if !isa(rng, R)
        throw(ArgumentError(
            "Cannot set RNG of type $(typeof(rng)) on an integrator " *
            "whose RNG type parameter is $R. " *
            "Construct a new integrator via `init(prob, alg; rng = your_rng)` instead."
        ))
    end
    integrator.rng = rng
    nothing
end

# Copy the state `src` into the existing array `dest`. Nested arrays are copied
# recursively, like saved states; other states, including scalars and `SVector`s, are
# broadcast into `dest`.
@inline function copy_state!(dest, src)
    if src isa AbstractArray && eltype(src) <: AbstractArray
        recursivecopy!(dest, src)
    else
        dest .= src
    end
    return dest
end

# Store the state at the start of a step in `uprev` (only with `save_uprev = true`).
# Scalar and `SVector` states are replaced; other states are copied into the existing
# buffer, so no allocation occurs.
@inline store_uprev!(integrator::SSAIntegrator) = store_uprev!(integrator, integrator.uprev)
@inline store_uprev!(::SSAIntegrator, ::Nothing) = nothing
@inline function store_uprev!(integrator::SSAIntegrator, uprev)
    u = integrator.u
    if u isa Union{Number, SVector}
        integrator.uprev = u
    else
        copy_state!(uprev, u)
    end
    return nothing
end

# The `uprev` and `get_tmp_cache` storage for a new integrator: none unless
# `save_uprev = true`, and no scratch array for scalar and `SVector` states.
ssa_uprev_storage(::SSAStepper{false}, _) = (nothing, nothing)
function ssa_uprev_storage(::SSAStepper{true}, u0)
    if u0 isa Union{Number, SVector}
        return u0, nothing
    else
        return recursivecopy(u0), recursivecopy(u0)
    end
end

# The state at time `t`, which must be `integrator.t`, or, with `save_uprev = true`, lie
# in `[integrator.tprev, integrator.t)`, where the state equals `uprev`, since jumps and
# callbacks only change the state at the end of a step.
@inline function state_at(integrator::SSAIntegrator, t)
    t == integrator.t && return integrator.u
    uprev = integrator.uprev
    uprev === nothing && throw_ssa_evaluation_needs_uprev(integrator, t)
    (integrator.tprev <= t < integrator.t) ||
        throw_ssa_evaluation_out_of_step(integrator, t)
    return uprev
end

@noinline function throw_ssa_evaluation_needs_uprev(integrator, t)
    throw(ArgumentError("An `SSAStepper` integrator can only be evaluated at its current \
        time `integrator.t = $(integrator.t)`, but was evaluated at `t = $t`. To evaluate \
        it anywhere in the current step, `[integrator.tprev, integrator.t]`, as \
        `SavingCallback`s and `FunctionCallingCallback`s with save times require, solve \
        with `SSAStepper(; save_uprev = true)`. Alternatively, pass those times as \
        `tstops`, so that the integrator stops exactly at them."))
end

@noinline function throw_ssa_evaluation_out_of_step(integrator, t)
    throw(ArgumentError("An `SSAStepper` integrator can only be evaluated in the current \
        step, `[integrator.tprev, integrator.t] = [$(integrator.tprev), \
        $(integrator.t)]`, but was evaluated at `t = $t`. A jump process path cannot be \
        extrapolated, and states before the current step are only available from the \
        saved solution."))
end

@noinline function throw_ssa_tmp_cache_needs_uprev()
    throw(ArgumentError("`get_tmp_cache` is only available for `SSAStepper` integrators \
        solved with `SSAStepper(; save_uprev = true)`. It provides scratch space for \
        callbacks, such as an in-place `SavingCallback`, that evaluate the integrator \
        at times before `integrator.t`, which requires `save_uprev = true`. \
        Alternatively, pass the callback's times as `tstops`, so that the integrator \
        stops exactly at them."))
end

# Evaluate the piecewise-constant path at time `t`; see `state_at` for the valid times.
# Returns a copy for array states, and the value itself for scalar and `SVector` states.
(integrator::SSAIntegrator)(t) = recursivecopy(state_at(integrator, t))

# Copy the state at time `t` into the caller's `out` and return `out`.
(integrator::SSAIntegrator)(out, t) = copy_state!(out, state_at(integrator, t))

function SciMLBase.get_tmp_cache(integrator::SSAIntegrator)
    integrator.uprev === nothing && throw_ssa_tmp_cache_needs_uprev()
    cache = integrator.tmp_cache
    return cache === nothing ? nothing : (cache,)
end

# Save a snapshot of the current state, matching OrdinaryDiffEq's save semantics
# (`copyat_or_push!` stores a `recursivecopy` of the live state).
@inline function save_current_state!(integrator::SSAIntegrator, t)
    push!(integrator.sol.t, t)
    copyat_or_push!(integrator.sol.u, length(integrator.sol.u) + 1, integrator.u)
    nothing
end

# SciMLBase v3 / DiffEqBase v7 renamed the integrator's `u_modified` field to
# `derivative_discontinuity` and internal callback code now reads/writes the
# field directly (e.g. `integrator.derivative_discontinuity`, not via a
# method). Aliasing here so both names access the same underlying storage —
# v6-era callbacks that look for `:u_modified` and v7-era callbacks that look
# for `:derivative_discontinuity` both keep working without renaming the
# struct field (which would be a breaking ABI change).
@inline function Base.getproperty(integrator::SSAIntegrator, sym::Symbol)
    sym === :derivative_discontinuity && return getfield(integrator, :u_modified)
    sym === :ps && return SII.ParameterIndexingProxy(integrator)
    return getfield(integrator, sym)
end

@inline function Base.setproperty!(integrator::SSAIntegrator, sym::Symbol, val)
    sym === :derivative_discontinuity &&
        return setfield!(integrator, :u_modified, convert(Bool, val))
    return setfield!(integrator, sym, convert(fieldtype(typeof(integrator), sym), val))
end

function Base.propertynames(integrator::SSAIntegrator, private::Bool = false)
    return (fieldnames(SSAIntegrator)..., :derivative_discontinuity)
end

function SciMLBase.derivative_discontinuity!(integrator::SSAIntegrator, bool::Bool)
    integrator.u_modified = bool
end

function SciMLBase.__solve(jump_prob::JumpProblem, alg::SSAStepper; kwargs...)
    # init will handle kwargs merging via init_call
    integrator = init(jump_prob, alg; kwargs...)
    solve!(integrator)
    integrator.sol
end

function DiffEqBase.solve!(integrator::SSAIntegrator)
    end_time = integrator.sol.prob.tspan[2]
    while should_continue_solve(integrator) # It stops before adding a tstop over
        step!(integrator)
    end

    # if the user terminated the solve we shouldn't advance in time any more
    if integrator.sol.retcode !== ReturnCode.Terminated
        # The advance to the end time is a final step without a jump, so it starts a new
        # step for evaluating the integrator (and `tprev` always marks a step's start).
        if integrator.t < end_time
            store_uprev!(integrator)
            integrator.tprev = integrator.t
        end
        integrator.t = end_time

        # Save the pending `saveat` times before the end time first: the state there is
        # the one from before the final callback pass, and saving them afterwards would
        # also put them after any saves the callbacks make at the end time.
        if integrator.saveat !== nothing && !isempty(integrator.saveat)
            # Split to help prediction
            while integrator.cur_saveat <= length(integrator.saveat) &&
                  integrator.saveat[integrator.cur_saveat] < integrator.t
                save_current_state!(integrator, integrator.saveat[integrator.cur_saveat])
                integrator.cur_saveat += 1
            end
        end

        # check callbacks one last time
        if !(integrator.opts.callback.discrete_callbacks isa Tuple{})
            DiffEqBase.apply_discrete_callback!(integrator,
                integrator.opts.callback.discrete_callbacks...)
        end

        if integrator.save_end &&
           (isempty(integrator.sol.t) || integrator.sol.t[end] != end_time)
            save_current_state!(integrator, end_time)
        end
    end

    DiffEqBase.finalize!(integrator.opts.callback, integrator.u, integrator.t, integrator)
    if integrator.save_end
        SciMLBase.save_final_discretes!(integrator, integrator.opts.callback)
    end

    if integrator.sol.retcode === ReturnCode.Default
        integrator.sol = SciMLBase.solution_new_retcode(integrator.sol, ReturnCode.Success)
    end
end

"""
    check_continuous_callback_error(callback)

Check if the callback contains any continuous callbacks and throw an informative error.
SSAStepper only supports DiscreteCallbacks for event detection.
"""
function check_continuous_callback_error(callback)
    if callback === nothing
        return nothing
    end

    if callback isa DiffEqBase.ContinuousCallback
        error("SSAStepper does not support continuous callbacks. Only DiscreteCallbacks " *
              "are supported for event detection with SSAStepper. Please use an ODE/SDE " *
              "solver (e.g., Tsit5()) if you need continuous event detection.")
    elseif callback isa DiffEqBase.CallbackSet
        n_continuous = length(callback.continuous_callbacks)
        if n_continuous > 0
            error("SSAStepper does not support continuous callbacks (found $n_continuous " *
                  "continuous callback$(n_continuous > 1 ? "s" : "")). Only DiscreteCallbacks " *
                  "are supported for event detection with SSAStepper. Please use an ODE/SDE " *
                  "solver (e.g., Tsit5()) if you need continuous event detection.")
        end
    end
    return nothing
end

# Normalize SSAStepper's `alias` keyword to `(alias_u0, alias_p, alias_tstops)`, each
# `nothing` (solver default), `true` (aliasing permitted), or `false`. `alias_f` and
# `alias_du0` have no effect: SSAStepper never evaluates `f` and has no `du0`.
ssa_alias_choices(::Nothing) = (nothing, nothing, nothing)
ssa_alias_choices(alias::Bool) = (alias, alias, alias)
function ssa_alias_choices(alias::SciMLBase.DiscreteAliasSpecifier)
    (alias.alias_u0, alias.alias_p, nothing)
end
function ssa_alias_choices(alias::SciMLBase.ODEAliasSpecifier)
    (alias.alias_u0, alias.alias_p, alias.alias_tstops)
end
function ssa_alias_choices(alias)
    throw(ArgumentError("SSAStepper's `alias` keyword accepts `nothing`, a `Bool`, a " *
                        "`SciMLBase.DiscreteAliasSpecifier`, or a " *
                        "`SciMLBase.ODEAliasSpecifier`, but received a $(typeof(alias)). " *
                        "`alias` controls `u0`, `p`, and `tstops`; it never controls " *
                        "jump state."))
end

# Returns `(tstops, alias_tstops, copied_tstops)` for the integrator. `alias_tstops`
# means the integrator may insert into `tstops`, and `copied_tstops` means it already
# owns them. Only a caller's `Vector` of the time type can be aliased; other containers
# are materialized, as aliasing is permitted but not required.
function ssa_init_tstops(tstops, ::Type{T}, alias_tstops) where {T}
    if tstops isa Vector{T}
        if alias_tstops === true
            return tstops, true, false
        elseif alias_tstops === false
            return copy(tstops), true, true
        else
            # Default: never mutate the caller's array; copy before the first insertion.
            return tstops, false, false
        end
    end
    owned = (tstops isa Number) ? T[tstops] : collect(T, tstops)
    return owned, true, true
end

function SciMLBase.__init(jump_prob::JumpProblem,
        alg::SSAStepper;
        save_start = true,
        save_end = true,
        seed = nothing,
        rng = nothing,
        alias = nothing,
        alias_jump = KeywordNotPassed(),
        saveat = nothing,
        callback = nothing,
        tstops = nothing,
        numsteps_hint = 100)
    check_alias_jump_removed(jump_prob, alias_jump)
    alias_u0, alias_p, alias_tstops = ssa_alias_choices(alias)

    if !(jump_prob.prob isa DiscreteProblem)
        error("SSAStepper only supports DiscreteProblems.")
    end
    prob = jump_prob.prob

    # Check for continuous callbacks in the jump system
    isempty(jump_prob.jump_callback.continuous_callbacks) ||
        error("SSAStepper does not support continuous callbacks in the jump system. " *
              "Please use an ODE/SDE solver over ODE or SDE problems instead.")

    # Check for continuous callbacks passed via kwargs (from JumpProblem constructor or solve)
    check_continuous_callback_error(callback)

    _rng = resolve_rng(rng, seed)

    # The problem's jump state is always reused and re-initialized below; callers
    # isolate problems used concurrently (e.g. via `deepcopy` or an `EnsembleProblem`).
    cb = jump_prob.jump_callback.discrete_callbacks[end]
    opts = (callback = CallbackSet(callback),)

    # Defaults match OrdinaryDiffEq: copy `u0` and reuse `p`.
    u0 = (alias_u0 === true) ? prob.u0 : recursivecopy(prob.u0)
    p = (alias_p === false) ? recursivecopy(prob.p) : prob.p

    if save_start
        t = [prob.tspan[1]]
        u = [recursivecopy(u0)]
    else
        t = typeof(prob.tspan[1])[]
        u = typeof(prob.u0)[]
    end
    save_everystep = any(cb.save_positions)

    sol = SciMLBase.build_solution(prob, alg, t, u, dense = save_everystep,
        calculate_error = false,
        stats = DiffEqBase.Stats(0),
        interp = SciMLBase.ConstantInterpolation(t, u))

    _saveat = (saveat isa Number) ? (prob.tspan[1]:saveat:prob.tspan[2]) : saveat
    if _saveat !== nothing && !isempty(_saveat) && _saveat[1] == prob.tspan[1]
        cur_saveat = 2
    else
        cur_saveat = 1
    end

    if _saveat !== nothing && !isempty(_saveat)
        sizehint!(u, length(_saveat) + 1)
        sizehint!(t, length(_saveat) + 1)
    elseif save_everystep
        sizehint!(u, numsteps_hint)
        sizehint!(t, numsteps_hint)
    else
        sizehint!(u, save_start + save_end)
        sizehint!(t, save_start + save_end)
    end

    tdir = sign(prob.tspan[2] - prob.tspan[1])
    (tdir <= 0) &&
        error("The time interval to solve over is non-increasing, i.e. tspan[2] <= tspan[1]. This is not allowed for pure jump problem.")

    # Stash callable tstops (e.g. SymbolicTstops); use an owned empty vector for init.
    tType = eltype(prob.tspan)
    if tstops === nothing
        callable_tstops = nothing
        _tstops, _alias_tstops, copied_tstops = tType[], true, true
    elseif tstops isa AbstractArray || tstops isa Tuple || tstops isa Number
        callable_tstops = nothing
        _tstops, _alias_tstops, copied_tstops = ssa_init_tstops(tstops, tType, alias_tstops)
    else
        callable_tstops = tstops
        _tstops, _alias_tstops, copied_tstops = tType[], true, true
    end

    # With `save_uprev = true`, `uprev` and the scratch cache are distinct from `u0`, even
    # when `u0` aliases the problem's initial condition.
    uprev, tmp_cache = ssa_uprev_storage(alg, u0)

    integrator = SSAIntegrator(prob.f, u0, prob.tspan[1], prob.tspan[1], uprev, tdir,
        p, sol, 1, prob.tspan[1], cb, _saveat, save_everystep,
        save_end, cur_saveat, opts, _tstops, 1, false, true, _alias_tstops,
        copied_tstops, _rng, tmp_cache)
    cb.initialize(cb, integrator.u, prob.tspan[1], integrator)
    DiffEqBase.initialize!(opts.callback, integrator.u, prob.tspan[1], integrator)
    if save_start
        SciMLBase.save_discretes_if_enabled!(integrator, opts.callback; skip_duplicates = true)
    end

    # Evaluate callable tstops now that callbacks are initialized and parameters finalized.
    if callable_tstops !== nothing
        for ts in callable_tstops(SII.parameter_values(integrator), prob.tspan)
            add_tstop!(integrator, ts)
        end
    end

    integrator
end

function DiffEqBase.get_tstops(integrator::SSAIntegrator)
    @view integrator.tstops[(integrator.tstops_idx):end]
end
DiffEqBase.get_tstops_array(integrator::SSAIntegrator) = DiffEqBase.get_tstops(integrator)

# ODE integrators seem to add tf into tstops which SSAIntegrator does not do
# so must account for it here.
function DiffEqBase.get_tstops_max(integrator::SSAIntegrator)
    tstops = DiffEqBase.get_tstops_array(integrator)
    tf = integrator.sol.prob.tspan[2]
    if !isempty(tstops)
        return max(maximum(tstops), tf)
    else
        return tf
    end
end

function DiffEqBase.add_tstop!(integrator::SSAIntegrator, tstop)
    if tstop > integrator.t
        future_tstops = @view integrator.tstops[(integrator.tstops_idx):end]
        insert_index = integrator.tstops_idx + searchsortedfirst(future_tstops, tstop) - 1

        # if not aliasing and have not already copied the user tstops array
        if !integrator.alias_tstops && !integrator.copied_tstops
            integrator.copied_tstops = true
            integrator.tstops = copy(integrator.tstops)
        end

        Base.insert!(integrator.tstops, insert_index, tstop)
    end
    nothing
end

# The Jump aggregators should not register the next jump through add_tstop! for SSAIntegrator
# such that we can achieve maximum performance
@inline function register_next_jump_time!(integrator::SSAIntegrator,
        p::AbstractSSAJumpAggregator, t)
    integrator.tstop = p.next_jump_time
    nothing
end

function DiffEqBase.step!(integrator::SSAIntegrator)
    store_uprev!(integrator)
    integrator.tprev = integrator.t
    next_jump_time = integrator.tstop > integrator.t ? integrator.tstop :
                     typemax(integrator.tstop)

    doaffect = false
    if !isempty(integrator.tstops) &&
       integrator.tstops_idx <= length(integrator.tstops) &&
       integrator.tstops[integrator.tstops_idx] < next_jump_time
        integrator.t = integrator.tstops[integrator.tstops_idx]
        integrator.tstops_idx += 1
    else
        integrator.t = integrator.tstop
        doaffect = true # delay effect until after saveat
    end

    @inbounds if integrator.saveat !== nothing && !isempty(integrator.saveat)
        # Split to help prediction
        while integrator.cur_saveat <= length(integrator.saveat) &&
              integrator.saveat[integrator.cur_saveat] < integrator.t
            save_current_state!(integrator, integrator.saveat[integrator.cur_saveat])
            integrator.cur_saveat += 1
        end
    end

    # FP error means the new time may equal the old if the next jump time is
    # sufficiently small, hence we add this check to execute jumps until
    # this is no longer true.
    integrator.u_modified = true
    while integrator.t == integrator.tstop
        doaffect && integrator.cb.affect!(integrator)
    end

    jump_modified_u = integrator.u_modified

    if !(integrator.opts.callback.discrete_callbacks isa Tuple{})
        discrete_modified,
        saved_in_cb = DiffEqBase.apply_discrete_callback!(integrator,
            integrator.opts.callback.discrete_callbacks...)
    else
        saved_in_cb = false
    end

    !saved_in_cb && jump_modified_u && savevalues!(integrator)

    nothing
end

function DiffEqBase.savevalues!(integrator::SSAIntegrator, force = false)
    saved, savedexactly = false, false

    # No saveat in here since it would only use previous values,
    # so in the specific case of SSAStepper it's already handled

    if integrator.save_everystep || force
        saved = true
        savedexactly = true
        save_current_state!(integrator, integrator.t)
    end

    saved, savedexactly
end

function should_continue_solve(integrator::SSAIntegrator)
    end_time = integrator.sol.prob.tspan[2]

    # we continue the solve if there is a tstop between now and end_time
    has_tstop = !isempty(integrator.tstops) &&
                integrator.tstops_idx <= length(integrator.tstops) &&
                integrator.tstops[integrator.tstops_idx] < end_time

    # we continue the solve if there will be a jump between now and end_time
    has_jump = integrator.t < integrator.tstop < end_time

    integrator.keep_stepping && (has_jump || has_tstop)
end

function reset_aggregated_jumps!(integrator::SSAIntegrator, uprev = nothing; kwargs...)
    if haskey(kwargs, :update_jump_params)
        throw(ArgumentError("`update_jump_params` keyword argument has been removed. " *
                            "Rate updates are now handled automatically by `initialize!` " *
                            "via `fill_scaled_rates!`."))
    end
    reset_aggregated_jumps!(integrator, uprev, integrator.cb)
    nothing
end

function DiffEqBase.terminate!(integrator::SSAIntegrator, retcode = ReturnCode.Terminated)
    integrator.keep_stepping = false
    integrator.sol = SciMLBase.solution_new_retcode(integrator.sol, retcode)
    nothing
end

function SciMLBase.isdenseplot(sol::ODESolution{
        T, N, uType, uType2, DType, tType, rateType, discType, P,
        <:SSAStepper}) where {T, N, uType, uType2, DType, tType, rateType, discType, P}
    sol.dense
end
