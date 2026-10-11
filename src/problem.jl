function isinplace_jump(p, rj)
    if p isa DiscreteProblem && p.f === SciMLBase.DISCRETE_INPLACE_DEFAULT &&
       rj !== nothing
        # Just a default discrete problem f, so don't use it for iip
        DiffEqBase.isinplace(rj)
    else
        DiffEqBase.isinplace(p)
    end
end

"""
$(TYPEDEF)

Defines a collection of jump processes to associate with another problem type.

  - [Documentation Page](https://docs.sciml.ai/JumpProcesses/stable/jump_types/)
  - [Tutorial
    Page](https://docs.sciml.ai/JumpProcesses/stable/tutorials/discrete_stochastic_example/)
  - [FAQ
    Page](https://docs.sciml.ai/JumpProcesses/stable/tutorials/discrete_stochastic_example/#FAQ)

### Constructors

`JumpProblem`s can be constructed by first building another problem type to which the jumps
will be associated. For example, to  simulate a collection of jump processes for which the
transition rates are constant *between* jumps (called [`ConstantRateJump`](@ref)s or
[`MassActionJump`](@ref)s), we must first construct a
[`DiscreteProblem`](https://docs.sciml.ai/DiffEqDocs/stable/types/discrete_types/)

```julia
prob = DiscreteProblem(u0, tspan, p)
```

where `u0` is the initial condition, `p` the parameters and `tspan` the time span. If we
wanted to have the jumps coupled with a system of ODEs, or have transition rates with
explicit time dependence, we would use an `ODEProblem` instead that defines the ODE portion
of the dynamics.

Given `prob` we define the jumps via

  - `JumpProblem(prob, aggregator::AbstractAggregatorAlgorithm, jumps::JumpSet ; kwargs...)`
  - `JumpProblem(prob, aggregator::AbstractAggregatorAlgorithm, jumps...; kwargs...)`

Here `aggregator` specifies the underlying algorithm for calculating next jump times and
types, for example [`Direct`](@ref). The collection of different `AbstractJump` types can
then be passed within a single [`JumpSet`](@ref) or as subsequent sequential arguments.

### Fields

$(FIELDS)

## Keyword Arguments

  - `save_positions=(true,true)` when including variable rates and `(false,true)` for constant
    rates, specifies whether to save the system's state (before, after) the jump occurs. **NOTE**,
    this only controls saving for non-`VariableRateJump`s. `VariableRateJump` saving is controlled
    at the individual jump level via the value of this kwarg passed to the individual `VariableRateJump`s
    constructor when it is created.
  - `spatial_system`, for spatial problems the underlying spatial structure.
  - `hopping_constants`, for spatial problems the spatial transition rate coefficients.
  - `use_vrj_bounds = true`, set to false to disable handling bounded `VariableRateJump`s with
    a supporting aggregator (such as `Coevolve`). They will then be handled via the continuous
    integration interface, and treated like general `VariableRateJump`s.
  - `vr_aggregator`, indicates the aggregator to use for sampling variable rate jumps. Current
    default is `VR_FRM`.
  - `tstops`, time stops to pass through to the solver. Can be an `AbstractVector` of times
    or a callable `(p, tspan) -> times`.
  - `callback`, callbacks to run in every solve of the problem. They are combined with any
    `callback` passed to `solve` or `init`, which replaces them instead when
    `merge_callbacks = false` is also passed. Do not store callbacks on the wrapped problem
    `prob`; see the warning below.

!!! warning "Callbacks on the wrapped problem are not supported"

    Pass callbacks to the `JumpProblem` constructor, with the `callback` keyword, or to
    `solve`/`init`. Do not store them on the wrapped problem, for example with
    `ODEProblem(f, u0, tspan, p; callback)`: the constructor raises an `ArgumentError` if
    `prob` stores a `callback`, since only some solvers would run it. Use
    `remake(prob; callback = nothing)` to remove callbacks from an existing problem.

Please see the [tutorial
page](https://docs.sciml.ai/JumpProcesses/stable/tutorials/discrete_stochastic_example/) in
the DifferentialEquations.jl [docs](https://docs.sciml.ai/JumpProcesses/stable/) for usage
examples and commonly asked questions.

!!! warning "Thread safety and problem reuse"

    A `JumpProblem` stores mutable jump state: the jump aggregator and its callbacks.
    Solvers owned by JumpProcesses, [`SSAStepper`](@ref) and the OrdinaryDiffEq ODE/DAE
    pathways, reuse this state directly and re-initialize it at every `init`; they never
    copy it. A `JumpProblem` is therefore **not thread-safe** and supports only one active
    solve or integrator at a time:

      - Repeated serial solves of one problem are safe, and avoid copying.
      - Concurrent solves from multiple threads or tasks, and integrators that are alive at
        the same time, each need an independent copy, such as `deepcopy(jprob)`.
      - `remake(jprob; ...)` shares the original's jump state, so remade problems are not
        independent for concurrent use.
      - `EnsembleProblem` supplies isolation. `EnsembleSerial` reuses the problem
        sequentially. `EnsembleThreads` with `safetycopy = false` copies the problem once
        per spawned task. `safetycopy = true` copies it before every trajectory, before
        `prob_func` is called. A `prob_func` that returns some other, shared problem
        defeats these copies, and callbacks passed as `solve` keyword arguments are shared
        by all trajectories.
      - `EnsembleDistributed` with `pmap_batch_size > 1` and `safetycopy = false` can run
        concurrent tasks on one worker's copy of the problem; use `safetycopy = true` or
        `pmap_batch_size = 1`.
      - StochasticDiffEq SDE/RODE solvers apply their own jump-state copying policy.

    CPU ensemble methods can assign separate RNGs to trajectories; a custom `rng_func` must
    not return one shared mutable RNG. Separate RNGs alone do not isolate mutable jump
    state, and kernel ensembles have their own backend RNG rules. See the
    [ensembles and problem reuse
    tutorial](https://docs.sciml.ai/JumpProcesses/stable/tutorials/ensembles_and_problem_reuse/)
    for patterns that balance performance and safety.
"""
mutable struct JumpProblem{iip, P, A, C, J <: Union{Nothing, AbstractJumpAggregator}, J1,
    J2, J3, J4, K} <: SciMLBase.AbstractJumpProblem{P, J}
    """
    The type of problem to couple the jumps to. For a pure jump process use `DiscreteProblem`, to couple to ODEs, `ODEProblem`, etc.
    """
    prob::P
    """
    The aggregator algorithm that determines the next jump times and types for `ConstantRateJump`s and `MassActionJump`s. Examples include `Direct`.
    """
    aggregator::A
    """
    The underlying state data associated with the chosen aggregator.
    """
    discrete_jump_aggregation::J
    """
    `CallBackSet` with the underlying `ConstantRate` and `VariableRate` jumps.
    """
    jump_callback::C
    """
    The `ConstantRateJump`s.
    """
    constant_jumps::J1
    """
    The `VariableRateJump`s.
    """
    variable_jumps::J2
    """
    The `RegularJump`s.
    """
    regular_jump::J3
    """
    The `MassActionJump`s.
    """
    massaction_jump::J4
    """
    kwargs to pass on to solve call.
    """
    kwargs::K
end

# Callbacks stored on the wrapped problem would run only on solvers that `init` the wrapped
# problem (OrdinaryDiffEq, including `FunctionMap`); `SSAStepper` and StochasticDiffEq never
# read them. Reject them, so that every solver runs the same callbacks.
function check_no_wrapped_callbacks(prob)
    if hasproperty(prob, :kwargs) && haskey(prob.kwargs, :callback) &&
       !is_empty_callback(prob.kwargs[:callback])
        throw(ArgumentError("""
            Callbacks stored on the problem wrapped by a `JumpProblem` are not supported, \
            but the wrapped `$(nameof(typeof(prob)))` stores a `callback` keyword argument. \
            OrdinaryDiffEq solvers would run these callbacks, while `SSAStepper` and \
            StochasticDiffEq solvers would ignore them. Pass the callbacks to the \
            `JumpProblem` constructor instead, `JumpProblem(prob, aggregator, jumps...; \
            callback)`, or to `solve`/`init`. To remove callbacks from an existing problem, \
            use `remake(prob; callback = nothing)`."""))
    end
    nothing
end

is_empty_callback(::Nothing) = true
function is_empty_callback(cb::CallbackSet)
    isempty(cb.continuous_callbacks) && isempty(cb.discrete_callbacks)
end
is_empty_callback(_) = false

# Also used by `remake`, so a new wrapped problem is checked too.
function JumpProblem(p::P, a::A, dj::J, jc::C, cj::J1, vj::J2, rj::J3, mj::J4,
        kwargs::K) where {P, A, J, C, J1, J2, J3, J4, K}
    check_no_wrapped_callbacks(p)
    iip = isinplace_jump(p, rj)
    JumpProblem{iip, P, A, C, J, J1, J2, J3, J4, K}(p, a, dj, jc, cj, vj, rj, mj,
        kwargs)
end

######## remaking ######

# for a problem where prob.u0 is an ExtendedJumpArray, create an ExtendedJumpArray that 
# aliases and resets prob.u0.jump_u while having newu0 as the new u component.
function remake_extended_u0(prob, newu0)
    jump_u = prob.u0.jump_u
    ttype = eltype(prob.tspan)
    @. jump_u = zero(ttype)
    ExtendedJumpArray(newu0, jump_u)
end

Base.@pure remaker_of(prob::T) where {T <: JumpProblem} = SciMLBase.parameterless_type(T)
"""
    remake(jprob::JumpProblem; u0 = missing, p = missing, tspan, prob)

Construct a `JumpProblem` with a new initial condition `u0`, parameters `p`, time span
`tspan`, or wrapped problem `prob`.

A new wrapped problem `prob` must not store callbacks, as for the `JumpProblem`
constructor; an `ArgumentError` is raised if it does.

The new problem shares the original's jump aggregator and jump callbacks rather than
copying them. Solvers owned by JumpProcesses re-initialize this state from the problem being
solved at every `init`, so the original and remade problems can be solved one after
another. They must not be solved concurrently, or used for integrators that are alive at
the same time; use `deepcopy` to obtain an independent problem. `SortingDirect`'s learned
search order is also shared.
"""
function DiffEqBase.remake(jprob::JumpProblem; u0 = missing, p = missing,
        interpret_symbolicmap = true, use_defaults = false, kwargs...)
    T = remaker_of(jprob)

    errmesg = """
    JumpProblems can currently only be remade with new u0, p, tspan or prob fields. To change other fields create a new JumpProblem. Feel free to open an issue on JumpProcesses to discuss further.
    """
    !issubset(keys(kwargs), (:tspan, :prob)) && error(errmesg)

    if :prob ∉ keys(kwargs)
        # Update u0 when we are wrapping via ExtendedJumpArrays. If the user passes an
        # ExtendedJumpArray we assume they properly initialized it
        prob = jprob.prob
        if (prob.u0 isa ExtendedJumpArray) && (u0 !== missing)
            if u0 isa ExtendedJumpArray
                # User provided ExtendedJumpArray directly - use as-is (current behavior)
                # This gives users full control over jump_u values
                final_u0 = u0
            else
                # Resolve symbolic maps if needed (handles [:X => 3.0], [sys.X => 3.0], Dict, etc.)
                resolved_u0, _ = SciMLBase.updated_u0_p(
                    prob, u0, missing; interpret_symbolicmap, use_defaults)

                # Extract state values and wrap with a zeroed jump_u; the `VR_FRM` callbacks
                # draw new thresholds from the integrator's RNG at every `init`.
                # This handles both:
                #   - symbolic maps (resolved_u0 is ExtendedJumpArray with updated u)
                #   - numeric arrays (resolved_u0 is Vector{Float64})
                state_vals = resolved_u0 isa ExtendedJumpArray ? resolved_u0.u : resolved_u0

                # Validate type compatibility
                if typeof(state_vals) != typeof(prob.u0.u)
                    error("Passed in u0 is incompatible with current u0 which has type: $(typeof(prob.u0.u)).")
                end

                final_u0 = remake_extended_u0(prob, state_vals)
            end
            newprob = DiffEqBase.remake(
                prob; u0 = final_u0, p, interpret_symbolicmap, use_defaults, kwargs...)
        else
            newprob = DiffEqBase.remake(
                prob; u0, p, interpret_symbolicmap, use_defaults, kwargs...)
        end

    else
        ((u0 !== missing) || (p !== missing) || (:tspan ∈ keys(kwargs))) &&
            error("If remaking a JumpProblem you can not pass both prob and any of u0, p, or tspan.")
        newprob = kwargs[:prob]

        # when passing a new wrapped problem directly we require u0 has the correct type
        (typeof(newprob.u0) == typeof(jprob.prob.u0)) ||
            error("The new u0 within the passed prob does not have the same type as the existing u0. Please pass a u0 of type $(typeof(jprob.prob.u0)).")
    end

    T(newprob, jprob.aggregator, jprob.discrete_jump_aggregation, jprob.jump_callback,
        jprob.constant_jumps, jprob.variable_jumps, jprob.regular_jump,
        jprob.massaction_jump, jprob.kwargs)
end

DiffEqBase.isinplace(::JumpProblem{iip}) where {iip} = iip
JumpProblem(prob::JumpProblem) = prob

function JumpProblem(prob, jumps::ConstantRateJump; kwargs...)
    JumpProblem(prob, JumpSet(jumps); kwargs...)
end
function JumpProblem(prob, jumps::VariableRateJump; kwargs...)
    JumpProblem(prob, JumpSet(jumps); kwargs...)
end
function JumpProblem(prob, jumps::RegularJump; kwargs...)
    JumpProblem(prob, JumpSet(jumps); kwargs...)
end
function JumpProblem(prob, jumps::MassActionJump; kwargs...)
    JumpProblem(prob, JumpSet(jumps); kwargs...)
end
function JumpProblem(prob, jumps::AbstractJump...; kwargs...)
    JumpProblem(prob, JumpSet(jumps...); kwargs...)
end

function JumpProblem(prob, aggregator::AbstractAggregatorAlgorithm,
        jumps::ConstantRateJump; kwargs...)
    JumpProblem(prob, aggregator, JumpSet(jumps); kwargs...)
end
function JumpProblem(prob, aggregator::AbstractAggregatorAlgorithm,
        jumps::VariableRateJump; kwargs...)
    JumpProblem(prob, aggregator, JumpSet(jumps); kwargs...)
end
function JumpProblem(prob, aggregator::AbstractAggregatorAlgorithm, jumps::RegularJump;
        kwargs...)
    JumpProblem(prob, aggregator, JumpSet(jumps); kwargs...)
end
function JumpProblem(prob, aggregator::AbstractAggregatorAlgorithm,
        jumps::AbstractMassActionJump; kwargs...)
    JumpProblem(prob, aggregator, JumpSet(jumps); kwargs...)
end
function JumpProblem(prob, aggregator::AbstractAggregatorAlgorithm, jumps::AbstractJump...;
        kwargs...)
    JumpProblem(prob, aggregator, JumpSet(jumps...); kwargs...)
end
function JumpProblem(prob, jumps::JumpSet; vartojumps_map = nothing,
        jumptovars_map = nothing, dep_graph = nothing,
        spatial_system = nothing, hopping_constants = nothing, kwargs...)
    ps = (; vartojumps_map, jumptovars_map, dep_graph, spatial_system, hopping_constants)
    aggtype = select_aggregator(jumps; ps...)
    return JumpProblem(prob, aggtype(), jumps; ps..., kwargs...)
end

# this makes it easier to test the aggregator selection
function JumpProblem(prob, aggregator::NullAggregator, jumps::JumpSet; kwargs...)
    JumpProblem(prob, jumps; kwargs...)
end

make_kwarg(; kwargs...) = kwargs

function JumpProblem(prob, aggregator::AbstractAggregatorAlgorithm, jumps::JumpSet;
        vr_aggregator::VariableRateAggregator = VR_FRM(),
        save_positions = prob isa SciMLBase.AbstractDiscreteProblem ?
                         (false, true) : (true, true),
        spatial_system = nothing, hopping_constants = nothing,
        callback = nothing, tstops = nothing, use_vrj_bounds = true, kwargs...)
    if haskey(kwargs, :rng)
        throw(ArgumentError("`rng` is no longer a keyword argument for `JumpProblem`. Pass `rng` to `solve` or `init` instead, e.g. `solve(jprob, SSAStepper(); rng = my_rng)`."))
    end
    if haskey(kwargs, :scale_rates)
        throw(ArgumentError("`scale_rates` is no longer a keyword argument for `JumpProblem`. Set `scale_rates` on the `MassActionJump` directly instead."))
    end
    if haskey(kwargs, :useiszero)
        throw(ArgumentError("`useiszero` is no longer a keyword argument for `JumpProblem`. Set `useiszero` on the `MassActionJump` directly instead."))
    end
    haskey(kwargs, :alias_jump) && throw(ArgumentError(ALIAS_JUMP_REMOVED_MSG))
    # Check before spatial flattening, which rebuilds the problem without its keywords.
    check_no_wrapped_callbacks(prob)

    # keep the original MAJ (including {Nothing} parameterized MAJs);
    # fill_scaled_rates! in each aggregator's initialize! handles rate setup
    maj = jumps.massaction_jump

    ## Spatial jumps handling
    if spatial_system !== nothing && hopping_constants !== nothing
        (num_crjs(jumps) == num_vrjs(jumps) == 0) ||
            error("Spatial aggregators only support MassActionJumps currently.")

        if is_spatial(aggregator)
            kwargs = merge((; hopping_constants, spatial_system), kwargs)
        else
            prob, maj = flatten(maj, prob, spatial_system, hopping_constants; kwargs...)
        end
    end

    ndiscjumps = get_num_majumps(maj) + num_crjs(jumps)
    crjs = jumps.constant_jumps

    # separate bounded variable rate jumps *if* the aggregator can use them
    if use_vrj_bounds && supports_variablerates(aggregator) && (num_bndvrjs(jumps) > 0)
        bvrjs = filter(isbounded, jumps.variable_jumps)
        cvrjs = filter(!isbounded, jumps.variable_jumps)
        kwargs = merge((; variable_jumps = bvrjs), kwargs)
        ndiscjumps += length(bvrjs)
    else
        bvrjs = nothing
        cvrjs = jumps.variable_jumps
    end

    t, end_time, u = prob.tspan[1], prob.tspan[2], prob.u0

    # handle majs, crjs, and bounded vrjs
    if (ndiscjumps == 0) && !is_spatial(aggregator)
        disc_agg = nothing
        constant_jump_callback = CallbackSet()
    else
        disc_agg = aggregate(aggregator, u, prob.p, t, end_time, crjs, maj,
            save_positions; kwargs...)
        constant_jump_callback = DiscreteCallback(disc_agg)
    end

    # handle any remaining vrjs
    if length(cvrjs) > 0
        # Handle variable rate jumps based on vr_aggregator
        new_prob, variable_jump_callback = configure_jump_problem(prob, vr_aggregator,
            jumps, cvrjs)
    else
        new_prob = prob
        variable_jump_callback = CallbackSet()
        cvrjs = JumpSet().variable_jumps
    end

    jump_cbs = CallbackSet(constant_jump_callback, variable_jump_callback)
    iip = isinplace_jump(prob, jumps.regular_jump)
    solkwargs = tstops === nothing ? make_kwarg(; callback) : make_kwarg(; callback, tstops)

    JumpProblem{iip, typeof(new_prob), typeof(aggregator), typeof(jump_cbs),
        typeof(disc_agg), typeof(crjs), typeof(cvrjs), typeof(jumps.regular_jump),
        typeof(maj), typeof(solkwargs)}(new_prob, aggregator, disc_agg,
        jump_cbs, crjs, cvrjs, jumps.regular_jump, maj, solkwargs)
end

# Special dispatch for PureLeaping aggregator - bypasses all aggregation
function JumpProblem(prob, aggregator::PureLeaping, jumps::JumpSet;
        save_positions = prob isa SciMLBase.AbstractDiscreteProblem ?
                         (false, true) : (true, true),
        spatial_system = nothing, hopping_constants = nothing,
        callback = nothing, tstops = nothing, kwargs...)
    if haskey(kwargs, :rng)
        throw(ArgumentError("`rng` is no longer a keyword argument for `JumpProblem`. Pass `rng` to `solve` or `init` instead, e.g. `solve(jprob, SSAStepper(); rng = my_rng)`."))
    end
    if haskey(kwargs, :scale_rates)
        throw(ArgumentError("`scale_rates` is no longer a keyword argument for `JumpProblem`. Set `scale_rates` on the `MassActionJump` directly instead."))
    end
    if haskey(kwargs, :useiszero)
        throw(ArgumentError("`useiszero` is no longer a keyword argument for `JumpProblem`. Set `useiszero` on the `MassActionJump` directly instead."))
    end
    haskey(kwargs, :alias_jump) && throw(ArgumentError(ALIAS_JUMP_REMOVED_MSG))
    check_no_wrapped_callbacks(prob)

    # Validate no spatial systems (not currently supported)
    (spatial_system !== nothing || hopping_constants !== nothing) &&
        error("PureLeaping does not currently support spatial problems.")

    # keep the original MAJ (including {Nothing} parameterized MAJs);
    # fill_scaled_rates! in the tau-leaping solver handles rate setup
    maj = jumps.massaction_jump

    # For PureLeaping, all jumps are handled by the tau-leaping solver
    # No discrete jump aggregation or variable rate callbacks are created
    disc_agg = nothing
    jump_cbs = CallbackSet()

    # Store all jump types for access by tau-leaping solver
    crjs = jumps.constant_jumps
    vrjs = jumps.variable_jumps

    iip = isinplace_jump(prob, jumps.regular_jump)
    solkwargs = tstops === nothing ? make_kwarg(; callback) : make_kwarg(; callback, tstops)

    JumpProblem{iip, typeof(prob), typeof(aggregator), typeof(jump_cbs),
        typeof(disc_agg), typeof(crjs), typeof(vrjs), typeof(jumps.regular_jump),
        typeof(maj), typeof(solkwargs)}(prob, aggregator, disc_agg,
        jump_cbs, crjs, vrjs, jumps.regular_jump, maj, solkwargs)
end

aggregator(jp::JumpProblem{iip, P, A}) where {iip, P, A} = A

@inline function extend_tstops!(tstops, jp::JumpProblem)
    !(jp.jump_callback.discrete_callbacks isa Tuple{}) &&
        push!(tstops, jp.jump_callback.discrete_callbacks[1].condition.next_jump_time)
end

### Displays
num_constant_rate_jumps(aggregator::AbstractSSAJumpAggregator) = length(aggregator.rates)

function Base.summary(io::IO, prob::JumpProblem)
    type_color, no_color = SciMLBase.get_colorizers(io)
    print(io,
        type_color, nameof(typeof(prob)),
        no_color, " with problem ",
        type_color, nameof(typeof(prob.prob)),
        no_color, " with aggregator ",
        type_color, typeof(prob.aggregator))
end
function Base.show(io::IO, mime::MIME"text/plain", A::JumpProblem)
    summary(io, A)
    println(io)
    println(io, "Number of jumps with discrete aggregation: ",
        A.discrete_jump_aggregation === nothing ? 0 :
        num_constant_rate_jumps(A.discrete_jump_aggregation))
    println(io, "Number of jumps with continuous aggregation: ", length(A.variable_jumps))
    nmajs = (A.massaction_jump !== nothing) ? get_num_majumps(A.massaction_jump) : 0
    println(io, "Number of mass action jumps: ", nmajs)
    if A.regular_jump !== nothing
        println(io, "Have a regular jump")
    end
end
