function isinplace_jump(p, rj)
    if p isa DiscreteProblem && p.f === DiffEqBase.DISCRETE_INPLACE_DEFAULT &&
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
prob = DiscreteProblem(u0, p, tspan)
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
- `rng`, the random number generator to use. Defaults to Julia's built-in generator.
- `save_positions=(true,true)` when including variable rates and `(false,true)` for constant
  rates, specifies whether to save the system's state (before, after) the jump occurs.
- `spatial_system`, for spatial problems the underlying spatial structure.
- `hopping_constants`, for spatial problems the spatial transition rate coefficients.
- `use_vrj_bounds = true`, set to false to disable handling bounded `VariableRateJump`s with
  a supporting aggregator (such as `Coevolve`). They will then be handled via the continuous
  integration interface, and treated like general `VariableRateJump`s.
- `vr_aggregator`, indicates the aggregator to use for sampling variable rate jumps. Current
  default is `VRFRMODE`.

Please see the [tutorial
page](https://docs.sciml.ai/JumpProcesses/stable/tutorials/discrete_stochastic_example/) in
the DifferentialEquations.jl [docs](https://docs.sciml.ai/JumpProcesses/stable/) for usage
examples and commonly asked questions.
"""
mutable struct JumpProblem{iip, P, A, C, J <: Union{Nothing, AbstractJumpAggregator}, J2,
    J3, J4, R, K} <: DiffEqBase.AbstractJumpProblem{P, J}
    """The type of problem to couple the jumps to. For a pure jump process use `DiscreteProblem`, to couple to ODEs, `ODEProblem`, etc."""
    prob::P
    """The aggregator algorithm that determines the next jump times and types for `ConstantRateJump`s and `MassActionJump`s. Examples include `Direct`."""
    aggregator::A
    """The underlying state data associated with the chosen aggregator."""
    discrete_jump_aggregation::J
    """`CallBackSet` with the underlying `ConstantRate` and `VariableRate` jumps."""
    jump_callback::C
    """The `VariableRateJump`s."""
    variable_jumps::J2
    """The `RegularJump`s."""
    regular_jump::J3
    """The `MassActionJump`s."""
    massaction_jump::J4
    """The random number generator to use."""
    rng::R
    """kwargs to pass on to solve call."""
    kwargs::K
end
function JumpProblem(p::P, a::A, dj::J, jc::C, vj::J2, rj::J3, mj::J4,
        rng::R, kwargs::K) where {P, A, J, C, J2, J3, J4, R, K}
    iip = isinplace_jump(p, rj)
    JumpProblem{iip, P, A, C, J, J2, J3, J4, R, K}(p, a, dj, jc, vj, rj, mj, rng, kwargs)
end

######## remaking ######

# for a problem where prob.u0 is an ExtendedJumpArray, create an ExtendedJumpArray that 
# aliases and resets prob.u0.jump_u while having newu0 as the new u component.
function remake_extended_u0(prob, newu0, rng)
    jump_u = prob.u0.jump_u
    ttype = eltype(prob.tspan)
    @. jump_u = -randexp(rng, ttype)
    ExtendedJumpArray(newu0, jump_u)
end

Base.@pure remaker_of(prob::T) where {T <: JumpProblem} = DiffEqBase.parameterless_type(T)
function DiffEqBase.remake(jprob::JumpProblem; kwargs...)
    T = remaker_of(jprob)

    errmesg = """
    JumpProblems can currently only be remade with new u0, p, tspan or prob fields. To change other fields create a new JumpProblem. Feel free to open an issue on JumpProcesses to discuss further.
    """
    !issubset(keys(kwargs), (:u0, :p, :tspan, :prob)) && error(errmesg)

    if :prob ∉ keys(kwargs)
        # Update u0 when we are wrapping via ExtendedJumpArrays. If the user passes an
        # ExtendedJumpArray we assume they properly initialized it
        prob = jprob.prob
        if (prob.u0 isa ExtendedJumpArray) && (:u0 in keys(kwargs))
            newu0 = kwargs[:u0]
            # if newu0 is of the wrapped type, initialize a new ExtendedJumpArray
            if typeof(newu0) == typeof(prob.u0.u)
                u0 = remake_extended_u0(prob, newu0, jprob.rng)
                _kwargs = @set! kwargs[:u0] = u0
            elseif typeof(newu0) != typeof(prob.u0)
                error("Passed in u0 is incompatible with current u0 which has type: $(typeof(prob.u0.u)).")
            else
                _kwargs = kwargs
            end
            newprob = DiffEqBase.remake(jprob.prob; _kwargs...)
        else
            newprob = DiffEqBase.remake(jprob.prob; kwargs...)
        end

        # if the parameters were changed we must remake the MassActionJump too
        if (:p ∈ keys(kwargs)) && using_params(jprob.massaction_jump)
            update_parameters!(jprob.massaction_jump, newprob.p; kwargs...)
        end
    else
        any(k -> k in keys(kwargs), (:u0, :p, :tspan)) &&
            error("If remaking a JumpProblem you can not pass both prob and any of u0, p, or tspan.")
        newprob = kwargs[:prob]

        # when passing a new wrapped problem directly we require u0 has the correct type
        (typeof(newprob.u0) == typeof(jprob.prob.u0)) ||
            error("The new u0 within the passed prob does not have the same type as the existing u0. Please pass a u0 of type $(typeof(jprob.prob.u0)).")

        # we can't know if p was changed, so we must remake the MassActionJump
        if using_params(jprob.massaction_jump)
            update_parameters!(jprob.massaction_jump, newprob.p; kwargs...)
        end
    end

    T(newprob, jprob.aggregator, jprob.discrete_jump_aggregation, jprob.jump_callback,
        jprob.variable_jumps, jprob.regular_jump, jprob.massaction_jump, jprob.rng,
        jprob.kwargs)
end

# for updating parameters in JumpProblems to update MassActionJumps
function SII.finalize_parameters_hook!(prob::JumpProblem, p)
    if using_params(prob.massaction_jump)
        update_parameters!(prob.massaction_jump, SII.parameter_values(prob))
    end
    nothing
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
        vr_aggregator::VariableRateAggregator = VRFRMODE(),
        save_positions = prob isa DiffEqBase.AbstractDiscreteProblem ?
                         (false, true) : (true, true),
        rng = DEFAULT_RNG, scale_rates = true, useiszero = true,
        spatial_system = nothing, hopping_constants = nothing,
        callback = nothing, use_vrj_bounds = true, kwargs...)

    # initialize the MassActionJump rate constants with the user parameters
    if using_params(jumps.massaction_jump)
        rates = jumps.massaction_jump.param_mapper(prob.p)
        maj = MassActionJump(rates, jumps.massaction_jump.reactant_stoch,
            jumps.massaction_jump.net_stoch,
            jumps.massaction_jump.param_mapper; scale_rates = scale_rates,
            useiszero = useiszero,
            nocopy = true)
    else
        maj = jumps.massaction_jump
    end

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
        disc_agg = aggregate(aggregator, u, prob.p, t, end_time, jumps.constant_jumps, maj,
            save_positions, rng; kwargs...)
        constant_jump_callback = DiscreteCallback(disc_agg)
    end

    # handle any remaining vrjs
    if length(cvrjs) > 0
        # Handle variable rate jumps based on vr_aggregator
        new_prob, variable_jump_callback, cont_agg = configure_jump_problem(prob, 
            vr_aggregator, jumps, cvrjs; rng)  
    else
        new_prob = prob
        variable_jump_callback = CallbackSet()
        cont_agg = JumpSet().variable_jumps
    end

    jump_cbs = CallbackSet(constant_jump_callback, variable_jump_callback)
    iip = isinplace_jump(prob, jumps.regular_jump)
    solkwargs = make_kwarg(; callback)

    JumpProblem{iip, typeof(new_prob), typeof(aggregator),
        typeof(jump_cbs), typeof(disc_agg),
        typeof(cont_agg),
        typeof(jumps.regular_jump),
        typeof(maj), typeof(rng), typeof(solkwargs)}(new_prob, aggregator, disc_agg,
        jump_cbs, cont_agg,
        jumps.regular_jump, maj, rng,
        solkwargs)
end

aggregator(jp::JumpProblem{iip, P, A, C, J}) where {iip, P, A, C, J} = A

@inline function extend_tstops!(tstops,
        jp::JumpProblem{P, A, C, J, J2}) where {P, A, C, J, J2}
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
