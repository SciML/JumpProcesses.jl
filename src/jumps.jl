"""
$(TYPEDEF)

Defines a jump process with a rate (i.e. hazard, intensity, or propensity) that
does not *explicitly* depend on time. More precisely, one where the rate
function is constant *between* the occurrence of jumps. For detailed examples
and usage information, see the
- [Tutorial](https://docs.sciml.ai/JumpProcesses/stable/tutorials/discrete_stochastic_example/)

## Fields

$(FIELDS)

## Examples
Suppose `u[1]` gives the number of particles and `p[1]` the probability per time
each particle can decay away. A corresponding `ConstantRateJump` for this jump
process is
```julia
rate(u,p,t) = p[1]*u[1]
affect!(integrator) = integrator.u[1] -= 1
crj = ConstantRateJump(rate, affect!)
```
Notice, here that `rate` changes in time, but is constant between the occurrence
of jumps (when `u[1]` will decrease).
"""
struct ConstantRateJump{F1, F2} <: AbstractJump
    """Function `rate(u,p,t)` that returns the jump's current rate."""
    rate::F1
    """Function `affect(integrator)` that updates the state for one occurrence of the jump."""
    affect!::F2
end

"""
$(TYPEDEF)

Defines a jump process with a rate (i.e. hazard, intensity, or propensity) that may
explicitly depend on time. More precisely, one where the rate function is allowed to change
*between* the occurrence of jumps. For detailed examples and usage information, see the
- [Tutorial](https://docs.sciml.ai/JumpProcesses/stable/tutorials/discrete_stochastic_example/)

Note that two types of `VariableRateJump`s are currently supported, with different
performance characteritistics.
- A general `VariableRateJump` or `VariableRateJump` will refer to one in which only `rate`
  and `affect` functions are specified.

    * These are the most general in what they can represent, but require the use of an
      `ODEProblem` or `SDEProblem` whose underlying timestepper handles their evolution in
      time (via the callback interface).
    * This is the least performant jump type in simulations.

- Bounded `VariableRateJump`s require passing the keyword arguments `urate` and
  `rateinterval`, corresponding to functions `urate(u, p, t)` and `rateinterval(u, p, t)`,
  see below. These must calculate a time window over which the rate function is bounded by a
  constant. Note that it is ok if the rate bound would be violated within the time interval
  due to a change in `u` arising from another `ConstantRateJump`, `MassActionJump` or
  *bounded* `VariableRateJump` being executed, as the chosen aggregator will then handle
  recalculating the rate bound and interval. *However, if the bound could be violated within
  the time interval due to a change in `u` arising from continuous dynamics such as a
  coupled ODE, SDE, or a general `VariableRateJump`, bounds should not be given.* This
  ensures the jump is classified as a general `VariableRateJump` and properly handled. One
  can also optionally provide a lower bound function, `lrate(u, p, t)`, via the `lrate`
  keyword argument. This can lead to increased performance. The validity of the lower bound
  should hold under the same conditions and rate interval as `urate`.

    * Bounded `VariableRateJump`s can currently be used in the `Coevolve` aggregator, and
      can therefore be efficiently simulated in pure-jump `DiscreteProblem`s using the
      `SSAStepper` time-stepper.
    * These can be substantially more performant than general `VariableRateJump`s without
      the rate bound functions.

Reemphasizing, the additional user provided functions leveraged by bounded
`VariableRateJumps`, `urate(u, p, t)`, `rateinterval(u, p, t)`, and the optional `lrate(u,
p, t)` require that
- For `s` in `[t, t + rateinterval(u, p, t)]`, we have that `lrate(u, p, t) <= rate(u, p, s)
  <= urate(u, p, t)`.
- It is ok if these bounds would be violated during the time window due to another
  `ConstantRateJump`, `MassActionJump` or bounded `VariableRateJump` occurring. However,
  they must remain valid if `u` changes for any other reason (for example, due to
  continuous dynamics like ODEs, SDEs, or general `VariableRateJump`s).

## Fields

$(FIELDS)

## Examples
Suppose `u[1]` gives the number of particles and `t*p[1]` the probability per time each
particle can decay away. A corresponding `VariableRateJump` for this jump process is
```julia
rate(u,p,t) = t*p[1]*u[1]
affect!(integrator) = integrator.u[1] -= 1
vrj = VariableRateJump(rate, affect!)
```

To define a bounded `VariableRateJump` that can be used with supporting aggregators such as
`Coevolve`, we must define bounds and a rate interval:
```julia
rateinterval(u,p,t) = (1 / p[1]) * 2
rate(u,p,t) = t * p[1] * u[1]
lrate(u, p, t) = rate(u, p, t)
urate(u,p,t) = rate(u, p, t + rateinterval(u,p,t))
affect!(integrator) = integrator.u[1] -= 1
vrj = VariableRateJump(rate, affect!; lrate = lrate, urate = urate,
                                      rateinterval = rateinterval)
```

## Notes
- When using an aggregator that supports bounded `VariableRateJump`s, `DiscreteProblem` can
  be used. Otherwise, `ODEProblem` or `SDEProblem` must be used.
- **When not using aggregators that support bounded `VariableRateJump`s, or when there are
  general `VariableRateJump`s, `integrator`s store an effective state type that wraps the
  main state vector.** See [`ExtendedJumpArray`](@ref) for details on using this object. In
  this case all `ConstantRateJump`, `VariableRateJump` and callback `affect!` functions
  receive an integrator with `integrator.u` an [`ExtendedJumpArray`](@ref).
- Salis H., Kaznessis Y.,  Accurate hybrid stochastic simulation of a system of coupled
  chemical or biochemical reactions, Journal of Chemical Physics, 122 (5),
  DOI:10.1063/1.1835951 is used for calculating jump times with `VariableRateJump`s within
  ODE/SDE integrators.
"""
struct VariableRateJump{R, F, R2, R3, R4, I, T, T2} <: AbstractJump
    """Function `rate(u,p,t)` that returns the jump's current rate given state
    `u`, parameters `p` and time `t`."""
    rate::R
    """Function `affect!(integrator)` that updates the state for one occurrence
    of the jump given `integrator`."""
    affect!::F
    """Optional function `lrate(u, p, t)` that computes a lower bound on the rate in the
    interval `t` to `t + rateinterval(u, p, t)` at time `t` given state `u` and parameters
    `p`. This bound must rigorously hold during the time interval as long as another
    `ConstantRateJump`, `MassActionJump`, or *bounded* `VariableRateJump` has not been
    sampled. When using aggregators that support bounded `VariableRateJump`s, currently only
    `Coevolve`, providing a lower-bound can lead to improved performance.
    """
    lrate::R2
    """Optional function `urate(u, p, t)` for general `VariableRateJump`s, but is required
    to define a bounded `VariableRateJump`, which can be used with supporting aggregators,
     currently only `Coevolve`, and offers improved computational performance. Computes an
    upper bound for the rate in the interval `t` to `t + rateinterval(u, p, t)` at time `t`
    given state `u` and parameters `p`. This bound must rigorously hold during the time
    interval as long as another `ConstantRateJump`, `MassActionJump`, or *bounded*
    `VariableRateJump` has not been sampled. """
    urate::R3
    """Optional function `rateinterval(u, p, t)` for general `VariableRateJump`s, but is
    required to define a bounded `VariableRateJump`, which can be used with supporting
     aggregators, currently only `Coevolve`, and offers improved computational performance.
    Computes the time interval from time `t` over which the `urate` and `lrate` bounds will
    hold, `t` to `t + rateinterval(u, p, t)`, given state `u` and parameters `p`. This bound
    must rigorously hold during the time interval as long as another `ConstantRateJump`,
    `MassActionJump`, or *bounded* `VariableRateJump` has not been sampled. """
    rateinterval::R4
    idxs::I
    rootfind::Bool
    interp_points::Int
    save_positions::Tuple{Bool, Bool}
    abstol::T
    reltol::T2
end

isbounded(::VariableRateJump) = true
isbounded(::VariableRateJump{R, F, R2, Nothing}) where {R, F, R2} = false
haslrate(::VariableRateJump) = true
haslrate(::VariableRateJump{R, F, Nothing}) where {R, F} = false
nullrate(u, p, t::T) where {T <: Number} = zero(T)

"""
```
function VariableRateJump(rate, affect!; lrate = nothing, urate = nothing,
                                         rateinterval = nothing, rootfind = true,
                                         idxs = nothing,
                                         save_positions = (false, true),
                                         interp_points = 10,
                                         abstol = 1e-12, reltol = 0)
```
"""
function VariableRateJump(rate, affect!;
        lrate = nothing, urate = nothing,
        rateinterval = nothing, rootfind = true,
        idxs = nothing,
        save_positions = (false, true),
        interp_points = 10,
        abstol = 1e-12, reltol = 0)
    if !(urate !== nothing && rateinterval !== nothing) &&
       !(urate === nothing && rateinterval === nothing)
        error("`urate` and `rateinterval` must both be `nothing`, or must both be defined.")
    end

    if lrate !== nothing
        (urate !== nothing) ||
            error("If a lower bound rate, `lrate`, is given than an upper bound rate, `urate`, and rate interval, `rateinterval`, must also be provided.")
    end

    VariableRateJump(rate, affect!, lrate, urate, rateinterval, idxs, rootfind,
        interp_points, save_positions, abstol, reltol)
end

"""
$(TYPEDEF)

Representation for encoding rates and multiple simultaneous jumps for use in τ-leaping type
methods.

### Constructors
- `RegularJump(rate, c, numjumps; mark_dist = nothing)`

## Fields

$(FIELDS)

## Examples
```julia
function rate!(out, u, p, t)
    out[1] = (0.1 / 1000.0) * u[1] * u[2]
    out[2] = 0.01u[2]
    nothing
end

const dc = zeros(3,2)
function c(du, u, p, t, counts, mark) 
    mul!(du, dc, counts)
    nothing
end

rj = RegularJump(rate!, c, 2)

## Notes
- `mark_dist` is not currently used or supported in τ-leaping methods.
```
"""
struct RegularJump{iip, R, C, MD}
    """
    Function `rate!(rate_vals, u, p, t)` that returns the current rates, i.e.
    intensities or propensities, for all possible jumps in `rate_vals`.
    """
    rate::R
    """
    Function `c(du, u, p, t, counts, mark)` that executes the `i`th jump `counts[i]` times,
    saving the output in `du[i]`.
    """
    c::C
    """ Number of jumps in the system."""
    numjumps::Int
    """ A distribution for marks. Not currently used or supported. """
    mark_dist::MD
    function RegularJump{iip}(rate, c, numjumps::Int; mark_dist = nothing) where {iip}
        new{iip, typeof(rate), typeof(c), typeof(mark_dist)}(rate, c, numjumps, mark_dist)
    end
end

DiffEqBase.isinplace(::RegularJump{iip, R, C, MD}) where {iip, R, C, MD} = iip

function RegularJump(rate, c, numjumps::Int; kwargs...)
    RegularJump{DiffEqBase.isinplace(rate, 4)}(rate, c, numjumps; kwargs...)
end

# deprecate old call
function RegularJump(rate, c, dc::AbstractMatrix; constant_c = false, mark_dist = nothing)
    @warn("The RegularJump interface has changed to be matrix-free. See the documentation for more details.")
    function _c(du, u, p, t, counts, mark)
        c(dc, u, p, t, mark)
        mul!(du, dc, counts)
    end
    RegularJump{true}(rate, _c, size(dc, 2); mark_dist = mark_dist)
end

"""
$(TYPEDEF)

Optimized representation for `ConstantRateJump`s that can be represented in mass
action form, offering improved performance within jump algorithms compared to
`ConstantRateJump`. For detailed examples and usage information, see the
- [Main
  Docs](https://docs.sciml.ai/JumpProcesses/stable/jump_types/#Defining-a-Mass-Action-Jump)
- [Tutorial](https://docs.sciml.ai/JumpProcesses/stable/tutorials/discrete_stochastic_example/)

### Constructors
- `MassActionJump(reactant_stoich, net_stoich; scale_rates = true, param_idxs = nothing)`

Here `reactant_stoich` denotes the reactant stoichiometry for each reaction and
`net_stoich` the net stoichiometry for each reaction.

## Fields

$(FIELDS)

## Keyword Arguments
- `scale_rates = true`, whether to rescale the reaction rate constants according
  to the stoichiometry.
- `nocopy = false`, whether the `MassActionJump` can alias the `scaled_rates` and
  `reactant_stoch` from the input. Note, if `scale_rates=true` this will
  potentially modify both of these.
- `param_idxs = nothing`, indexes in the parameter vector, `JumpProblem.prob.p`,
  that correspond to each reaction's rate.

See the tutorial and main docs for details.

## Examples
An SIR model with `S + I --> 2I` at rate β as the first reaction and `I --> R`
at rate ν as the second reaction can be encoded by
```julia
p        = (β=1e-4, ν=.01)
u0       = [999, 1, 0]       # (S,I,R)
tspan    = (0.0, 250.0)
rateidxs = [1, 2]           # i.e. [β,ν]
reactant_stoich =
[
  [1 => 1, 2 => 1],         # 1*S and 1*I
  [2 => 1]                  # 1*I
]
net_stoich =
[
  [1 => -1, 2 => 1],        # -1*S and 1*I
  [2 => -1, 3 => 1]         # -1*I and 1*R
]
maj = MassActionJump(reactant_stoich, net_stoich; param_idxs=rateidxs)
prob = DiscreteProblem(u0, tspan, p)
jprob = JumpProblem(prob, Direct(), maj)
```

## Notes
- By default, reaction rates are rescaled when constructing the `MassActionJump`
  as explained in the [main
  docs](https://docs.sciml.ai/JumpProcesses/stable/jump_types/#Defining-a-Mass-Action-Jump).
  Disable this with the kwarg `scale_rates=false`.
- Also see the [main
  docs](https://docs.sciml.ai/JumpProcesses/stable/jump_types/#Defining-a-Mass-Action-Jump)
  for how to specify reactions with no products or no reactants.


"""
struct MassActionJump{T, S, U, V} <: AbstractMassActionJump
    """The (scaled) reaction rate constants."""
    scaled_rates::T
    """The reactant stoichiometry vectors."""
    reactant_stoch::S
    """The net stoichiometry vectors."""
    net_stoch::U
    """Parameter mapping functor to identify reaction rate constants with parameters in `p` vectors."""
    param_mapper::V

    function MassActionJump{T, S, U, V}(rates::T, rs_in::S, ns::U, pmapper::V,
            scale_rates::Bool, useiszero::Bool,
            nocopy::Bool) where {T <: AbstractVector, S, U, V}
        sr = nocopy ? rates : copy(rates)
        rs = nocopy ? rs_in : copy(rs_in)
        for i in eachindex(rs)
            if useiszero && (length(rs[i]) == 1) && iszero(rs[i][1][1])
                rs[i] = typeof(rs[i])()
            end
        end

        if scale_rates && !isempty(sr)
            scalerates!(sr, rs)
        end
        new(sr, rs, ns, pmapper)
    end
    function MassActionJump{Nothing, Vector{S},
            Vector{U}, V}(::Nothing, rs_in::Vector{S},
            ns::Vector{U}, pmapper::V,
            scale_rates::Bool,
            useiszero::Bool,
            nocopy::Bool) where {S <: AbstractVector,
            U <: AbstractVector, V}
        rs = nocopy ? rs_in : copy(rs_in)
        for i in eachindex(rs)
            if useiszero && (length(rs[i]) == 1) && iszero(rs[i][1][1])
                rs[i] = typeof(rs[i])()
            end
        end
        new(nothing, rs, ns, pmapper)
    end
    function MassActionJump{T, S, U, V}(rate::T, rs_in::S, ns::U, pmapper::V,
            scale_rates::Bool, useiszero::Bool,
            nocopy::Bool) where {T <: Number, S, U, V}
        rs = rs_in
        if useiszero && (length(rs) == 1) && iszero(rs[1][1])
            rs = typeof(rs)()
        end
        sr = scale_rates ? scalerate(rate, rs) : rate
        new(sr, rs, ns, pmapper)
    end
    function MassActionJump{Nothing, S, U, V}(::Nothing, rs_in::S, ns::U, pmapper::V,
            scale_rates::Bool, useiszero::Bool,
            nocopy::Bool) where {S, U, V}
        rs = rs_in
        if useiszero && (length(rs) == 1) && iszero(rs[1][1])
            rs = typeof(rs)()
        end
        new(nothing, rs, ns, pmapper)
    end
end
function MassActionJump(usr::T, rs::S, ns::U, pmapper::V; scale_rates = true,
        useiszero = true, nocopy = false) where {T, S, U, V}
    MassActionJump{T, S, U, V}(usr, rs, ns, pmapper, scale_rates, useiszero, nocopy)
end
function MassActionJump(usr::T, rs, ns; scale_rates = true, useiszero = true,
        nocopy = false) where {T <: AbstractVector}
    MassActionJump(usr, rs, ns, nothing; scale_rates = scale_rates, useiszero = useiszero,
        nocopy = nocopy)
end
function MassActionJump(usr::T, rs, ns; scale_rates = true, useiszero = true,
        nocopy = false) where {T <: Number}
    MassActionJump(usr, rs, ns, nothing; scale_rates = scale_rates, useiszero = useiszero,
        nocopy = nocopy)
end

# with parameter indices or mapping, multiple jump case
function MassActionJump(rs, ns; param_idxs = nothing, param_mapper = nothing,
        scale_rates = true, useiszero = true, nocopy = false)
    if param_mapper === nothing
        (param_idxs === nothing) &&
            error("If no parameter indices are given via param_idxs, an explicit parameter mapping must be passed in via param_mapper.")
        pmapper = MassActionJumpParamMapper(param_idxs)
    else
        (param_idxs !== nothing) &&
            error("Only one of param_idxs and param_mapper should be passed.")
        pmapper = param_mapper
    end

    MassActionJump(nothing, nocopy ? rs : copy(rs), ns, pmapper; scale_rates = scale_rates,
        useiszero = useiszero, nocopy = true)
end

using_params(maj::MassActionJump{T, S, U, Nothing}) where {T, S, U} = false
using_params(maj::MassActionJump) = true
using_params(maj::Nothing) = false
@inline get_num_majumps(maj::MassActionJump) = length(maj.net_stoch)
@inline get_num_majumps(maj::Nothing) = 0

struct MassActionJumpParamMapper{U}
    param_idxs::U
end

# create the initial parameter vector for use in a MassActionJump
# Note these are unscaled
function (ratemap::MassActionJumpParamMapper{U})(params) where {U <: AbstractArray}
    [params[pidx] for pidx in ratemap.param_idxs]
end

# Note this is unscaled
function (ratemap::MassActionJumpParamMapper{U})(params) where {U <: Int}
    params[ratemap.param_idxs]
end

# update a maj with parameter vectors
function (ratemap::MassActionJumpParamMapper{U})(maj::MassActionJump, newparams;
        scale_rates,
        kwargs...) where {U <: AbstractArray}
    for i in 1:get_num_majumps(maj)
        maj.scaled_rates[i] = newparams[ratemap.param_idxs[i]]
    end
    scale_rates && scalerates!(maj.scaled_rates, maj.reactant_stoch)
    nothing
end

function to_collection(ratemap::MassActionJumpParamMapper{Int})
    MassActionJumpParamMapper([ratemap.param_idxs])
end

function Base.merge!(pmap1::MassActionJumpParamMapper{U},
        pmap2::MassActionJumpParamMapper{U}) where {U <: AbstractVector}
    append!(pmap1.param_idxs, pmap2.param_idxs)
end

function Base.merge!(pmap1::MassActionJumpParamMapper{U},
        pmap2::MassActionJumpParamMapper{V}) where {U <: AbstractVector,
        V <: Int}
    push!(pmap1.param_idxs, pmap2.param_idxs)
end

function Base.merge(pmap1::MassActionJumpParamMapper{Int},
        pmap2::MassActionJumpParamMapper{Int})
    MassActionJumpParamMapper([pmap1.param_idxs, pmap2.param_idxs])
end

"""
update_parameters!(maj::MassActionJump, newparams; scale_rates=true)

Updates the passed in MassActionJump with the parameter values in `newparams`.

Notes:

  - Requires the jump to have been constructed with a user-passed `param_idxs` or `param_mapper`.
  - `scale_rates=true` will scale the parameter representing the jump rate by an
    appropriate combinatoric factor. i.e for 3A --> B at rate k it will scale
    k --> k/3!.
"""
function update_parameters!(maj::MassActionJump, newparams; scale_rates = true, kwargs...)
    (maj.param_mapper === nothing) &&
        error("MassActionJumps must be constructed with param_idxs or a param_mapper to be updateable.")
    maj.param_mapper(maj, newparams; scale_rates, kwargs)
end

"""
$(TYPEDEF)

Defines a collection of jumps that should collectively be included in a simulation.

## Fields

$(FIELDS)

## Examples
Here we construct two jumps, store them in a `JumpSet`, and then simulate the resulting
process.
```julia
using JumpProcesses, OrdinaryDiffEq

rate1(u,p,t) = p[1]
affect1!(integrator) = (integrator.u[1] += 1)
crj = ConstantRateJump(rate1, affect1!)

rate2(u,p,t) = (t/(1+t))*p[2]*u[1]
affect2!(integrator) = (integrator.u[1] -= 1)
vrj = VariableRateJump(rate2, affect2!)

jset = JumpSet(crj, vrj)

f!(du,u,p,t) = (du .= 0)
u0 = [0.0]
p = (20.0, 2.0)
tspan = (0.0, 200.0)
oprob = ODEProblem(f!, u0, tspan, p)
jprob = JumpProblem(oprob, Direct(), jset)
sol = solve(jprob, Tsit5())
```
"""
struct JumpSet{T1, T2, T3, T4} <: AbstractJump
    """Collection of [`VariableRateJump`](@ref)s"""
    variable_jumps::T1
    """Collection of [`ConstantRateJump`](@ref)s"""
    constant_jumps::T2
    """Collection of [`RegularJump`](@ref)s"""
    regular_jump::T3
    """Collection of [`MassActionJump`](@ref)s"""
    massaction_jump::T4
end
function JumpSet(vj, cj, rj, maj::MassActionJump{S, T, U, V}) where {S <: Number, T, U, V}
    JumpSet(vj, cj, rj, check_majump_type(maj))
end

JumpSet(jump::ConstantRateJump) = JumpSet((), (jump,), nothing, nothing)
JumpSet(jump::VariableRateJump) = JumpSet((jump,), (), nothing, nothing)
JumpSet(jump::RegularJump) = JumpSet((), (), jump, nothing)
JumpSet(jump::AbstractMassActionJump) = JumpSet((), (), nothing, jump)
function JumpSet(; variable_jumps = (), constant_jumps = (),
        regular_jumps = nothing, massaction_jumps = nothing)
    JumpSet(variable_jumps, constant_jumps, regular_jumps, massaction_jumps)
end
JumpSet(jb::Nothing) = JumpSet()

# For Varargs, use recursion to make it type-stable
function JumpSet(jumps::AbstractJump...)
    JumpSet(split_jumps((), (), nothing, nothing, jumps...)...)
end

# handle vector of mass action jumps
function JumpSet(vjs, cjs, rj, majv::Vector{T}) where {T <: MassActionJump}
    if isempty(majv)
        error("JumpSets do not accept empty mass action jump collections; use \"nothing\" instead.")
    end

    maj = setup_majump_to_merge(majv[1].scaled_rates, majv[1].reactant_stoch,
        majv[1].net_stoch, majv[1].param_mapper)
    for i in 2:length(majv)
        massaction_jump_combine(maj, majv[i])
    end

    JumpSet(vjs, cjs, rj, maj)
end

@inline get_num_majumps(jset::JumpSet) = get_num_majumps(jset.massaction_jump)
@inline num_majumps(jset::JumpSet) = get_num_majumps(jset)

@inline function num_crjs(jset::JumpSet)
    (jset.constant_jumps !== nothing) ? length(jset.constant_jumps) : 0
end

@inline function num_vrjs(jset::JumpSet)
    (jset.variable_jumps !== nothing) ? length(jset.variable_jumps) : 0
end

@inline function num_bndvrjs(jset::JumpSet)
    (jset.variable_jumps !== nothing) ? count(isbounded, jset.variable_jumps) : 0
end

@inline function num_continvrjs(jset::JumpSet)
    (jset.variable_jumps !== nothing) ? count(!isbounded, jset.variable_jumps) : 0
end

num_jumps(jset::JumpSet) = num_majumps(jset) + num_crjs(jset) + num_vrjs(jset)
num_discretejumps(jset::JumpSet) = num_majumps(jset) + num_crjs(jset) + num_bndvrjs(jset)
num_cdiscretejumps(jset::JumpSet) = num_majumps(jset) + num_crjs(jset)

@inline split_jumps(vj, cj, rj, maj) = vj, cj, rj, maj
@inline function split_jumps(vj, cj, rj, maj, v::VariableRateJump, args...)
    split_jumps((vj..., v), cj, rj, maj, args...)
end
@inline function split_jumps(vj, cj, rj, maj, c::ConstantRateJump, args...)
    split_jumps(vj, (cj..., c), rj, maj, args...)
end
@inline function split_jumps(vj, cj, rj, maj, c::RegularJump, args...)
    split_jumps(vj, cj, regular_jump_combine(rj, c), maj, args...)
end
@inline function split_jumps(vj, cj, rj, maj, c::MassActionJump, args...)
    split_jumps(vj, cj, rj, massaction_jump_combine(maj, c), args...)
end
@inline function split_jumps(vj, cj, rj, maj, j::JumpSet, args...)
    split_jumps((vj..., j.variable_jumps...),
        (cj..., j.constant_jumps...),
        regular_jump_combine(rj, j.regular_jump),
        massaction_jump_combine(maj, j.massaction_jump), args...)
end

regular_jump_combine(rj1::RegularJump, rj2::Nothing) = rj1
regular_jump_combine(rj1::Nothing, rj2::RegularJump) = rj2
regular_jump_combine(rj1::Nothing, rj2::Nothing) = rj1
function regular_jump_combine(rj1::RegularJump, rj2::RegularJump)
    error("Only one regular jump is allowed in a JumpSet")
end

# functionality to merge two mass action jumps together
function check_majump_type(maj::MassActionJump{S, T, U, V}) where {S <: Number, T, U, V}
    setup_majump_to_merge(maj.scaled_rates, maj.reactant_stoch, maj.net_stoch,
        maj.param_mapper)
end
function check_majump_type(maj::MassActionJump{Nothing, T, U, V}) where {T, U, V}
    setup_majump_to_merge(maj.scaled_rates, maj.reactant_stoch, maj.net_stoch,
        maj.param_mapper)
end

# if given containers of rates and stoichiometry directly create a jump
function setup_majump_to_merge(sr::T, rs::AbstractVector{S}, ns::AbstractVector{U},
        pmapper) where {T <: AbstractVector, S <: AbstractArray,
        U <: AbstractArray}
    MassActionJump(sr, rs, ns, pmapper; scale_rates = false)
end

# if just given the data for one jump (and not in a container) wrap in a vector
function setup_majump_to_merge(sr::S, rs::T, ns::U,
        pmapper) where {S <: Number, T <: AbstractArray,
        U <: AbstractArray}
    MassActionJump([sr], [rs], [ns],
        (pmapper === nothing) ? pmapper : to_collection(pmapper);
        scale_rates = false)
end

# if no rate field setup yet
function setup_majump_to_merge(::Nothing, rs::T, ns::U,
        pmapper) where {T <: AbstractArray, U <: AbstractArray}
    MassActionJump(nothing, [rs], [ns],
        (pmapper === nothing) ? pmapper : to_collection(pmapper);
        scale_rates = false)
end

# when given a collection of reactions to add to maj
function majump_merge!(maj::MassActionJump{U, <:AbstractVector{V}, <:AbstractVector{W}, X},
        sr::U, rs::AbstractVector{V}, ns::AbstractVector{W},
        param_mapper) where {U <: Union{AbstractVector, Nothing},
        V <: AbstractVector, W <: AbstractVector, X}
    (U <: AbstractVector) && append!(maj.scaled_rates, sr)
    append!(maj.reactant_stoch, rs)
    append!(maj.net_stoch, ns)
    if maj.param_mapper === nothing
        (param_mapper === nothing) ||
            error("Error, trying to merge a MassActionJump with a parameter mapping to one without a parameter mapping.")
    else
        merge!(maj.param_mapper, param_mapper)
    end
    maj
end

# when given a single jump's worth of data to add to maj
function majump_merge!(maj::MassActionJump{U, V, W, X}, sr::T, rs::S1, ns::S2,
        param_mapper) where {T <: Union{Number, Nothing},
        S1 <: AbstractArray, S2 <: AbstractArray,
        U <: Union{AbstractVector{T}, Nothing},
        V <: AbstractVector{S1},
        W <: AbstractVector{S2}, X}
    (T <: Number) && push!(maj.scaled_rates, sr)
    push!(maj.reactant_stoch, rs)
    push!(maj.net_stoch, ns)
    if maj.param_mapper === nothing
        (param_mapper === nothing) ||
            error("Error, trying to merge a MassActionJump with a parameter mapping to one without a parameter mapping.")
    else
        merge!(maj.param_mapper, param_mapper)
    end

    maj
end

# when maj only stores a single jump's worth of data (and not in a collection)
# create a new jump with the merged data stored in vectors
function majump_merge!(maj::MassActionJump{T, S, U, V}, sr::T, rs::S, ns::U,
        param_mapper::V) where {T <: Union{Number, Nothing},
        S <: AbstractArray{<:Pair},
        U <: AbstractArray{<:Pair}, V}
    rates = (T <: Nothing) ? nothing : [maj.scaled_rates, sr]
    if maj.param_mapper === nothing
        (param_mapper === nothing) ||
            error("Error, trying to merge a MassActionJump with a parameter mapping to one without a parameter mapping.")
        return MassActionJump(rates, [maj.reactant_stoch, rs], [maj.net_stoch, ns],
            param_mapper; scale_rates = false)
    else
        return MassActionJump(rates, [maj.reactant_stoch, rs], [maj.net_stoch, ns],
            merge(maj.param_mapper, param_mapper); scale_rates = false)
    end
end

massaction_jump_combine(maj1::MassActionJump, maj2::Nothing) = maj1
massaction_jump_combine(maj1::Nothing, maj2::MassActionJump) = maj2
massaction_jump_combine(maj1::Nothing, maj2::Nothing) = maj1
function massaction_jump_combine(maj1::MassActionJump, maj2::MassActionJump)
    majump_merge!(maj1, maj2.scaled_rates, maj2.reactant_stoch, maj2.net_stoch,
        maj2.param_mapper)
end

##### helper methods for unpacking rates and affects! from constant jumps #####
function get_jump_info_tuples(jumps)
    if (jumps !== nothing) && !isempty(jumps)
        rates = ((c.rate for c in jumps)...,)
        affects! = ((c.affect! for c in jumps)...,)
    else
        rates = ()
        affects! = ()
    end

    rates, affects!
end

function get_jump_info_fwrappers(u, p, t, constant_jumps)
    RateWrapper = FunctionWrappers.FunctionWrapper{typeof(t),
        Tuple{typeof(u), typeof(p), typeof(t)}}

    if (constant_jumps !== nothing) && !isempty(constant_jumps)
        rates = [RateWrapper(c.rate) for c in constant_jumps]
        affects! = Any[(x -> (c.affect!(x); nothing)) for c in constant_jumps]
    else
        rates = Vector{RateWrapper}()
        affects! = Any[]
    end

    rates, affects!
end

function get_jump_info_vr_fwrappers(vjumps, prob)
    t, u = prob.tspan[1], prob.u0
    RateWrapper = FunctionWrappers.FunctionWrapper{typeof(t),
        Tuple{typeof(u), typeof(prob.p), typeof(t)}}

    if (vjumps !== nothing) && !isempty(vjumps)
        rates = [RateWrapper(c.rate) for c in vjumps]
        affects! = Any[(x -> (c.affect!(x); nothing)) for c in vjumps]
    else
        rates = Vector{RateWrapper}()
        affects! = Any[]
    end

    rates, affects!
end
