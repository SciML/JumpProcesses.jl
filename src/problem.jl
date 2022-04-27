function isinplace_jump(p,rj)
    if p isa DiscreteProblem && p.f === DiffEqBase.DISCRETE_INPLACE_DEFAULT && rj !== nothing
        # Just a default discrete problem f, so don't use it for iip
        DiffEqBase.isinplace(rj)
    else
        DiffEqBase.isinplace(p)
    end
end

"""
$(TYPEDEF)

Defines a collection of jump processes to associate with another problem type.
- [Documentation Page](https://diffeq.sciml.ai/stable/types/jump_types/) 
- [Tutorial
  Page](https://diffeq.sciml.ai/stable/tutorials/discrete_stochastic_example/)
- [FAQ
  Page](https://diffeq.sciml.ai/stable/tutorials/discrete_stochastic_example/#FAQ)

### Constructors

`JumpProblem`s can be constructed by first building another problem type to
which the jumps will be associated. For example, to  simulate a collection of
jump processes for which the transition rates do not explicitly depend on time
(called `ConstantRateJump`s or `MassActionJump`s), we must first construct a
`DiscreteProblem`
```julia
prob = DiscreteProblem(u0, p, tspan)
```
where `u0` is the initial condition, `p` the parameters and `tspan` the time
span. If we wanted to have the jumps coupled with a system of ODEs, or have
transition rates with explicit time dependence, we would use an `ODEProblem`
instead that defines the ODE portion of the dynamics. 

Given `prob` we define the jumps via
- `JumpProblem(prob, aggregator::AbstractAggregatorAlgorithm, jumps::JumpSet ;
  kwargs...)`
- `JumpProblem(prob, aggregator::AbstractAggregatorAlgorithm, jumps...;
  kwargs...)`

Here `aggregator` specifies the underlying algorithm for calculating next jump
times and types, for example `Direct`. The collection of different
`AbstractJump` types can then be passed within a single `JumpSet` or as
subsequent sequential arguments. 

### Fields

$(FIELDS)

## Keyword Arguments
- `rng`, the random number generator to use. On 1.7 and up defaults to Julia's
  builtin generator, below 1.7 uses RandomNumbers.jl's
  `Xorshifts.Xoroshiro128Star(rand(UInt64))`.
- `save_positions=(true,true)`, specifies whether to save the system's state
  (before,after) the jump occurs. 
- `spatial_system`, for spatial problems the underlying spatial structure.
- `hopping_constants`, for spatial problems the spatial transition rate
  coefficients.

Please see the [tutorial
page](https://diffeq.sciml.ai/stable/tutorials/discrete_stochastic_example/) in
the DifferentialEquations.jl [docs](https://diffeq.sciml.ai/stable/) for usage
examples and commonly asked questions.
"""
mutable struct JumpProblem{iip,P,A,C,J<:Union{Nothing,AbstractJumpAggregator},J2,J3,J4} <: DiffEqBase.AbstractJumpProblem{P,J}
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
end
function JumpProblem(p::P,a::A,dj::J,jc::C,vj::J2,rj::J3,mj::J4) where {P,A,J,C,J2,J3,J4}
    iip = isinplace_jump(p,rj)
    JumpProblem{iip,P,A,C,J,J2,J3,J4}(p,a,dj,jc,vj,rj,mj)
end

# for remaking
Base.@pure remaker_of(prob::T) where {T <: JumpProblem} = DiffEqBase.parameterless_type(T)
function DiffEqBase.remake(thing::JumpProblem; kwargs...)
  T = remaker_of(thing)

  errmesg = """
  JumpProblems can currently only be remade with new u0, p, tspan or prob fields. To change other fields create a new JumpProblem. Feel free to open an issue on DiffEqJump to discuss further.
  """
  !issubset(keys(kwargs),(:u0,:p,:tspan,:prob)) && error(errmesg)

  if :prob ∉ keys(kwargs)
    dprob = DiffEqBase.remake(thing.prob; kwargs...)

    # if the parameters were changed we must remake the MassActionJump too
    if (:p ∈ keys(kwargs)) && using_params(thing.massaction_jump)
      update_parameters!(thing.massaction_jump, dprob.p; kwargs...)
    end 
  else
    any(k -> k in keys(kwargs), (:u0,:p,:tspan)) && error("If remaking a JumpProblem you can not pass both prob and any of u0, p, or tspan.")
    dprob = kwargs[:prob]

    # we can't know if p was changed, so we must remake the MassActionJump
    if using_params(thing.massaction_jump)
      update_parameters!(thing.massaction_jump, dprob.p; kwargs...)
    end 
  end

  T(dprob, thing.aggregator, thing.discrete_jump_aggregation, thing.jump_callback,
     thing.variable_jumps, thing.regular_jump, thing.massaction_jump)
end

DiffEqBase.isinplace(::JumpProblem{iip}) where {iip} = iip
JumpProblem(prob::JumpProblem) = prob

JumpProblem(prob,jumps::ConstantRateJump;kwargs...) = JumpProblem(prob,JumpSet(jumps);kwargs...)
JumpProblem(prob,jumps::VariableRateJump;kwargs...) = JumpProblem(prob,JumpSet(jumps);kwargs...)
JumpProblem(prob,jumps::RegularJump;kwargs...) = JumpProblem(prob,JumpSet(jumps);kwargs...)
JumpProblem(prob,jumps::MassActionJump;kwargs...) = JumpProblem(prob,JumpSet(jumps);kwargs...)
JumpProblem(prob,jumps::AbstractJump...;kwargs...) = JumpProblem(prob,JumpSet(jumps...);kwargs...)

JumpProblem(prob,aggregator::AbstractAggregatorAlgorithm,jumps::ConstantRateJump;kwargs...) = JumpProblem(prob,aggregator,JumpSet(jumps);kwargs...)
JumpProblem(prob,aggregator::AbstractAggregatorAlgorithm,jumps::VariableRateJump;kwargs...) = JumpProblem(prob,aggregator,JumpSet(jumps);kwargs...)
JumpProblem(prob,aggregator::AbstractAggregatorAlgorithm,jumps::RegularJump;kwargs...) = JumpProblem(prob,aggregator,JumpSet(jumps);kwargs...)
JumpProblem(prob,aggregator::AbstractAggregatorAlgorithm,jumps::AbstractMassActionJump;kwargs...) = JumpProblem(prob,aggregator,JumpSet(jumps);kwargs...)
JumpProblem(prob,aggregator::AbstractAggregatorAlgorithm,jumps::AbstractJump...;kwargs...) = JumpProblem(prob,aggregator,JumpSet(jumps...);kwargs...)
JumpProblem(prob,jumps::JumpSet;kwargs...) = JumpProblem(prob,NullAggregator(),jumps;kwargs...)

function JumpProblem(prob, aggregator::AbstractAggregatorAlgorithm, jumps::JumpSet;
                     save_positions = typeof(prob) <: DiffEqBase.AbstractDiscreteProblem ? (false,true) : (true,true),
                     rng = DEFAULT_RNG, scale_rates = true, useiszero = true, spatial_system=nothing, hopping_constants=nothing, kwargs...)

  # initialize the MassActionJump rate constants with the user parameters
  if using_params(jumps.massaction_jump) 
    rates = jumps.massaction_jump.param_mapper(prob.p)
    maj = MassActionJump(rates, jumps.massaction_jump.reactant_stoch, jumps.massaction_jump.net_stoch, 
                         jumps.massaction_jump.param_mapper; scale_rates=scale_rates, useiszero=useiszero, 
                         nocopy=true)
  else
    maj = jumps.massaction_jump
  end

  ## Spatial jumps handling
  if spatial_system !== nothing && hopping_constants !== nothing && !is_spatial(aggregator) # check if need to flatten
    prob, maj = flatten(maj, prob, spatial_system, hopping_constants; kwargs...)
  end
  ## Constant Rate Handling
  t,end_time,u = prob.tspan[1],prob.tspan[2],prob.u0
  if (typeof(jumps.constant_jumps) <: Tuple{}) && (maj === nothing) && !is_spatial(aggregator) # check if there are no jumps
    disc = nothing
    constant_jump_callback = CallbackSet()
  else
    disc = aggregate(aggregator,u,prob.p,t,end_time,jumps.constant_jumps,maj,save_positions,rng; spatial_system = spatial_system, hopping_constants = hopping_constants, kwargs...)
    constant_jump_callback = DiscreteCallback(disc)
  end

  iip = isinplace_jump(prob, jumps.regular_jump)

  ## Variable Rate Handling
  if typeof(jumps.variable_jumps) <: Tuple{}
    new_prob = prob
    variable_jump_callback = CallbackSet()
  else
    new_prob = extend_problem(prob,jumps)
    variable_jump_callback = build_variable_callback(CallbackSet(),0,jumps.variable_jumps...)
  end
  callbacks = CallbackSet(constant_jump_callback,variable_jump_callback)

  JumpProblem{iip,typeof(new_prob),typeof(aggregator),typeof(callbacks),
              typeof(disc),typeof(jumps.variable_jumps),
              typeof(jumps.regular_jump),typeof(maj)}(
                        new_prob,aggregator,disc,
                        callbacks,
                        jumps.variable_jumps,
                        jumps.regular_jump, maj)
end

function extend_problem(prob::DiffEqBase.AbstractDiscreteProblem,jumps)
  error("VariableRateJumps require a continuous problem, like an ODE/SDE/DDE/DAE problem.")
end

function extend_problem(prob::DiffEqBase.AbstractODEProblem,jumps)
  function jump_f(du::ExtendedJumpArray,u::ExtendedJumpArray,p,t)
    prob.f(du.u,u.u,p,t)
    update_jumps!(du,u,p,t,length(u.u),jumps.variable_jumps...)
  end
  T = eltype(prob.u0)
  u0 = ExtendedJumpArray(prob.u0,[T(-randexp()) for i in 1:length(jumps.variable_jumps)])
  remake(prob,f=ODEFunction{true}(jump_f),u0=u0)
end

function extend_problem(prob::DiffEqBase.AbstractSDEProblem,jumps)
  function jump_f(du,u,p,t)
    prob.f(du.u,u.u,p,t)
    update_jumps!(du,u,p,t,length(u.u),jumps.variable_jumps...)
  end

  if prob.noise_rate_prototype === nothing
    jump_g = function (du,u,p,t)
      prob.g(du.u,u.u,p,t)
    end
  else
    jump_g = function (du,u,p,t)
      prob.g(du,u.u,p,t)
    end
  end

  T = eltype(prob.u0)
  u0 = ExtendedJumpArray(prob.u0,[T(-randexp()) for i in 1:length(jumps.variable_jumps)])
  remake(prob,f=SDEFunction{true}(jump_f,jump_g),g=jump_g,u0=u0)
end

function extend_problem(prob::DiffEqBase.AbstractDDEProblem,jumps)
  jump_f = function (du,u,h,p,t)
    prob.f(du.u,u.u,h,p,t)
    update_jumps!(du,u,p,t,length(u.u),jumps.variable_jumps...)
  end
  T = eltype(prob.u0)
  u0 = ExtendedJumpArray(prob.u0,[T(-randexp()) for i in 1:length(jumps.variable_jumps)])
  ramake(prob,f=DDEFunction{true}(jump_f),u0=u0)
end

# Not sure if the DAE one is correct: Should be a residual of sorts
function extend_problem(prob::DiffEqBase.AbstractDAEProblem,jumps)
  jump_f = function (out,du,u,p,t)
    prob.f(out.u,du.u,u.u,t)
    update_jumps!(du,u,t,length(u.u),jumps.variable_jumps...)
  end
  T = eltype(prob.u0)
  u0 = ExtendedJumpArray(prob.u0,[T(-randexp()) for i in 1:length(jumps.variable_jumps)])
  remake(prob,f=DAEFunction{true}(jump_f),u0=u0)
end

function build_variable_callback(cb,idx,jump,jumps...)
  idx += 1
  condition = function (u,t,integrator)
    u.jump_u[idx]
  end
  affect! = function (integrator)
    jump.affect!(integrator)
    integrator.u.jump_u[idx] = -randexp()
  end
  new_cb = ContinuousCallback(condition,affect!;
                      idxs = jump.idxs,
                      rootfind = jump.rootfind,
                      interp_points = jump.interp_points,
                      save_positions = jump.save_positions,
                      abstol = jump.abstol,
                      reltol = jump.reltol)
  build_variable_callback(CallbackSet(cb,new_cb),idx,jumps...)
end

function build_variable_callback(cb,idx,jump)
  idx += 1
  condition = function (u,t,integrator)
    u.jump_u[idx]
  end
  affect! = function (integrator)
    jump.affect!(integrator)
    integrator.u.jump_u[idx] = -randexp()
  end
  new_cb = ContinuousCallback(condition,affect!;
                      idxs = jump.idxs,
                      rootfind = jump.rootfind,
                      interp_points = jump.interp_points,
                      save_positions = jump.save_positions,
                      abstol = jump.abstol,
                      reltol = jump.reltol)
  CallbackSet(cb,new_cb)
end

aggregator(jp::JumpProblem{P,A,C,J,J2}) where {P,A,C,J,J2} = A

@inline function extend_tstops!(tstops,jp::JumpProblem{P,A,C,J,J2}) where {P,A,C,J,J2}
  !(typeof(jp.jump_callback.discrete_callbacks) <: Tuple{}) && push!(tstops,jp.jump_callback.discrete_callbacks[1].condition.next_jump_time)
end

@inline function update_jumps!(du,u,p,t,idx,jump)
  idx += 1
  du[idx] = jump.rate(u,p,t)
end

@inline function update_jumps!(du,u,p,t,idx,jump,jumps...)
  idx += 1
  du[idx] = jump.rate(u,p,t)
  update_jumps!(du,u,p,t,idx,jumps...)
end


### Displays
num_constant_rate_jumps(aggregator::AbstractSSAJumpAggregator) = length(aggregator.rates)

Base.summary(io::IO, prob::JumpProblem) = string(DiffEqBase.parameterless_type(prob)," with problem ",DiffEqBase.parameterless_type(prob.prob)," and aggregator ",typeof(prob.aggregator))
function Base.show(io::IO, mime::MIME"text/plain", A::JumpProblem)
  println(io,summary(A))
  println(io,"Number of constant rate jumps: ",A.discrete_jump_aggregation === nothing ? 0 : num_constant_rate_jumps(A.discrete_jump_aggregation))
  println(io,"Number of variable rate jumps: ",length(A.variable_jumps))
  if A.regular_jump !== nothing
    println(io,"Have a regular jump")
  end
  if (A.massaction_jump !== nothing) && (get_num_majumps(A.massaction_jump) > 0)
    println(io,"Have a mass action jump")
  end
end

TreeViews.hastreeview(x::JumpProblem) = true
