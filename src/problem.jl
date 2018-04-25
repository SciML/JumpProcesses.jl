mutable struct JumpProblem{P,A,C,J<:Union{Void,AbstractJumpAggregator},J2,J3,J4} <: AbstractJumpProblem{P,J}
  prob::P
  aggregator::A
  discrete_jump_aggregation::J
  jump_callback::C
  variable_jumps::J2
  regular_jump::J3
  massaction_jump::J4
end

JumpProblem(prob,jumps::ConstantRateJump;kwargs...) = JumpProblem(prob,JumpSet(jumps);kwargs...)
JumpProblem(prob,jumps::VariableRateJump;kwargs...) = JumpProblem(prob,JumpSet(jumps);kwargs...)
JumpProblem(prob,jumps::RegularJump;kwargs...) = JumpProblem(prob,JumpSet(jumps);kwargs...)
JumpProblem(prob,jumps::MassActionJump;kwargs...) = JumpProblem(prob,JumpSet(jumps);kwargs...)
JumpProblem(prob,jumps::AbstractJump...;kwargs...) = JumpProblem(prob,JumpSet(jumps...);kwargs...)

JumpProblem(prob,aggregator::AbstractAggregatorAlgorithm,jumps::ConstantRateJump;kwargs...) = JumpProblem(prob,aggregator,JumpSet(jumps);kwargs...)
JumpProblem(prob,aggregator::AbstractAggregatorAlgorithm,jumps::VariableRateJump;kwargs...) = JumpProblem(prob,aggregator,JumpSet(jumps);kwargs...)
JumpProblem(prob,aggregator::AbstractAggregatorAlgorithm,jumps::RegularJump;kwargs...) = JumpProblem(prob,aggregator,JumpSet(jumps);kwargs...)
JumpProblem(prob,aggregator::AbstractAggregatorAlgorithm,jumps::MassActionJump;kwargs...) = JumpProblem(prob,aggregator,JumpSet(jumps);kwargs...)
JumpProblem(prob,aggregator::AbstractAggregatorAlgorithm,jumps::AbstractJump...;kwargs...) = JumpProblem(prob,aggregator,JumpSet(jumps...);kwargs...)
JumpProblem(prob,jumps::JumpSet;kwargs...) = JumpProblem(prob,NullAggregator(),jumps;kwargs...)

function JumpProblem(prob, aggregator::AbstractAggregatorAlgorithm, jumps::JumpSet;
                     save_positions = typeof(prob) <: AbstractDiscreteProblem ? (false,true) : (true,true),
                     rng = Xorshifts.Xoroshiro128Star(rand(UInt64)), kwargs...)

  ## Constant Rate Handling
  t,end_time,u = prob.tspan[1],prob.tspan[2],prob.u0
  if (typeof(jumps.constant_jumps) <: Tuple{}) && (jumps.massaction_jump == nothing)
    disc = nothing
    constant_jump_callback = CallbackSet()
  else    
    disc = aggregate(aggregator,u,prob.p,t,end_time,jumps.constant_jumps,jumps.massaction_jump,save_positions,rng;kwargs...)    
    constant_jump_callback = DiscreteCallback(disc)
  end

  ## Variable Rate Handling
  if typeof(jumps.variable_jumps) <: Tuple{}
    new_prob = prob
    variable_jump_callback = CallbackSet()
  else
    new_prob = extend_problem(prob,jumps)
    variable_jump_callback = build_variable_callback(CallbackSet(),0,jumps.variable_jumps...)
  end
  callbacks = CallbackSet(constant_jump_callback,variable_jump_callback)
  JumpProblem{typeof(new_prob),typeof(aggregator),typeof(callbacks),
              typeof(disc),typeof(jumps.variable_jumps),
              typeof(jumps.regular_jump),typeof(jumps.massaction_jump)}(
                        new_prob,aggregator,disc,
                        callbacks,
                        jumps.variable_jumps,
                        jumps.regular_jump, jumps.massaction_jump)
end

function extend_problem(prob::AbstractODEProblem,jumps)
  function jump_f(du,u,p,t)
    prob.f(@view(du[1:length(u.u)]),u.u,p,t)
    update_jumps!(du,u,p,t,length(u.u),jumps.variable_jumps...)
  end
  u0 = ExtendedJumpArray(prob.u0,[-randexp() for i in 1:length(jumps.variable_jumps)])
  ODEProblem(jump_f,u0,prob.tspan,prob.p)
end

function extend_problem(prob::AbstractSDEProblem,jumps)
  function jump_f(du,u,p,t)
    prob.f(@view(du[1:length(u.u)]),u.u,p,t)
    update_jumps!(du,u,p,t,length(u.u),jumps.variable_jumps...)
  end
  u0 = ExtendedJumpArray(prob.u0,[-randexp() for i in 1:length(jumps.variable_jumps)])
  SDEProblem(jump_f,prob.g,u0,prob.tspan,prob.p)
end

function extend_problem(prob::AbstractDDEProblem,jumps)
  jump_f = function (du,u,h,p,t)
    prob.f(@view(du[1:length(u.u)]),u.u,h,p,t)
    update_jumps!(du,u,p,t,length(u.u),jumps.variable_jumps...)
  end
  u0 = ExtendedJumpArray(prob.u0,[-randexp() for i in 1:length(jumps.variable_jumps)])
  DDEProblem(jump_f,prob.h,u0,prob.lags,prob.tspan,prob.p)
end

# Not sure if the DAE one is correct: Should be a residual of sorts
function extend_problem(prob::AbstractDAEProblem,jumps)
  jump_f = function (out,du,u,p,t)
    prob.f(@view(out[1:length(u.u)]),du.u,u.u,t)
    update_jumps!(du,u,t,length(u.u),jumps.variable_jumps...)
  end
  u0 = ExtendedJumpArray(prob.u0,[-randexp() for i in 1:length(jumps.variable_jumps)])
  DAEProblem(jump_f,prob.h,u0,prob.lags,prob.tspan)
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

Base.summary(prob::JumpProblem) = string(DiffEqBase.parameterless_type(prob)," with problem ",DiffEqBase.parameterless_type(prob.prob)," and aggregator ",typeof(prob.aggregator))
function Base.show(io::IO, A::JumpProblem)
  println(io,summary(A))
  println(io,"Number of constant rate jumps: ",A.discrete_jump_aggregation == nothing ? 0 : length(A.discrete_jump_aggregation.rates))
  println(io,"Number of variable rate jumps: ",length(A.variable_jumps))
  if A.regular_jump != nothing
    println(io,"Have a regular jump")
  end
  if (A.massaction_jump != nothing) && (get_num_majumps(A.massaction_jump) > 0)
    println(io,"Have a mass action jump")
  end
end
