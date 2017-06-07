type JumpProblem{P,A,C,J<:Union{Void,AbstractJumpAggregator},J2} <: AbstractJumpProblem{P,J}
  prob::P
  aggregator::A
  discrete_jump_aggregation::J
  jump_callback::C
  variable_jumps::J2
end

JumpProblem(prob,aggregator::AbstractAggregatorAlgorithm,jumps::ConstantRateJump;kwargs...) = JumpProblem(prob,aggregator,JumpSet(jumps);kwargs...)
JumpProblem(prob,aggregator::AbstractAggregatorAlgorithm,jumps::VariableRateJump;kwargs...) = JumpProblem(prob,aggregator,JumpSet(jumps);kwargs...)
JumpProblem(prob,aggregator::AbstractAggregatorAlgorithm,jumps...;kwargs...) = JumpProblem(prob,aggregator,JumpSet(jumps...);kwargs...)

function JumpProblem(prob,aggregator::Direct,jumps::JumpSet;
                     save_positions = typeof(prob) <: AbstractDiscreteProblem ? (false,true) : (true,true))

  ## Constant Rate Handling
  t,end_time,u = prob.tspan[1],prob.tspan[2],prob.u0
  if typeof(jumps.constant_jumps) <: Tuple{}
    disc = nothing
    constant_jump_callback = CallbackSet()
  else
    disc = aggregate(aggregator,t,u,end_time,jumps.constant_jumps,save_positions)
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
              typeof(disc),typeof(jumps.variable_jumps)}(
                        new_prob,aggregator,disc,
                        callbacks,
                        jumps.variable_jumps)
end

function extend_problem(prob::AbstractODEProblem,jumps)
  jump_f = function (t,u,du)
    prob.f(t,u.u,@view du[1:length(u.u)])
    update_jumps!(du,t,u,length(u.u),jumps.variable_jumps...)
  end
  u0 = ExtendedJumpArray(prob.u0,[-randexp() for i in 1:length(jumps.variable_jumps)])
  ODEProblem(jump_f,u0,prob.tspan)
end

function extend_problem(prob::AbstractSDEProblem,jumps)
  jump_f = function (t,u,du)
    prob.f(t,u.u,@view du[1:length(u.u)])
    update_jumps!(du,t,u,length(u.u),jumps.variable_jumps...)
  end
  u0 = ExtendedJumpArray(prob.u0,[-randexp() for i in 1:length(jumps.variable_jumps)])
  SDEProblem(jump_f,prob.g,u0,prob.tspan)
end

function extend_problem(prob::AbstractDDEProblem,jumps)
  jump_f = function (t,u,h,du)
    prob.f(t,u.u,h,@view du[1:length(u.u)])
    update_jumps!(du,t,u,length(u.u),jumps.variable_jumps...)
  end
  u0 = ExtendedJumpArray(prob.u0,[-randexp() for i in 1:length(jumps.variable_jumps)])
  DDEProblem(jump_f,prob.h,u0,prob.lags,prob.tspan)
end

# Not sure if the DAE one is correct: Should be a residual of sorts
function extend_problem(prob::AbstractDAEProblem,jumps)
  jump_f = function (t,u,du,out)
    prob.f(t,u.u,du,@view out[1:length(u.u)])
    update_jumps!(du,t,u,length(u.u),jumps.variable_jumps...)
  end
  u0 = ExtendedJumpArray(prob.u0,[-randexp() for i in 1:length(jumps.variable_jumps)])
  DAEProblem(jump_f,prob.h,u0,prob.lags,prob.tspan)
end

function build_variable_callback(cb,idx,jump,jumps...)
  idx += 1
  condition = function (t,u,integrator)
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
  condition = function (t,u,integrator)
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

aggregator{P,A,C,J,J2}(jp::JumpProblem{P,A,C,J,J2}) = A

@inline function extend_tstops!{P,A,C,J,J2}(tstops,jp::JumpProblem{P,A,C,J,J2})
  !(typeof(jp.jump_callback.discrete_callbacks) <: Tuple{}) && push!(tstops,jp.jump_callback.discrete_callbacks[1].condition.next_jump)
end

@inline function update_jumps!(du,t,u,idx,jump)
  idx += 1
  du[idx] = jump.rate(t,u)
end

@inline function update_jumps!(du,t,u,idx,jump,jumps...)
  idx += 1
  du[idx] = jump.rate(t,u)
  update_jumps!(du,t,u,idx,jumps...)
end


### Displays

Base.summary(prob::JumpProblem) = string(DiffEqBase.parameterless_type(prob)," with problem ",DiffEqBase.parameterless_type(prob.prob)," and aggregator ",typeof(prob.aggregator))
function Base.show(io::IO, A::JumpProblem)
  println(io,summary(A))
  println(io,"Number of constant rate jumps: ",length(A.discrete_jump_aggregation.rates))
  println(io,"Number of variable rate jumps: ",length(A.variable_jumps))
end
function Base.display(io::IO, A::JumpProblem)
  println(io,summary(A))
  println(io,"Number of constant rate jumps: ",length(A.discrete_jump_aggregation.rates))
  println(io,"Number of variable rate jumps: ",length(A.variable_jumps))
end
function Base.print(io::IO,A::JumpProblem)
  show(io,A)
end
function Base.println(io::IO,A::JumpProblem)
  show(io,A)
end
Base.print(A::JumpProblem) = print(STDOUT,A)
Base.println(A::JumpProblem) = println(STDOUT,A)
