type JumpProblem{P,A,C,J,J2} <: AbstractJumpProblem{P}
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
    constant_jump_callback = nothing
  else
    disc = aggregate(aggregator,t,u,end_time,jumps.constant_jumps,save_positions)
    constant_jump_callback = DiscreteCallback(disc)
  end

  ## Variable Rate Handling
  if typeof(jumps.variable_jumps) <: Tuple{}
    new_prob = prob
  else
    jump_f = function (t,u,du)
      f(t,u,du)
      update_jumps!(du,t,u,length(u.u),variable_jumps...)
    end
    new_prob = extend_problem(prob,jump_f,jumps.variable_jumps)
  end

  JumpProblem{typeof(new_prob),typeof(aggregator),typeof(constant_jump_callback),
              typeof(disc),typeof(jumps.variable_jumps)}(
                        new_prob,aggregator,disc,
                        constant_jump_callback,
                        jumps.variable_jumps)
end

function extend_problem(prob::ODEProblem,jump_f,variable_jumps)
  ODEProblem(jump_f,prob.u0,prob.tspan)
end

aggregator{P,A,C,J,J2}(jp::JumpProblem{P,A,C,J,J2}) = A

@inline function extend_tstops!{P,A,C,J,J2}(tstops,jp::JumpProblem{P,A,C,J,J2})
  !(typeof(jp.jump_callback) <: Void) && push!(tstops,jp.jump_callback.condition.next_jump)
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
