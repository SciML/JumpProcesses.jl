type JumpProblem{P,A,J,J2} <: AbstractJumpProblem{P}
  prob::P
  aggregator::A
  discrete_jump_aggregation::J
  variable_jumps::J2
end

JumpProblem(prob,aggregator::AbstractAggregatorAlgorithm,jumps::ConstantRateJump;kwargs...) = JumpProblem(prob,aggregator,JumpSet(jumps);kwargs...)
JumpProblem(prob,aggregator::AbstractAggregatorAlgorithm,jumps::VariableRateJump;kwargs...) = JumpProblem(prob,aggregator,JumpSet(jumps);kwargs...)
JumpProblem(prob,aggregator::AbstractAggregatorAlgorithm,jumps...;kwargs...) = JumpProblem(prob,aggregator,JumpSet(jumps...);kwargs...)

function JumpProblem(prob,aggregator::Direct,jumps::JumpSet;
                     save_positions = typeof(prob) <: AbstractDiscreteProblem ? (false,true) : (true,true))
  t,end_time,u = prob.tspan[1],prob.tspan[2],prob.u0
  disc = DirectJumpAggregation(t,u,end_time,jumps.constant_jumps,save_positions)
  JumpProblem{typeof(prob),typeof(aggregator),typeof(disc),typeof(jumps.variable_jumps)}(prob,aggregator,disc,jumps.variable_jumps)
end

aggregator{P,A,J,J2}(jp::JumpProblem{P,A,J,J2}) = A
