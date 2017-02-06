type JumpProblem{P,J<:JumpSet} <: AbstractJumpProblem{P}
  prob::P
  jumps::J
end

JumpProblem(prob,jumps::ConstantRateJump) = JumpProblem(prob,JumpSet(jumps))
JumpProblem(prob,jumps::VariableRateJump) = JumpProblem(prob,JumpSet(jumps))
JumpProblem(prob,jumps::JumpSet) = JumpProblem(prob,jumps)
JumpProblem(prob,jumps...) = JumpProblem(prob,JumpSet(jumps...))

GillespieProblem(prob,rs::Reaction...) = JumpProblem(prob,
                                         build_jumps_from_reaction(rs...))
