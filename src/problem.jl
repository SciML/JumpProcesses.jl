type JumpProblem{P,J} <: AbstractJumpProblem{P}
  prob::P
  jumps::J
end

JumpProblem(prob,jumps::ConstantRateJump) = JumpProblem(prob,JumpSet(jumps))
JumpProblem(prob,jumps::VariableRateJump) = JumpProblem(prob,JumpSet(jumps))
