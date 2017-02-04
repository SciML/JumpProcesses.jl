module JumpDiffEq

using DiffEqBase

import DiffEqBase: DiscreteCallback, init, solve, solve!

abstract AbstractJump
abstract AbstractJumpProblem{P,J} <: DEProblem

include("jumps.jl")
include("problem.jl")
include("callbacks.jl")
include("solve.jl")

export ConstantRateJump, VariableRateJump, JumpSet, JumpProblem

export init, solve, solve!

end # module
