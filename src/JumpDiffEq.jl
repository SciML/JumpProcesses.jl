module JumpDiffEq

using DiffEqBase

import DiffEqBase: DiscreteCallback, init, solve, solve!

abstract AbstractJump
abstract AbstractJumpProblem{P,J} <: DEProblem

include("jumps.jl")
include("problem.jl")
include("callbacks.jl")
include("compound_constant.jl")
include("solve.jl")

export ConstantRateJump, VariableRateJump, JumpSet, JumpProblem, CompoundConstantRateJump

export init, solve, solve!

end # module
