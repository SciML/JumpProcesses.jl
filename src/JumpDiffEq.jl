module JumpDiffEq

using DiffEqBase

import DiffEqBase: DiscreteCallback, init, solve, solve!

abstract AbstractJump
abstract AbstractJumpProblem{P,J} <: DEProblem

include("jumps.jl")
include("gillespie.jl")
include("problem.jl")
include("callbacks.jl")
include("compound_constant.jl")
include("solve.jl")

export ConstantRateJump, VariableRateJump, JumpSet, CompoundConstantRateJump

export JumpProblem, GillespieProblem

export Reaction, ReactionSet, build_jumps_from_reaction

export init, solve, solve!

end # module
