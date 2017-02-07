module JumpDiffEq

using DiffEqBase

import DiffEqBase: DiscreteCallback, init, solve, solve!
import Base: size, getindex, setindex!

abstract AbstractJump
abstract AbstractAggregatorAlgorithm
abstract AbstractJumpAggregator
abstract AbstractJumpProblem{P,J} <: DEProblem

include("aggregators/aggregators.jl")
include("aggregators/direct.jl")
include("jumps.jl")
include("problem.jl")
include("callbacks.jl")
include("solve.jl")

export AbstractJump, AbstractAggregatorAlgorithm, AbstractJumpAggregator, AbstractJumpProblem

export ConstantRateJump, VariableRateJump, JumpSet, CompoundConstantRateJump

export JumpProblem

export Direct

export init, solve, solve!

end # module
