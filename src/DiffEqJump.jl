module DiffEqJump

using DiffEqBase

import DiffEqBase: DiscreteCallback, init, solve, solve!, plot_indices
import Base: size, getindex, setindex!, length, similar, linearindexing, indices,
       display, show

import RecursiveArrayTools: recursivecopy!

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
include("extended_jump_array.jl")
include("coupling.jl")

export AbstractJump, AbstractAggregatorAlgorithm, AbstractJumpAggregator, AbstractJumpProblem

export ConstantRateJump, VariableRateJump, JumpSet, CompoundConstantRateJump

export JumpProblem

export SplitCoupledJumpProblem

export Direct

export init, solve, solve!

export ExtendedJumpArray

end # module
