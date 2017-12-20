__precompile__()

module DiffEqJump

using DiffEqBase, Compat, Requires

import DiffEqBase: DiscreteCallback, init, solve, solve!, plot_indices
import Base: size, getindex, setindex!, length, similar, indices, show

import RecursiveArrayTools: recursivecopy!

@compat abstract type AbstractJump end
@compat abstract type AbstractAggregatorAlgorithm end
@compat abstract type AbstractJumpAggregator end
@compat abstract type AbstractJumpProblem{P,J} <: DEProblem end

include("aggregators/aggregators.jl")
include("aggregators/direct.jl")
include("jumps.jl")
include("problem.jl")
include("callbacks.jl")
include("solve.jl")
include("extended_jump_array.jl")
include("coupled_array.jl")
include("coupling.jl")
include("juno_rendering.jl")

export AbstractJump, AbstractAggregatorAlgorithm, AbstractJumpAggregator, AbstractJumpProblem

export ConstantRateJump, VariableRateJump, JumpSet, CompoundConstantRateJump

export JumpProblem

export SplitCoupledJumpProblem

export Direct

export init, solve, solve!

export ExtendedJumpArray

end # module
