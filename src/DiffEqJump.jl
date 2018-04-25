__precompile__()

module DiffEqJump

using DiffEqBase, Compat, Requires, Distributions, RandomNumbers, FunctionWrappers

import DiffEqBase: DiscreteCallback, init, solve, solve!, plot_indices
import Base: size, getindex, setindex!, length, similar, indices, show

import RecursiveArrayTools: recursivecopy!

@compat abstract type AbstractJump end
@compat abstract type AbstractAggregatorAlgorithm end
@compat abstract type AbstractJumpAggregator end
@compat abstract type AbstractJumpProblem{P,J} <: DEProblem end

include("jumps.jl")
include("massaction_rates.jl")
include("aggregators/aggregators.jl")
include("aggregators/ssajump.jl")
include("aggregators/direct.jl")
include("aggregators/frm.jl")
include("problem.jl")
include("callbacks.jl")
include("solve.jl")
include("extended_jump_array.jl")
include("coupled_array.jl")
include("coupling.jl")
include("SSA_stepper.jl")
include("simple_regular_solve.jl")
include("juno_rendering.jl")

export AbstractJump, AbstractAggregatorAlgorithm, AbstractJumpAggregator, 
       AbstractSSAJumpAggregator, AbstractJumpProblem

export ConstantRateJump, VariableRateJump, RegularJump, MassActionJump, 
       JumpSet, CompoundConstantRateJump

export JumpProblem

export SplitCoupledJumpProblem

export Direct, DirectFW, FRM, FRMFW

export init, solve, solve!

export ExtendedJumpArray

end # module
