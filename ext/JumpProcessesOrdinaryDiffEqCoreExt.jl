module JumpProcessesOrdinaryDiffEqCoreExt

using JumpProcesses
import DiffEqBase
import OrdinaryDiffEqCore: OrdinaryDiffEqAlgorithm, DAEAlgorithm,
    StochasticDiffEqAlgorithm, StochasticDiffEqRODEAlgorithm

# Ambiguity fix: OrdinaryDiffEqCore defines
#   __init(::Union{..., AbstractJumpProblem}, ::Union{OrdinaryDiffEqAlgorithm,
#          DAEAlgorithm, StochasticDiffEqAlgorithm, StochasticDiffEqRODEAlgorithm})
# which is ambiguous with JumpProcesses'
#   __init(::AbstractJumpProblem{P}, ::DEAlgorithm)
#
# This method resolves the ambiguity by being more specific in the problem type
# (AbstractJumpProblem vs Union{..., AbstractJumpProblem, ...}) while matching
# the exact algorithm union from OrdinaryDiffEqCore.
function DiffEqBase.__init(
        _jump_prob::DiffEqBase.AbstractJumpProblem{P},
        alg::Union{OrdinaryDiffEqAlgorithm, DAEAlgorithm,
            StochasticDiffEqAlgorithm, StochasticDiffEqRODEAlgorithm};
        merge_callbacks = true, kwargs...) where {P}
    kwargs = DiffEqBase.merge_problem_kwargs(_jump_prob; merge_callbacks, kwargs...)
    JumpProcesses.__jump_init(_jump_prob, alg; kwargs...)
end

end
