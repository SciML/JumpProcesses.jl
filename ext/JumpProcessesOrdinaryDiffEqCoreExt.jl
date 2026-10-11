module JumpProcessesOrdinaryDiffEqCoreExt

using JumpProcesses
import SciMLBase
import OrdinaryDiffEqCore: OrdinaryDiffEqAlgorithm, DAEAlgorithm,
                           StochasticDiffEqAlgorithm, StochasticDiffEqRODEAlgorithm

# `init_call` has already merged the problem's stored keywords before `__init`; see the
# generic `__init` in src/solve.jl. `merge_callbacks` is discarded, not forwarded.
function _jump_init(_jump_prob, alg; merge_callbacks = true, kwargs...)
    JumpProcesses.__jump_init(_jump_prob, alg; kwargs...)
end

function SciMLBase.__init(
        _jump_prob::JumpProcesses.JumpProblem{IIP, P},
        alg::Union{OrdinaryDiffEqAlgorithm, DAEAlgorithm,
            StochasticDiffEqAlgorithm, StochasticDiffEqRODEAlgorithm};
        kwargs...) where {IIP, P}
    # Cover the full intersection with OrdinaryDiffEqCore's initializer. SDE/RODE
    # initialization belongs to StochasticDiffEqCore, whose per-algorithm methods are
    # more specific.
    _jump_init(_jump_prob, alg; kwargs...)
end

end
