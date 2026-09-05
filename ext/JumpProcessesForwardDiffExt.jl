module JumpProcessesForwardDiffExt

using JumpProcesses: ExtendedJumpArray
import DiffEqBase
import ForwardDiff

@inline function DiffEqBase.ODE_DEFAULT_NORM(
        u::ExtendedJumpArray{<:ForwardDiff.Dual}, t::ForwardDiff.Dual
    )
    return invoke(DiffEqBase.ODE_DEFAULT_NORM, Tuple{ExtendedJumpArray, Any}, u, t)
end

end
