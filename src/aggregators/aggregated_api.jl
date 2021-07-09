"""
    reset_aggregated_jumps!(integrator,uprev = nothing)

Reset the state of jump processes and associated solvers following a change
in parameters or such.
"""

function reset_aggregated_jumps!(integrator,uprev = nothing)
     reset_aggregated_jumps!(integrator,uprev,integrator.opts.callback)
     nothing
end

function reset_aggregated_jumps!(integrator,uprev,callback::Nothing)
     nothing
end


function reset_aggregated_jumps!(integrator,uprev,callback::CallbackSet)
    if !isempty(callback.discrete_callbacks)
        reset_aggregated_jumps!(integrator,uprev,callback.discrete_callbacks...)
    end
    nothing
end

function reset_aggregated_jumps!(integrator,uprev,cb::DiscreteCallback,cbs...)
    if typeof(cb.condition) <: AbstractSSAJumpAggregator
        maj = cb.condition.ma_jumps
        using_params(maj) && update_parameters!(cb.condition.ma_jumps,integrator.p)
        cb.condition(cb,integrator.u,integrator.t,integrator)
    end
    reset_aggregated_jumps!(integrator,uprev,cbs...)
    nothing
end

function reset_aggregated_jumps!(integrator,uprev,cb::DiscreteCallback)
    if typeof(cb.condition) <: AbstractSSAJumpAggregator
        maj = cb.condition.ma_jumps
        using_params(maj) && update_parameters!(cb.condition.ma_jumps,integrator.p)
        cb.condition(cb,integrator.u,integrator.t,integrator)
    end
    nothing
end
