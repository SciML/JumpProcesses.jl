function reset_aggregated_jumps!(integrator,uprev = nothing)
     reset_aggregated_jumps!(integrator,uprev,integrator.cb)
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
        cb.condition(cb,integrator.u,integrator.t,integrator)
    end
    reset_aggregated_jumps!(integrator,uprev,cbs...)
    nothing
end

function reset_aggregated_jumps!(integrator,uprev,cb::DiscreteCallback)
    if typeof(cb.condition) <: AbstractSSAJumpAggregator
        cb.condition(cb,integrator.u,integrator.t,integrator)
    end
    nothing
end
