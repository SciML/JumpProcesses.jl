function reset_aggregated_jumps!(integrator,uprev = nothing)
     reset_aggregated_jumps!(integrator,uprev,integrator.opts.callback)
     nothing
end

function reset_aggregated_jumps!(integrator,uprev,callback::CallbackSet)
    if !isempty(callback.discrete_callbacks)
        reset_aggregated_jumps!(integrator,uprev,callback.discrete_callbacks...)
    end
    nothing
end

function reset_aggregated_jumps!(integrator,uprev,cb::DiscreteCallback,cbs...)
    reset_aggregated_jumps!(integrator,uprev,cbs...)
    nothing
end

reset_aggregated_jumps!(integrator,uprev,cb::DiscreteCallback) = nothing

function reset_aggregated_jumps!(integrator,uprev,cb::AbstractSSAJumpAggregator,cbs...)
    cb(cb,integrator.u,t,integrator) # This overload is the aggregated cb's init
    reset_aggregated_jumps!(integrator,uprev,cbs...)
    nothing
end

function reset_aggregated_jumps!(integrator,uprev,cb::AbstractSSAJumpAggregator)
    reset_aggregated_jumps!(integrator,uprev,cbs...)
    nothing
end
