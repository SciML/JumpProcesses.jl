"""
    reset_aggregated_jumps!(integrator,uprev = nothing)

Reset the state of jump processes and associated solvers following a change
in parameters or such.
"""

function reset_aggregated_jumps!(integrator,uprev = nothing; update_jump_params=true)
     reset_aggregated_jumps!(integrator,uprev,integrator.opts.callback,
                             update_jump_params=update_jump_params)
     nothing
end

function reset_aggregated_jumps!(integrator,uprev,callback::Nothing; update_jump_params=true)
     nothing
end


function reset_aggregated_jumps!(integrator,uprev,callback::CallbackSet; update_jump_params=true)
    if !isempty(callback.discrete_callbacks)
        reset_aggregated_jumps!(integrator,uprev,callback.discrete_callbacks..., 
                                update_jump_params=update_jump_params)
    end
    nothing
end

function reset_aggregated_jumps!(integrator,uprev,cb::DiscreteCallback,cbs...; update_jump_params=true)
    if typeof(cb.condition) <: AbstractSSAJumpAggregator
        maj = cb.condition.ma_jumps
        update_jump_params && using_params(maj) && update_parameters!(cb.condition.ma_jumps,integrator.p)
        cb.condition(cb,integrator.u,integrator.t,integrator)
    end
    reset_aggregated_jumps!(integrator,uprev,cbs...; 
                            update_jump_params=update_jump_params)
    nothing
end

function reset_aggregated_jumps!(integrator,uprev,cb::DiscreteCallback; update_jump_params=true)
    if typeof(cb.condition) <: AbstractSSAJumpAggregator
        maj = cb.condition.ma_jumps
        update_jump_params && using_params(maj) && update_parameters!(cb.condition.ma_jumps,integrator.p)
        cb.condition(cb,integrator.u,integrator.t,integrator)
    end
    nothing
end
