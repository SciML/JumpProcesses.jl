"""
    reset_aggregated_jumps!(integrator, uprev = nothing; update_jump_params=true)

Reset the state of jump processes and associated solvers following a change
in parameters or such.

Notes

  - `update_jump_params=true` will recalculate the rates stored within any
    MassActionJump that was built from the parameter vector. If the parameter
    vector is unchanged, this can safely be set to false to improve performance.
"""
function reset_aggregated_jumps!(
        integrator, uprev = nothing; update_jump_params = true,
        kwargs...
    )
    reset_aggregated_jumps!(
        integrator, uprev, integrator.opts.callback,
        update_jump_params = update_jump_params, kwargs...
    )
    return nothing
end

function reset_aggregated_jumps!(
        integrator, uprev, callback::Nothing;
        update_jump_params = true, kwargs...
    )
    return nothing
end

function reset_aggregated_jumps!(
        integrator, uprev, callback::CallbackSet;
        update_jump_params = true, kwargs...
    )
    if !isempty(callback.discrete_callbacks)
        reset_aggregated_jumps!(
            integrator, uprev, callback.discrete_callbacks...,
            update_jump_params = update_jump_params, kwargs...
        )
    end
    return nothing
end

function reset_aggregated_jumps!(
        integrator, uprev, cb::DiscreteCallback, cbs...;
        update_jump_params = true, kwargs...
    )
    if cb.condition isa AbstractSSAJumpAggregator
        maj = cb.condition.ma_jumps
        update_jump_params && using_params(maj) &&
            update_parameters!(cb.condition.ma_jumps, integrator.p; kwargs...)
        cb.condition(cb, integrator.u, integrator.t, integrator)
    end
    reset_aggregated_jumps!(
        integrator, uprev, cbs...;
        update_jump_params = update_jump_params, kwargs...
    )
    return nothing
end

function reset_aggregated_jumps!(
        integrator, uprev, cb::DiscreteCallback;
        update_jump_params = true, kwargs...
    )
    if cb.condition isa AbstractSSAJumpAggregator
        maj = cb.condition.ma_jumps
        update_jump_params && using_params(maj) &&
            update_parameters!(cb.condition.ma_jumps, integrator.p; kwargs...)
        cb.condition(cb, integrator.u, integrator.t, integrator)
    end
    return nothing
end
