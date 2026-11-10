"""
    reset_aggregated_jumps!(integrator, uprev = nothing)

Reset the state of jump processes and associated solvers following a change
in parameters or such. Rate updates are handled automatically by `initialize!`
via `fill_scaled_rates!`.
"""
function reset_aggregated_jumps!(integrator, uprev = nothing; kwargs...)
    if haskey(kwargs, :update_jump_params)
        throw(ArgumentError("`update_jump_params` keyword argument has been removed. " *
                            "Rate updates are now handled automatically by `initialize!` " *
                            "via `fill_scaled_rates!`."))
    end
    reset_aggregated_jumps!(integrator, uprev, integrator.opts.callback)
    nothing
end

function reset_aggregated_jumps!(integrator, uprev, callback::Nothing)
    nothing
end

function reset_aggregated_jumps!(integrator, uprev, callback::CallbackSet)
    if !isempty(callback.discrete_callbacks)
        reset_aggregated_jumps!(integrator, uprev, callback.discrete_callbacks...)
    end
    nothing
end

function reset_aggregated_jumps!(integrator, uprev, cb::DiscreteCallback, cbs...)
    if cb.condition isa AbstractSSAJumpAggregator
        cb.condition(cb, integrator.u, integrator.t, integrator)
    end
    reset_aggregated_jumps!(integrator, uprev, cbs...)
    nothing
end

function reset_aggregated_jumps!(integrator, uprev, cb::DiscreteCallback)
    if cb.condition isa AbstractSSAJumpAggregator
        cb.condition(cb, integrator.u, integrator.t, integrator)
    end
    nothing
end
