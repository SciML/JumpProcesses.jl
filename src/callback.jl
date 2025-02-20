using DiffEqCallbacks


mutable struct VariableRateJumpIntegrator{F, T, I}
    integrand_func::F
    integrand_values::IntegrandValues{T, I}
    integrand_cache::I
    accumulation_cache::I
end

function (integrator_callback::VariableRateJumpIntegrator)(integrator)
    # Determine the number of Gaussian points based on the solver's order
    n = if integrator.sol.prob isa Union{SDEProblem, RODEProblem}
        10  # Default for SDE/RODE problems
    else
        div(SciMLBase.alg_order(integrator.alg) + 1, 2)
    end

    # Zero out the accumulation cache
    recursive_zero!(integrator_callback.accumulation_cache)

    # Perform Gaussian quadrature integration
    for i in 1:n
        t_temp = ((integrator.t - integrator.tprev) / 2) * gauss_points[n][i] +
                 (integrator.t + integrator.tprev) / 2

        if DiffEqBase.isinplace(integrator.sol.prob)
            curu = first(get_tmp_cache(integrator))
            integrator(curu, t_temp)

            if integrator_callback.integrand_cache == nothing
                recursive_axpy!(
                    gauss_weights[n][i],
                    integrator_callback.integrand_func(curu, t_temp, integrator),
                    integrator_callback.accumulation_cache
                )
            else
                integrator_callback.integrand_func(
                    integrator_callback.integrand_cache, curu, t_temp, integrator
                )
                recursive_axpy!(
                    gauss_weights[n][i],
                    integrator_callback.integrand_cache,
                    integrator_callback.accumulation_cache
                )
            end
        else
            recursive_axpy!(
                gauss_weights[n][i],
                integrator_callback.integrand_func(integrator(t_temp), t_temp, integrator),
                integrator_callback.accumulation_cache
            )
        end
    end

    # Scale the accumulated result
    recursive_scalar_mul!(
        integrator_callback.accumulation_cache, (integrator.t - integrator.tprev) / 2
    )

    # Save the results
    push!(integrator_callback.integrand_values.ts, integrator.t)
    push!(integrator_callback.integrand_values.integrand, recursive_copy(integrator_callback.accumulation_cache))

    # Ensure the integrator state is not modified
    u_modified!(integrator, false)
end