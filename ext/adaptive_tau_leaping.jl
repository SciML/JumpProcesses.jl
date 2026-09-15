"""
    gpu_is_stiff(u, maj, epsilon, ::Type{RT})

Whether the system looks stiff at `u`, from the spread of the propensities, as in
Cao et al. (2007), Section III.B.

This is the default host test. The alternative there, an eigenvalue ratio of the
drift Jacobian, has no kernel-safe form and is rejected on the host side instead.
"""
@inline function gpu_is_stiff(u, maj, epsilon, ::Type{RT}) where {RT}
    numjumps = gpu_num_jumps(maj)

    npos = 0
    amax = zero(RT)
    amin = typemax(RT)
    @inbounds for j in 1:numjumps
        a = gpu_evalrxrate(u, j, maj, RT)
        a > zero(RT) || continue
        npos += 1
        amax = max(amax, a)
        amin = min(amin, a)
    end

    # A single active reaction has no ratio to speak of.
    npos <= 1 && return false

    total = zero(RT)
    @inbounds for i in eachindex(u)
        total += RT(u[i])
    end

    return amax / amin > epsilon * total
end

"""
    gpu_compute_tau_implicit(u, maj, max_hor, max_stoich, epsilon, dtmin,
                             implicit_epsilon_factor, ::Type{RT})

Leap size for a step that will be taken implicitly, following Cao et al. (2007),
Section III.A.

An implicit step is not restricted by the fast timescale it damps, so the size is
chosen at the state one explicit leap ahead and with the tolerance relaxed by
`implicit_epsilon_factor`.
"""
@inline function gpu_compute_tau_implicit(u, maj, max_hor, max_stoich, epsilon, dtmin,
        implicit_epsilon_factor, ::Type{RT}) where {RT}
    tau_explicit, can_react = gpu_compute_tau(
        u, maj, max_hor, max_stoich, epsilon, dtmin, RT)
    can_react || return dtmin, false

    # One deterministic drift step of that size, clamped at zero.
    predicted = SVector{length(u), RT}(u) + tau_explicit * gpu_drift(u, maj, RT)
    predicted = max.(predicted, zero(RT))

    tau, _ = gpu_compute_tau(predicted, maj, max_hor, max_stoich,
        epsilon * implicit_epsilon_factor, dtmin, RT)
    return max(tau, dtmin), true
end

"""
    adaptive_tau_leaping_kernel!(us, u0, maj, max_hor, max_stoich, saveat, t0, tend,
                                 epsilon, dtmin, implicit_epsilon_factor, trapezoidal)

Advance one adaptive tau-leaping trajectory per thread, choosing between an
explicit and an implicit step at every step according to whether the system
currently looks stiff.

Only a stiff step pays for the nonlinear solve. `trapezoidal` is a `Val` giving
the implicit formulation to use when one is taken.
"""
@kernel function adaptive_tau_leaping_kernel!(us, u0, maj, @Const(max_hor),
        @Const(max_stoich), @Const(saveat), t0, tend, epsilon, dtmin,
        implicit_epsilon_factor, ::Val{trapezoidal}) where {trapezoidal}
    i = @index(Global, Linear)

    @inbounds begin
        RT = eltype(saveat)
        numjumps = gpu_num_jumps(maj)
        nsave = length(saveat)
        nspec = length(u0)
        rng = PoissonRandom.PassthroughRNG()

        u = u0
        t = RT(t0)

        sidx = 1
        while sidx <= nsave && saveat[sidx] <= t
            store_state!(us, u, sidx, i)
            sidx += 1
        end

        # Only an implicit rejection needs this bound: an explicit retry redraws
        # its Poisson counts and so can succeed at an unchanged tau, while the
        # implicit solve is deterministic in (u, tau).
        tau_cap = typemax(RT)

        while sidx <= nsave
            use_implicit = gpu_is_stiff(u, maj, epsilon, RT)

            tau, can_react = if use_implicit
                gpu_compute_tau_implicit(u, maj, max_hor, max_stoich, epsilon, dtmin,
                    implicit_epsilon_factor, RT)
            else
                gpu_compute_tau(u, maj, max_hor, max_stoich, epsilon, dtmin, RT)
            end

            if !can_react
                while sidx <= nsave
                    store_state!(us, u, sidx, i)
                    sidx += 1
                end
                break
            end

            tau = min(tau, tau_cap, tend - t)
            if saveat[sidx] - t < tau
                tau = saveat[sidx] - t
            end

            # The state whose propensities drive the Poisson draws: the current
            # state for an explicit step, the implicitly predicted one otherwise.
            u_rates = u
            if use_implicit
                half = one(RT) / 2
                coeff = trapezoidal ? half * tau : tau
                const_term = trapezoidal ?
                             half * tau * gpu_drift(u, maj, RT) :
                             zero(SVector{nspec, RT})

                u_predicted, converged = gpu_solve_implicit(u, maj, coeff, const_term, RT)
                if !converged
                    tau <= dtmin && break
                    tau_cap = tau / 2
                    continue
                end
                u_rates = u_predicted
            end

            u_new = u
            for j in 1:numjumps
                lambda = gpu_evalrxrate(u_rates, j, maj, RT) * tau
                lambda <= zero(RT) && continue
                count = pois_rand(rng, lambda)
                count == 0 && continue

                lo = maj.ns_offsets[j]
                hi = maj.ns_offsets[j + 1] - one(eltype(maj.ns_offsets))
                for k in lo:hi
                    spec = maj.ns_species[k]
                    u_new = setindex(u_new, u_new[spec] + maj.ns_coeffs[k] * count, spec)
                end
            end

            negative = false
            for k in eachindex(u_new)
                u_new[k] < zero(eltype(u_new)) && (negative = true)
            end
            if negative
                # Halve tau to avoid negative populations, as per Cao et al. (2006), Section 3.3
                tau <= dtmin && break
                use_implicit && (tau_cap = tau / 2)
                continue
            end

            t += tau
            u = u_new

            if t >= saveat[sidx]
                store_state!(us, u, sidx, i)
                sidx += 1
            end
            tau_cap = typemax(RT)
        end
    end
end

"""
    __solve(ensembleprob, ::SimpleAdaptiveTauLeaping, ::EnsembleGPUKernel;
            trajectories, saveat, ...)

Solve an ensemble of mass action `JumpProblem`s with one adaptive tau-leaping
trajectory per GPU thread, switching between an explicit and an implicit step
according to a stiffness test.

`eigenvalue_check = true` is not supported: it needs the eigenvalues of the drift
Jacobian, which has no form that can run inside a kernel. The default
propensity-ratio test is fully supported.

`saveat` is required, randomness comes from the backend's own device RNG, and a
`prob_func` is not supported, all for the same reasons as the other mass action
kernels.
"""
function SciMLBase.__solve(ensembleprob::SciMLBase.AbstractEnsembleProblem,
        alg::SimpleAdaptiveTauLeaping,
        ensemblealg::EnsembleGPUKernel;
        trajectories,
        seed = nothing,
        dtmin = nothing,
        saveat = nothing,
        save_start = true,
        save_end = true,
        callback = nothing,
        kwargs...)
    if trajectories == 1
        return SciMLBase.__solve(ensembleprob, alg, EnsembleSerial(); trajectories = 1,
            seed, dtmin, saveat, save_start, save_end, callback, kwargs...)
    end

    callback === nothing ||
        error("EnsembleGPUKernel with SimpleAdaptiveTauLeaping does not support \
               callbacks, since they would have to run inside the GPU kernel.")

    alg.eigenvalue_check &&
        error("EnsembleGPUKernel with SimpleAdaptiveTauLeaping does not support \
               `eigenvalue_check = true`; judging stiffness that way needs the \
               eigenvalues of the drift Jacobian, which cannot be computed inside a GPU \
               kernel. Use the default propensity ratio test, or solve the ensemble with \
               a CPU ensemble algorithm such as EnsembleThreads.")

    seed !== nothing && Random.seed!(seed)

    ensemblealg.backend === nothing ? backend = CPU() : backend = ensemblealg.backend

    jump_prob = ensembleprob.prob
    jump_prob isa JumpProblem ||
        error("EnsembleGPUKernel with SimpleAdaptiveTauLeaping requires a JumpProblem, got $(typeof(jump_prob)).")

    validate_gpu_massaction_inputs(jump_prob) ||
        error("EnsembleGPUKernel with SimpleAdaptiveTauLeaping only supports JumpProblems \
               built from a DiscreteProblem whose jumps are all MassActionJumps, with no \
               user callbacks. RegularJumps are not supported here because their rate and \
               `c` are arbitrary Julia functions that cannot be evaluated inside a GPU \
               kernel; rewrite them as MassActionJumps to use this solver.")

    saveat === nothing &&
        error("EnsembleGPUKernel with SimpleAdaptiveTauLeaping requires `saveat`, since \
               the leap size adapts to the state and the number of steps a trajectory \
               takes is not known ahead of time. Pass a step (`saveat = 1.0`) or an \
               explicit collection of times.")

    ensembleprob.prob_func === SciMLBase.DEFAULT_PROB_FUNC ||
        error("EnsembleGPUKernel with SimpleAdaptiveTauLeaping does not support a \
               `prob_func`; the reaction data is uploaded to the device once and shared \
               by every trajectory. Ensembles of differing problems have to be solved \
               with a CPU ensemble algorithm such as EnsembleThreads.")

    prob = jump_prob.prob
    maj = jump_prob.massaction_jump

    TT = float(eltype(prob.tspan))
    t0, tend = TT(prob.tspan[1]), TT(prob.tspan[2])
    dtmin === nothing && (dtmin = 1.0e-10 * one(TT))

    save_times = gpu_save_times(saveat, (t0, tend), save_start, save_end)
    nsave = length(save_times)

    state_dim = length(prob.u0)
    ET = eltype(prob.u0)
    state_dim > 0 || error("The state must have at least one species.")

    u0 = SVector{state_dim, ET}(prob.u0)
    saveat_gpu = adapt(backend, save_times)
    maj_gpu = GPUMassActionJump(maj, backend, TT)

    reactant_stoch = maj.reactant_stoch
    numjumps = JumpProcesses.get_num_majumps(maj)
    hor = JumpProcesses.compute_hor(reactant_stoch, numjumps)
    max_hor, max_stoich = JumpProcesses.precompute_reaction_conditions(
        reactant_stoch, hor, state_dim, numjumps)
    max_hor_gpu = adapt(backend, convert(Vector{Int32}, max_hor))
    max_stoich_gpu = adapt(backend, convert(Vector{Int32}, max_stoich))

    us = allocate(backend, ET, (trajectories, state_dim, nsave))

    # `implicit_alg` is an algorithm object with no device representation, so the
    # formulation is resolved here and passed as a compile-time flag.
    kernel = adaptive_tau_leaping_kernel!(backend)
    kernel(us, u0, maj_gpu, max_hor_gpu, max_stoich_gpu, saveat_gpu, t0, tend,
        TT(alg.epsilon), TT(dtmin), TT(alg.implicit_epsilon_factor),
        Val(_is_trapezoidal(alg.implicit_alg)); ndrange = trajectories)
    KernelAbstractions.synchronize(backend)

    _us = Array(us)

    time = @elapsed sol = [begin
                               @views ensembleprob.output_func(
                                   SciMLBase.build_solution(prob,
                                       alg,
                                       save_times,
                                       [_us[i, :, j] for j in 1:nsave],
                                       k = nothing,
                                       stats = nothing,
                                       calculate_error = false,
                                       retcode = ReturnCode.Success),
                                   i)[1]
                           end
                           for i in 1:trajectories]

    return SciMLBase.EnsembleSolution(sol, time, true)
end
