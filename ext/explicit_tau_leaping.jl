"""
    gpu_compute_gi(u, max_hor, max_stoich, i, ::Type{RT})

Bound on the relative change of the propensities in species `i`, `g_i`, as in
Cao et al. (2006), Section IV, equation (27).

Mirrors `JumpProcesses.compute_gi`. `max_hor[i]` is the highest order of any
reaction that consumes species `i`, and `max_stoich[i]` is the largest
stoichiometry it appears with in such a reaction; both are precomputed on the
host, so the device only evaluates the closed form.
"""
@inline function gpu_compute_gi(u, max_hor, max_stoich, i, ::Type{RT}) where {RT}
    onert = one(RT)
    two = 2 * onert
    three = 3 * onert

    @inbounds begin
        hor = max_hor[i]
        stoich = max_stoich[i]
        x = RT(u[i])
    end

    # Written with `ifelse` rather than branches so that threads of a warp that
    # land on different reaction orders do not diverge. Both arms are therefore
    # always evaluated: when x_i is too small the reciprocals below divide by
    # zero or go negative, but IEEE division yields +/-Inf rather than trapping,
    # and those values are exactly the ones the selects discard.
    inv_xm1 = onert / (x - onert)     # 1 / (x_i - 1)
    inv_xm2 = two / (x - two)         # 2 / (x_i - 2)

    # 2S_i -> products; fall back to 2 when x_i <= 1
    g2_s2 = ifelse(x > onert, two + inv_xm1, two)
    # 2S_i + S_k -> products is exactly 3/2 of the order 2 form, including its
    # fallback, so it reuses it rather than recomputing the reciprocal
    g3_s2 = (three / two) * g2_s2
    # 3S_i -> products; fall back to 3 when x_i <= 2
    g3_s3 = ifelse(x > two, three + inv_xm1 + inv_xm2, three)

    g_hor2 = ifelse(stoich == 1, two, g2_s2)
    g_hor3 = ifelse(stoich == 1, three, ifelse(stoich == 2, g3_s2, g3_s3))

    # hor 0 and 1, and anything unsupported, take the default of 1
    return ifelse(hor == 2, g_hor2, ifelse(hor == 3, g_hor3, onert))
end

"""
    gpu_compute_tau(u, maj, max_hor, max_stoich, epsilon, dtmin, ::Type{RT})

Select the leap size at state `u`, following equation (20) of Cao et al. (2006),
and report whether any reaction can still fire.

Mirrors `JumpProcesses.compute_tau`, with one difference in how the moments are
accumulated. The host version sums `mu_i = sum_j nu_ij a_j` and
`sigma_i^2 = sum_j nu_ij^2 a_j` over the dense stoichiometry matrix. Here each
propensity is evaluated once and scattered into per-species accumulators through
the net stoichiometry, which visits only the nonzeros and so needs neither the
dense matrix nor a propensity cache in global memory. The two agree because the
entries the host walks past are zero.
"""
@inline function gpu_compute_tau(u, maj, max_hor, max_stoich, epsilon, dtmin,
        ::Type{RT}) where {RT}
    numjumps = gpu_num_jumps(maj)
    nspec = length(u)

    mu = zero(MVector{nspec, RT})
    sigma2 = zero(MVector{nspec, RT})
    can_react = false

    @inbounds for j in 1:numjumps
        a = gpu_evalrxrate(u, j, maj, RT)
        a > zero(RT) && (can_react = true)

        lo = maj.ns_offsets[j]
        hi = maj.ns_offsets[j + 1] - one(eltype(maj.ns_offsets))
        for k in lo:hi
            spec = maj.ns_species[k]
            v = RT(maj.ns_coeffs[k])
            mu[spec] += v * a
            sigma2[spec] += v * v * a
        end
    end

    # No reaction can fire, so there is no leap to size.
    can_react || return dtmin, false

    tau = typemax(RT)
    @inbounds for i in 1:nspec
        gi = gpu_compute_gi(u, max_hor, max_stoich, i, RT)
        bound = max(epsilon * RT(u[i]) / gi, one(RT))   # max(epsilon x_i / g_i, 1)
        m = abs(mu[i])
        s = sigma2[i]
        zed = zero(RT)
        # branch-free for the same reason as above; a zero denominator gives Inf,
        # which the select discards in favour of typemax
        mu_term = ifelse(m > zed, bound / m, typemax(RT))              # eq. (8), first term
        sigma_term = ifelse(s > zed, bound * bound / s, typemax(RT))   # eq. (8), second term
        tau = min(tau, mu_term, sigma_term)
    end

    return max(tau, dtmin), true
end

"""
    explicit_tau_leaping_kernel!(us, u0, maj, max_hor, max_stoich, saveat, t0, tend,
                                 epsilon, dtmin)

Advance one explicit tau-leaping trajectory per thread, writing the state
sampled on the `saveat` grid into `us[trajectory, species, save_idx]`.

The leap size is chosen per trajectory and per step, so trajectories fall out of
step with one another; the grid is what gives them a common, fixed-size output.
`tau` is shortened to land exactly on the next grid point when a step would
overshoot it, which is what the host solver does too, so a saved value is always
a state the trajectory actually occupied rather than an interpolation.

The state lives in registers as an `SVector` and the propensities are evaluated
twice per step, once to size the leap and once to draw the counts, so the kernel
needs no per-thread scratch in global memory.
"""
@kernel function explicit_tau_leaping_kernel!(us, u0, maj, @Const(max_hor),
        @Const(max_stoich), @Const(saveat), t0, tend, epsilon, dtmin)
    i = @index(Global, Linear)

    @inbounds begin
        RT = eltype(saveat)
        numjumps = gpu_num_jumps(maj)
        nsave = length(saveat)
        rng = PoissonRandom.PassthroughRNG()

        u = u0
        t = RT(t0)

        # Grid points at or before the initial time hold the initial condition.
        sidx = 1
        while sidx <= nsave && saveat[sidx] <= t
            store_state!(us, u, sidx, i)
            sidx += 1
        end

        # Upper bound on the next leap, carried across iterations so that a
        # rejected leap actually retries with a smaller one. Recomputing tau from
        # scratch would otherwise reproduce the rejected size.
        tau_cap = typemax(RT)

        while sidx <= nsave
            tau, can_react = gpu_compute_tau(
                u, maj, max_hor, max_stoich, epsilon, dtmin, RT)

            # Nothing can fire again, so the path is constant on the rest of the grid.
            if !can_react
                while sidx <= nsave
                    store_state!(us, u, sidx, i)
                    sidx += 1
                end
                break
            end

            tau = min(tau, tau_cap, tend - t)
            # Shorten the leap so it lands on the next grid point rather than past it.
            if saveat[sidx] - t < tau
                tau = saveat[sidx] - t
            end

            # Poisson counts come from the backend's device RNG, the same
            # generator the SimpleTauLeaping kernel draws from.
            u_new = u
            for j in 1:numjumps
                lambda = gpu_evalrxrate(u, j, maj, RT) * tau
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
                tau_cap = tau / 2
                continue
            end

            t += tau
            u = u_new

            if t >= saveat[sidx]
                store_state!(us, u, sidx, i)
                sidx += 1
            end
            tau_cap = typemax(RT)   # release the bound after a good leap
        end
    end
end

"""
    __solve(ensembleprob, ::SimpleExplicitTauLeaping, ::EnsembleGPUKernel;
            trajectories, saveat, ...)

Solve an ensemble of mass action `JumpProblem`s with one explicit tau-leaping
trajectory per GPU thread.

`saveat` is required. The leap size adapts to the state, so the number of steps a
trajectory takes is not known in advance and the solution is sampled onto a fixed
time grid instead. Each returned solution therefore contains exactly the grid
points, as if the problem had been built with `save_positions = (false, false)`.

Randomness comes from the backend's own device RNG rather than the `rng` stored
in the `JumpProblem`. `seed` is applied to the ambient generator, so it makes a
run reproducible on backends that draw from it, such as `CPU()`; seeding a GPU
backend is done through that backend's own `seed!`.

The reaction data is uploaded to the device once and shared by every thread, so
all trajectories solve the same problem and a `prob_func` is not supported.
"""
function SciMLBase.__solve(ensembleprob::SciMLBase.AbstractEnsembleProblem,
        alg::SimpleExplicitTauLeaping,
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
        error("EnsembleGPUKernel with SimpleExplicitTauLeaping does not support callbacks, \
               since they would have to run inside the GPU kernel.")

    seed !== nothing && Random.seed!(seed)

    ensemblealg.backend === nothing ? backend = CPU() : backend = ensemblealg.backend

    jump_prob = ensembleprob.prob
    jump_prob isa JumpProblem ||
        error("EnsembleGPUKernel with SimpleExplicitTauLeaping requires a JumpProblem, got $(typeof(jump_prob)).")

    validate_gpu_massaction_inputs(jump_prob) ||
        error("EnsembleGPUKernel with SimpleExplicitTauLeaping only supports JumpProblems \
               built from a DiscreteProblem whose jumps are all MassActionJumps, with no \
               user callbacks. RegularJumps are not supported here because their rate and \
               `c` are arbitrary Julia functions that cannot be evaluated inside a GPU \
               kernel; rewrite them as MassActionJumps to use this solver.")

    # The leap size adapts to the state, so there is no trajectory length to
    # allocate up front. Sampling on a time grid gives every trajectory the same,
    # exactly known output size.
    saveat === nothing &&
        error("EnsembleGPUKernel with SimpleExplicitTauLeaping requires `saveat`, since \
               the leap size adapts to the state and the number of steps a trajectory \
               takes is not known ahead of time. Pass a step (`saveat = 1.0`) or an \
               explicit collection of times.")

    ensembleprob.prob_func === SciMLBase.DEFAULT_PROB_FUNC ||
        error("EnsembleGPUKernel with SimpleExplicitTauLeaping does not support a \
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

    # Every trajectory starts from the same state, so it is passed by value and
    # lives in registers rather than being uploaded once per thread.
    u0 = SVector{state_dim, ET}(prob.u0)
    saveat_gpu = adapt(backend, save_times)
    maj_gpu = GPUMassActionJump(maj, backend, TT)

    # g_i depends only on the reactant stoichiometry, so the per-species highest
    # order and largest stoichiometry are precomputed once on the host.
    reactant_stoch = maj.reactant_stoch
    numjumps = JumpProcesses.get_num_majumps(maj)
    hor = JumpProcesses.compute_hor(reactant_stoch, numjumps)
    max_hor, max_stoich = JumpProcesses.precompute_reaction_conditions(
        reactant_stoch, hor, state_dim, numjumps)
    max_hor_gpu = adapt(backend, convert(Vector{Int32}, max_hor))
    max_stoich_gpu = adapt(backend, convert(Vector{Int32}, max_stoich))

    us = allocate(backend, ET, (trajectories, state_dim, nsave))

    kernel = explicit_tau_leaping_kernel!(backend)
    kernel(us, u0, maj_gpu, max_hor_gpu, max_stoich_gpu, saveat_gpu, t0, tend,
        TT(alg.epsilon), TT(dtmin); ndrange = trajectories)
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
