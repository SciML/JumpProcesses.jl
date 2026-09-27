"""
    gpu_falling_factorial(x, k, ::Type{RT})

The order `k` mass action term in a single species, `x (x-1) ... (x-k+1)`.
"""
@inline function gpu_falling_factorial(x, k, ::Type{RT}) where {RT}
    val = RT(x)
    specpop = RT(x)
    for _ in 2:k
        specpop -= one(RT)
        val *= specpop
    end
    return val
end

"""
    gpu_falling_factorial_deriv(x, k, ::Type{RT})

Derivative of [`gpu_falling_factorial`](@ref) in `x`. `compute_hor` rejects
reactions of order above 3, so only three cases can occur and each is written out
rather than differentiated numerically.

Branching on `k` is safe for warp divergence: the stoichiometric coefficients come
from arrays shared by every thread, so all threads of a warp take the same arm.
"""
@inline function gpu_falling_factorial_deriv(x, k, ::Type{RT}) where {RT}
    xr = RT(x)
    k == 1 && return one(RT)
    k == 2 && return 2 * xr - one(RT)
    return 3 * xr * xr - 6 * xr + 2 * one(RT)
end

"""
    gpu_drift(u, maj, ::Type{RT})

The deterministic drift `sum_j nu_j a_j(u)` of the mass action system.
"""
@inline function gpu_drift(u, maj, ::Type{RT}) where {RT}
    numjumps = gpu_num_jumps(maj)
    d = zero(MVector{length(u), RT})

    @inbounds for j in 1:numjumps
        a = gpu_evalrxrate(u, j, maj, RT)
        a == zero(RT) && continue

        lo = maj.ns_offsets[j]
        hi = maj.ns_offsets[j + 1] - one(eltype(maj.ns_offsets))
        for n in lo:hi
            d[maj.ns_species[n]] += RT(maj.ns_coeffs[n]) * a
        end
    end

    return SVector{length(u), RT}(d)
end

"""
    gpu_drift_jacobian(u, maj, coeff, ::Type{RT})

`I - coeff * d(drift)/du`, the Jacobian of the implicit residual.

The propensity derivatives are analytic. For a mass action reaction

`a_j = c_j * prod_k ff(u_{s_k}, m_k)`,

the derivative in one reactant is

`da_j/du_{s_m} = c_j * ff'(u_{s_m}, m_m) * prod_{k != m} ff(u_{s_k}, m_k)`,

and it is zero in every species that is not a reactant of `j`. The product over
the other factors is formed directly rather than by dividing the full product by
one factor, which would be undefined when that factor is zero.

`gpu_evalrxrate` clamps a propensity to zero once a reactant is too depleted to
supply its stoichiometry; the derivative is taken as zero there to match.
"""
@inline function gpu_drift_jacobian(u, maj, coeff, ::Type{RT}) where {RT}
    numjumps = gpu_num_jumps(maj)
    nspec = length(u)

    J = zero(MMatrix{nspec, nspec, RT})
    @inbounds for i in 1:nspec
        J[i, i] = one(RT)
    end

    @inbounds for j in 1:numjumps
        rlo = maj.rs_offsets[j]
        rhi = maj.rs_offsets[j + 1] - one(eltype(maj.rs_offsets))

        # Same depletion test as gpu_evalrxrate: if the propensity is clamped to
        # zero then so is every one of its derivatives.
        depleted = false
        for k in rlo:rhi
            specpop = RT(u[maj.rs_species[k]])
            for _ in 2:maj.rs_coeffs[k]
                specpop -= one(RT)
            end
            specpop <= zero(RT) && (depleted = true)
        end
        depleted && continue

        nlo = maj.ns_offsets[j]
        nhi = maj.ns_offsets[j + 1] - one(eltype(maj.ns_offsets))

        for m in rlo:rhi
            q = maj.rs_species[m]
            others = one(RT)
            for k in rlo:rhi
                k == m && continue
                others *= gpu_falling_factorial(u[maj.rs_species[k]], maj.rs_coeffs[k], RT)
            end
            dadu = maj.scaled_rates[j] *
                   gpu_falling_factorial_deriv(u[q], maj.rs_coeffs[m], RT) * others

            for n in nlo:nhi
                J[maj.ns_species[n], q] -= coeff * RT(maj.ns_coeffs[n]) * dadu
            end
        end
    end

    return SMatrix{nspec, nspec, RT}(J)
end

"""
    gpu_linsolve(J, b)

Solve `J x = b` by LU with partial pivoting, returning `(x, ok)`.

StaticArrays' own `\\` cannot be used here. For sizes above three it factorises
with `lu`, and both `lu` and the generated triangular substitution throw a
`SingularException`; a throw inside a GPU kernel traps and takes down the whole
launch, whereas the caller needs a recoverable signal so it can shrink `tau` and
retry. This reports a zero pivot through `ok` instead.
"""
@inline function gpu_linsolve(J::SMatrix{N, N, T}, b::SVector{N, T}) where {N, T}
    A = MMatrix{N, N, T}(J)
    x = MVector{N, T}(b)
    ok = true

    @inbounds for k in 1:N
        pivot = k
        amax = abs(A[k, k])
        for r in (k + 1):N
            ar = abs(A[r, k])
            if ar > amax
                amax = ar
                pivot = r
            end
        end

        if !(amax > zero(T))
            ok = false
            break
        end

        if pivot != k
            for c in 1:N
                tmp = A[k, c]
                A[k, c] = A[pivot, c]
                A[pivot, c] = tmp
            end
            tmp = x[k]
            x[k] = x[pivot]
            x[pivot] = tmp
        end

        akk = A[k, k]
        for r in (k + 1):N
            f = A[r, k] / akk
            A[r, k] = f
            for c in (k + 1):N
                A[r, c] -= f * A[k, c]
            end
            x[r] -= f * x[k]
        end
    end

    if ok
        @inbounds for k in N:-1:1
            s = x[k]
            for c in (k + 1):N
                s -= A[k, c] * x[c]
            end
            x[k] = s / A[k, k]
        end
    end

    return SVector{N, T}(x), ok
end

# Newton iterations allowed per step before the step is treated as failed. These
# systems converge in a handful of iterations; the cap only exists so a thread
# cannot spin.
const GPU_IMPLICIT_MAXITERS = 50

"""
    gpu_solve_implicit(u_current, maj, coeff, const_term, ::Type{RT})

Solve the implicit step for the state at the end of the leap, returning
`(u_predicted, converged)`.

The residual is

`F(u) = u - u_current - const_term - coeff * sum_j nu_j a_j(u)`,

which covers both formulations: the fully implicit step uses `coeff = tau` and
`const_term = 0`, and the trapezoidal step uses `coeff = tau/2` and
`const_term = (tau/2) * sum_j nu_j a_j(u_current)`. Because that second term does
not depend on `u` it is formed once by the caller rather than inside the
iteration, which is where the host version recomputes it.

Convergence is measured with a mixed absolute and relative tolerance. A pure
absolute `1e-6`, as the host uses, is unreachable in `Float32` once populations
reach a few thousand, since it falls below the spacing of the floats involved.
"""
@inline function gpu_solve_implicit(u_current, maj, coeff, const_term,
        ::Type{RT}) where {RT}
    nspec = length(u_current)
    u_cur = SVector{nspec, RT}(u_current)
    u = u_cur

    abstol = RT(1.0e-6)
    reltol = RT(1.0e-6)

    for _ in 1:GPU_IMPLICIT_MAXITERS
        F = u - u_cur - const_term - coeff * gpu_drift(u, maj, RT)

        scale = zero(RT)
        resid = zero(RT)
        @inbounds for i in 1:nspec
            resid = max(resid, abs(F[i]))
            scale = max(scale, abs(u[i]))
        end
        resid <= abstol + reltol * scale && return u, true

        J = gpu_drift_jacobian(u, maj, coeff, RT)
        delta, ok = gpu_linsolve(J, F)
        ok || return u, false

        u = u - delta

        # A diverging iterate is a failed step, not a reason to keep going.
        @inbounds for i in 1:nspec
            isfinite(u[i]) || return u, false
        end
    end

    return u, false
end

"""
    implicit_tau_leaping_kernel!(us, u0, maj, max_hor, max_stoich, saveat, t0, tend,
                                 epsilon, dtmin, trapezoidal)

Advance one implicit tau-leaping trajectory per thread, writing the state sampled
on the `saveat` grid into `us[trajectory, species, save_idx]`.

Each step solves for the state at the end of the leap, then draws the Poisson
counts from the propensities at that predicted state. The state itself advances
from `u_current`, not from the predicted state, which only supplies the
propensities.

`trapezoidal` is a `Val`, so the two formulations specialise into separate
kernels with no runtime branch.
"""
@kernel function implicit_tau_leaping_kernel!(us, u0, maj, @Const(max_hor),
        @Const(max_stoich), @Const(saveat), t0, tend, epsilon, dtmin,
        ::Val{trapezoidal}) where {trapezoidal}
    i = @index(Global, Linear)

    @inbounds begin
        RT = eltype(saveat)
        numjumps = gpu_num_jumps(maj)
        nsave = length(saveat)
        nspec = length(u0)
        rng = PoissonRandom.PassthroughRNG()

        u = u0
        t = RT(t0)

        # Grid points at or before the initial time hold the initial condition.
        sidx = 1
        while sidx <= nsave && saveat[sidx] <= t
            store_state!(us, u, sidx, i)
            sidx += 1
        end

        # Upper bound on the next leap, carried across iterations. This is what
        # makes a rejected step make progress: the implicit solve is
        # deterministic in (u_current, tau), so retrying at an unchanged tau
        # would fail in exactly the same way.
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

            # The trapezoidal residual carries a term in the propensities at the
            # current state. It does not change during the solve, so it is formed
            # once here rather than on every residual evaluation.
            half = one(RT) / 2
            coeff = trapezoidal ? half * tau : tau
            const_term = trapezoidal ?
                         half * tau * gpu_drift(u, maj, RT) :
                         zero(SVector{nspec, RT})

            u_predicted, converged = gpu_solve_implicit(u, maj, coeff, const_term, RT)

            if !converged
                # Out of room to shrink: report rather than silently accept an
                # unconverged state.
                tau <= dtmin && break
                tau_cap = tau / 2
                continue
            end

            # Sample the leap from the propensities at the predicted state, but
            # apply the counts to the current state.
            u_new = u
            for j in 1:numjumps
                lambda = gpu_evalrxrate(u_predicted, j, maj, RT) * tau
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

# Both implicit formulations share one host driver; they differ only in the
# residual, which the kernel selects on a `Val`.
const GPUImplicitLeapingAlgorithm = Union{
    SimpleImplicitTauLeaping, SimpleTrapezoidalLeaping}

@inline _is_trapezoidal(::SimpleImplicitTauLeaping) = false
@inline _is_trapezoidal(::SimpleTrapezoidalLeaping) = true

"""
    __solve(ensembleprob, alg, ::EnsembleGPUKernel; trajectories, saveat, ...)

Solve an ensemble of mass action `JumpProblem`s with one implicit tau-leaping
trajectory per GPU thread, for either `SimpleImplicitTauLeaping` or
`SimpleTrapezoidalLeaping`.

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
        alg::GPUImplicitLeapingAlgorithm,
        ensemblealg::EnsembleGPUKernel;
        trajectories,
        seed = nothing,
        dtmin = nothing,
        saveat = nothing,
        save_start = true,
        save_end = true,
        callback = nothing,
        kwargs...)
    algname = nameof(typeof(alg))

    if trajectories == 1
        return SciMLBase.__solve(ensembleprob, alg, EnsembleSerial(); trajectories = 1,
            seed, dtmin, saveat, save_start, save_end, callback, kwargs...)
    end

    callback === nothing ||
        error("EnsembleGPUKernel with $algname does not support callbacks, since they \
               would have to run inside the GPU kernel.")

    seed !== nothing && Random.seed!(seed)

    ensemblealg.backend === nothing ? backend = CPU() : backend = ensemblealg.backend

    jump_prob = ensembleprob.prob
    jump_prob isa JumpProblem ||
        error("EnsembleGPUKernel with $algname requires a JumpProblem, got $(typeof(jump_prob)).")

    validate_gpu_massaction_inputs(jump_prob) ||
        error("EnsembleGPUKernel with $algname only supports JumpProblems built from a \
               DiscreteProblem whose jumps are all MassActionJumps, with no user \
               callbacks. RegularJumps are not supported here because their rate and \
               `c` are arbitrary Julia functions that cannot be evaluated inside a GPU \
               kernel; rewrite them as MassActionJumps to use this solver.")

    # The leap size adapts to the state, so there is no trajectory length to
    # allocate up front. Sampling on a time grid gives every trajectory the same,
    # exactly known output size.
    saveat === nothing &&
        error("EnsembleGPUKernel with $algname requires `saveat`, since the leap size \
               adapts to the state and the number of steps a trajectory takes is not \
               known ahead of time. Pass a step (`saveat = 1.0`) or an explicit \
               collection of times.")

    ensembleprob.prob_func === SciMLBase.DEFAULT_PROB_FUNC ||
        error("EnsembleGPUKernel with $algname does not support a `prob_func`; the \
               reaction data is uploaded to the device once and shared by every \
               trajectory. Ensembles of differing problems have to be solved with a CPU \
               ensemble algorithm such as EnsembleThreads.")

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

    kernel = implicit_tau_leaping_kernel!(backend)
    kernel(us, u0, maj_gpu, max_hor_gpu, max_stoich_gpu, saveat_gpu, t0, tend,
        TT(alg.epsilon), TT(dtmin), Val(_is_trapezoidal(alg)); ndrange = trajectories)
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
