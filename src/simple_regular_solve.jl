"""
    SimpleTauLeaping()

Fixed-step tau-leaping algorithm for pure [`RegularJump`](@ref) problems.

Use `SimpleTauLeaping` with `JumpProblem(prob, PureLeaping(), regular_jump)` and pass
the timestep through the `dt` keyword to `solve`.

## Keyword Arguments

The algorithm constructor has no fields or keyword arguments. The `solve` method accepts:

  - `dt`: Required fixed timestep.
  - `seed`: Optional random seed for the jump problem RNG.
  - `saveat`: Optional scalar interval or collection of save times.
  - `save_start`: Whether to save the initial time. Defaults follow SciML save conventions.
  - `save_end`: Whether to save the final time. Defaults follow SciML save conventions.

## Returns

  - A stateless `SciMLBase.AbstractDEAlgorithm` value.

## Examples

```julia
using JumpProcesses, DiffEqBase

rate!(out, u, p, t) = (out[1] = 0.1 * u[1])
affect!(du, u, p, t, counts, mark) = (du[1] = -counts[1])
rj = RegularJump(rate!, affect!, 1)

prob = DiscreteProblem([20], (0.0, 2.0))
jprob = JumpProblem(prob, PureLeaping(), rj)
sol = solve(jprob, SimpleTauLeaping(); dt = 0.1)
```
"""
struct SimpleTauLeaping <: SciMLBase.AbstractDEAlgorithm end

"""
    SimpleExplicitTauLeaping(; epsilon = 0.05)
    SimpleExplicitTauLeaping(epsilon)

Adaptive explicit tau-leaping algorithm for pure [`MassActionJump`](@ref) problems.

Use `SimpleExplicitTauLeaping` with `JumpProblem(prob, PureLeaping(), mass_action_jump)`.
The algorithm computes step sizes from the mass-action propensities and the error-control
parameter `epsilon`.

## Arguments

  - `epsilon`: Positive floating-point error-control parameter. Smaller values generally
    produce smaller steps.

## Fields

  - `epsilon`: Stored error-control parameter used by the adaptive leaping step selector.

## Returns

  - A `SciMLBase.AbstractDEAlgorithm` value.

## Examples

```julia
using JumpProcesses, DiffEqBase

maj = MassActionJump([0.1], [[1 => 1]], [[1 => -1]])
prob = DiscreteProblem([20], (0.0, 2.0))
jprob = JumpProblem(prob, PureLeaping(), maj)
sol = solve(jprob, SimpleExplicitTauLeaping())
```
"""
struct SimpleExplicitTauLeaping{T <: AbstractFloat} <: SciMLBase.AbstractDEAlgorithm
    epsilon::T  # Error control parameter
end

SimpleExplicitTauLeaping(; epsilon = 0.05) = SimpleExplicitTauLeaping(epsilon)

"""
$(TYPEDEF)

Abstract supertype for the nonlinear formulations
[`SimpleImplicitTauLeaping`](@ref) can solve at each step.
"""
abstract type AbstractImplicitSolver end

"""
$(TYPEDEF)

Solve each implicit tau-leaping step in its unmodified, fully implicit form,

```math
X(t + \\tau) = X(t) + \\sum_j \\nu_j a_j(X(t + \\tau)) \\tau
```

as in Rathinam et al. (2003) and Cao et al. (2004).
"""
struct NewtonImplicitSolver <: AbstractImplicitSolver end

"""
$(TYPEDEF)

Solve each implicit tau-leaping step in trapezoidal form, averaging the
propensities at the current and the new state,

```math
X(t + \\tau) = X(t) + \\sum_j \\nu_j \\frac{a_j(X(t)) + a_j(X(t + \\tau))}{2} \\tau.
```

The trapezoidal formulation damps the excessive stiffness of the fully implicit
step and keeps the equilibrium distribution closer to the exact one.
"""
struct TrapezoidalImplicitSolver <: AbstractImplicitSolver end

"""
$(TYPEDEF)

An implicit tau-leaping method for stiff pure-jump problems.

Explicit tau-leaping is limited by the fastest reaction in the system, so a
stiff model forces a step size far smaller than the timescale of interest.
Each step here instead solves a nonlinear equation for the new state, which
lifts that restriction; see Rathinam et al. (2003) and Cao et al. (2004).

The deterministic part of the step is taken implicitly and the fluctuations are
then sampled with Poisson random variables, after which the step is rejected and
`tau` halved if it would drive a population negative.

## Fields

$(FIELDS)

## Notes

  - Only works with `JumpProblem`s defined from `DiscreteProblem`s that contain
    only a `MassActionJump`, built with the `PureLeaping()` aggregator.
  - Supports `saveat`, `save_start` and `save_end`.

## Examples

```julia
using JumpProcesses

maj = MassActionJump([1.0, 1.0], [[1 => 1], [2 => 1]], [[1 => -1, 2 => 1], [1 => 1, 2 => -1]])
prob = DiscreteProblem([100, 100], (0.0, 10.0))
jprob = JumpProblem(prob, PureLeaping(), maj)
sol = solve(jprob, SimpleImplicitTauLeaping())
```
"""
struct SimpleImplicitTauLeaping{T <: AbstractFloat, S <: AbstractImplicitSolver} <:
       SciMLBase.AbstractDEAlgorithm
    """Error control parameter used when selecting `tau`."""
    epsilon::T
    """The nonlinear formulation solved at each step."""
    solver::S
end

function SimpleImplicitTauLeaping(; epsilon = 0.05, solver = NewtonImplicitSolver())
    SimpleImplicitTauLeaping(epsilon, solver)
end

function validate_pure_leaping_inputs(jump_prob::JumpProblem, alg)
    if !(jump_prob.aggregator isa PureLeaping)
        @warn "When using $alg, please pass PureLeaping() as the aggregator to the \
        JumpProblem, i.e. call JumpProblem(::DiscreteProblem, PureLeaping(),...). \
        Passing $(jump_prob.aggregator) is deprecated and will be removed in the next breaking release."
    end
    isempty(jump_prob.jump_callback.continuous_callbacks) &&
        isempty(jump_prob.jump_callback.discrete_callbacks) &&
        isempty(jump_prob.constant_jumps) &&
        isempty(jump_prob.variable_jumps) &&
        get_num_majumps(jump_prob.massaction_jump) == 0 &&
        jump_prob.regular_jump !== nothing
end

function validate_pure_leaping_inputs(jump_prob::JumpProblem,
        alg::Union{SimpleExplicitTauLeaping, SimpleImplicitTauLeaping})
    if !(jump_prob.aggregator isa PureLeaping)
        @warn "When using $alg, please pass PureLeaping() as the aggregator to the \
        JumpProblem, i.e. call JumpProblem(::DiscreteProblem, PureLeaping(),...). \
        Passing $(jump_prob.aggregator) is deprecated and will be removed in the next breaking release."
    end
    isempty(jump_prob.jump_callback.continuous_callbacks) &&
        isempty(jump_prob.jump_callback.discrete_callbacks) &&
        isempty(jump_prob.constant_jumps) &&
        isempty(jump_prob.variable_jumps) &&
        jump_prob.massaction_jump !== nothing
end

"""
    _process_saveat(saveat, tspan, save_start, save_end)

Process `saveat` into a sorted vector of strictly interior save times (excluding
both `tspan` endpoints), and resolve `save_start`/`save_end` defaults following
OrdinaryDiffEq conventions.

Endpoint saving is controlled purely by the returned `save_start`/`save_end`
flags. When the user passes `nothing` for these, defaults are:
- No saveat or saveat is a Number: `true` for both.
- saveat is a collection: `true` if the corresponding endpoint is `in` the collection.
"""
function _process_saveat(saveat, tspan, save_start, save_end)
    t0, tf = tspan
    if isnothing(saveat)
        saveat_vec = Vector{typeof(t0)}()
        _save_start = something(save_start, true)
        _save_end = something(save_end, true)
    elseif saveat isa Number
        saveat_vec = collect(t0 + saveat:saveat:tf)
        if !isempty(saveat_vec) && last(saveat_vec) == tf
            pop!(saveat_vec)
        end
        _save_start = something(save_start, true)
        _save_end = something(save_end, true)
    else
        saveat_vec = sort!(collect(saveat))
        _save_start = something(save_start, insorted(t0, saveat_vec))
        _save_end = something(save_end, insorted(tf, saveat_vec))
        lo = searchsortedlast(saveat_vec, t0) + 1
        hi = searchsortedfirst(saveat_vec, tf) - 1
        saveat_vec = saveat_vec[lo:hi]
    end
    return saveat_vec, _save_start, _save_end
end

function DiffEqBase.solve(jump_prob::JumpProblem, alg::SimpleTauLeaping;
        seed = nothing, dt = error("dt is required for SimpleTauLeaping."),
        saveat = nothing, save_start = nothing, save_end = nothing)
    validate_pure_leaping_inputs(jump_prob, alg) ||
        error("SimpleTauLeaping can only be used with PureLeaping JumpProblems with only RegularJumps.")

    (; prob, rng) = jump_prob
    (seed !== nothing) && seed!(rng, seed)

    rj = jump_prob.regular_jump
    rate = rj.rate # rate function rate(out,u,p,t)
    numjumps = rj.numjumps # used for size information (# of jump processes)
    c = rj.c # matrix-free operator c(u_buffer, uprev, tprev, counts, p, mark)

    if !isnothing(rj.mark_dist) == nothing # https://github.com/JuliaDiffEq/DifferentialEquations.jl/issues/250
        error("Mark distributions are currently not supported in SimpleTauLeaping")
    end

    u0 = copy(prob.u0)
    du = similar(u0)
    rate_cache = zeros(float(eltype(u0)), numjumps)

    tspan = prob.tspan
    p = prob.p

    n = Int((tspan[2] - tspan[1]) / dt) + 1

    saveat_times, save_start, save_end = _process_saveat(saveat, tspan, save_start, save_end)

    if save_start
        usave = [copy(u0)]
        tsave = typeof(tspan[1])[tspan[1]]
    else
        usave = typeof(u0)[]
        tsave = typeof(tspan[1])[]
    end
    save_idx = 1

    # Pre-allocate working buffers — swap each step to avoid copying
    uprev = u0          # u0 is already a copy
    u_new = similar(u0)
    counts = zero(rate_cache)

    for i in 2:n
        tprev = tspan[1] + (i - 2) * dt
        t_new = tprev + dt
        rate(rate_cache, uprev, p, tprev)
        rate_cache .*= dt
        counts .= pois_rand.((rng,), rate_cache)
        c(du, uprev, p, tprev, counts, mark)
        u_new .= du .+ uprev

        # Save logic — only allocate (via copy) when actually saving
        if isempty(saveat_times)
            push!(usave, copy(u_new))
            push!(tsave, t_new)
        else
            while save_idx <= length(saveat_times) && t_new >= saveat_times[save_idx]
                push!(usave, copy(u_new))
                push!(tsave, saveat_times[save_idx])
                save_idx += 1
            end
        end

        uprev, u_new = u_new, uprev
    end

    # Save endpoint if requested and not already saved
    if save_end && (isempty(tsave) || tsave[end] != tspan[2])
        push!(usave, copy(uprev))
        push!(tsave, tspan[2])
    end

    sol = SciMLBase.build_solution(prob, alg, tsave, usave,
        calculate_error = false,
        interp = SciMLBase.ConstantInterpolation(tsave, usave))
end

# Compute the highest order of reaction (HOR) for each reaction j, as per Cao et al. (2006), Section IV.
# HOR is the sum of stoichiometric coefficients of reactants in reaction j.
# Extract the element type from reactant_stoch to avoid hardcoding type assumptions.
function compute_hor(reactant_stoch, numjumps)
    stoch_type = eltype(first(first(reactant_stoch)))
    hor = zeros(stoch_type, numjumps)
    for j in 1:numjumps
        order = sum(
            stoch for (spec_idx, stoch) in reactant_stoch[j]; init = zero(stoch_type))
        if order > 3
            error("Reaction $j has order $order, which is not supported (maximum order is 3).")
        end
        hor[j] = order
    end
    return hor
end

# Precompute reaction conditions for each species i, including:
# - max_hor: the highest order of reaction (HOR) where species i is a reactant.
# - max_stoich: the maximum stoichiometry (nu_ij) in reactions with max_hor.
# Used to optimize compute_gi, as per Cao et al. (2006), Section IV, equation (27).
function precompute_reaction_conditions(reactant_stoch, hor, numspecies, numjumps)
    hor_type = eltype(hor)
    max_hor = zeros(hor_type, numspecies)
    max_stoich = zeros(hor_type, numspecies)
    for j in 1:numjumps
        for (spec_idx, stoch) in reactant_stoch[j]
            if stoch > 0  # Species is a reactant
                if hor[j] > max_hor[spec_idx]
                    max_hor[spec_idx] = hor[j]
                    max_stoich[spec_idx] = stoch
                elseif hor[j] == max_hor[spec_idx]
                    max_stoich[spec_idx] = max(max_stoich[spec_idx], stoch)
                end
            end
        end
    end
    return max_hor, max_stoich
end

# Compute g_i for species i to bound the relative change in propensity functions,
# as per Cao et al. (2006), Section IV, equation (27).
# g_i is determined by the highest order of reaction (HOR) and maximum stoichiometry (nu_ij) where species i is a reactant:
# - HOR = 1 (first-order, e.g., S_i -> products): g_i = 1
# - HOR = 2 (second-order):
#   - nu_ij = 1 (e.g., S_i + S_k -> products): g_i = 2
#   - nu_ij = 2 (e.g., 2S_i -> products): g_i = 2 + 1/(x_i - 1)
# - HOR = 3 (third-order):
#   - nu_ij = 1 (e.g., S_i + S_k + S_m -> products): g_i = 3
#   - nu_ij = 2 (e.g., 2S_i + S_k -> products): g_i = (3/2) * (2 + 1/(x_i - 1))
#   - nu_ij = 3 (e.g., 3S_i -> products): g_i = 3 + 1/(x_i - 1) + 2/(x_i - 2)
# Uses precomputed max_hor and max_stoich to reduce work to O(num_species) per timestep.
function compute_gi(u, max_hor, max_stoich, i, t)
    one_max_hor = one(1 / one(eltype(u)))

    if max_hor[i] == 0  # No reactions involve species i as a reactant
        return one_max_hor
    elseif max_hor[i] == 1
        return one_max_hor
    elseif max_hor[i] == 2
        if max_stoich[i] == 1
            return 2 * one_max_hor
        else # if max_stoich[i] == 2
            return u[i] > one_max_hor ?
                   2 * one_max_hor + one_max_hor / (u[i] - one_max_hor) : 2 * one_max_hor  # Fallback to 2 if x_i <= 1
        end
    elseif max_hor[i] == 3
        if max_stoich[i] == 1
            return 3 * one_max_hor
        elseif max_stoich[i] == 2
            return u[i] > one_max_hor ?
                   (3 * one_max_hor / 2) *
                   (2 * one_max_hor + one_max_hor / (u[i] - one_max_hor)) : 3 * one_max_hor  # Fallback to 3 if x_i <= 1
        else # if max_stoich[i] == 3
            return u[i] > 2 * one_max_hor ?
                   3 * one_max_hor + one_max_hor / (u[i] - one_max_hor) +
                   2 * one_max_hor / (u[i] - 2 * one_max_hor) : 3 * one_max_hor  # Fallback to 3 if x_i <= 2
        end
    end
    return one_max_hor  # Default case
end

# Compute the tau-leaping step-size using equation (20) from Cao et al. (2006):
# tau = min_{i in I_rs} { max(epsilon * x_i / g_i, 1) / |mu_i(x)|, max(epsilon * x_i / g_i, 1)^2 / sigma_i^2(x) }
# where mu_i(x) and sigma_i^2(x) are defined in equations (9a) and (9b):
# mu_i(x) = sum_j nu_ij * a_j(x), sigma_i^2(x) = sum_j nu_ij^2 * a_j(x)
# I_rs is the set of reactant species (assumed to be all species here, as critical reactions are not specified).
function compute_tau(
        u, rate_cache, nu, hor, p, t, epsilon, rate, dtmin, max_hor, max_stoich, numjumps)
    rate(rate_cache, u, p, t)
    if all(<=(0), rate_cache)  # Handle case where all rates are zero or negative
        return dtmin
    end
    tau = typemax(typeof(t))
    for i in 1:length(u)
        mu = zero(eltype(u))
        sigma2 = zero(eltype(u))
        for j in 1:size(nu, 2)
            mu += nu[i, j] * rate_cache[j] # Equation (9a)
            sigma2 += nu[i, j]^2 * rate_cache[j] # Equation (9b)
        end
        gi = compute_gi(u, max_hor, max_stoich, i, t)
        bound = max(epsilon * u[i] / gi, one(eltype(u))) # max(epsilon * x_i / g_i, 1)
        mu_term = abs(mu) > 0 ? bound / abs(mu) : typemax(typeof(t)) # First term in equation (8)
        sigma_term = sigma2 > 0 ? bound^2 / sigma2 : typemax(typeof(t)) # Second term in equation (8)
        tau = min(tau, mu_term, sigma_term) # Equation (8)
    end
    return max(tau, dtmin)
end

# Function to generate a mass action rate function
function massaction_rate(maj, numjumps)
    return (out, u, p, t) -> begin
        for j in 1:numjumps
            out[j] = evalrxrate(u, j, maj)
        end
    end
end

function simple_explicit_tau_leaping_loop!(
        prob, alg, u_current, u_new, t_current, t_end, p, rng,
        rate, c, nu, hor, max_hor, max_stoich, numjumps, epsilon,
        dtmin, saveat_times, usave, tsave, du, counts, rate_cache, rate_effective, maj,
        save_end)
    save_idx = 1

    while t_current < t_end
        rate(rate_cache, u_current, p, t_current)
        if all(<=(0), rate_cache)  # No reactions can occur, step to final time
            t_current = t_end
            break
        end
        tau = compute_tau(u_current, rate_cache, nu, hor, p, t_current,
            epsilon, rate, dtmin, max_hor, max_stoich, numjumps)
        tau = min(tau, t_end - t_current)
        if !isempty(saveat_times) && save_idx <= length(saveat_times) &&
           t_current + tau > saveat_times[save_idx]
            tau = saveat_times[save_idx] - t_current
        end
        # Calculate Poisson random numbers only for positive rates
        rate_effective .= rate_cache .* tau
        for j in eachindex(counts)
            if rate_effective[j] <= zero(eltype(rate_effective))
                counts[j] = zero(eltype(counts))
            else
                counts[j] = pois_rand(rng, rate_effective[j])
            end
        end
        du .= 0
        if c !== nothing
            c(du, u_current, p, t_current, counts, nothing)
        else
            for j in 1:numjumps
                for (spec_idx, stoch) in maj.net_stoch[j]
                    du[spec_idx] += stoch * counts[j]
                end
            end
        end
        u_new .= u_current .+ du
        if any(<(0), u_new)
            # Halve tau to avoid negative populations, as per Cao et al. (2006), Section 3.3
            tau /= 2
            continue
        end
        t_new = t_current + tau

        # Save state if at a saveat time or if saveat is empty
        if isempty(saveat_times) ||
           (save_idx <= length(saveat_times) && t_new >= saveat_times[save_idx])
            push!(usave, copy(u_new))
            push!(tsave, t_new)
            if !isempty(saveat_times) && t_new >= saveat_times[save_idx]
                save_idx += 1
            end
        end

        u_current .= u_new
        t_current = t_new
    end

    # Save endpoint if requested and not already saved
    if save_end && (isempty(tsave) || tsave[end] != t_end)
        push!(usave, copy(u_current))
        push!(tsave, t_end)
    end
end

function DiffEqBase.solve(jump_prob::JumpProblem, alg::SimpleExplicitTauLeaping;
        seed = nothing,
        dtmin = nothing,
        saveat = nothing, save_start = nothing, save_end = nothing)
    validate_pure_leaping_inputs(jump_prob, alg) ||
        error("SimpleExplicitTauLeaping can only be used with PureLeaping JumpProblem with a MassActionJump.")

    prob = jump_prob.prob
    rng = jump_prob.rng
    tspan = prob.tspan

    if dtmin === nothing
        dtmin = 1e-10 * one(typeof(tspan[2]))
    end

    (seed !== nothing) && seed!(rng, seed)

    maj = jump_prob.massaction_jump
    numjumps = get_num_majumps(maj)
    rj = jump_prob.regular_jump
    # Extract rates
    rate = rj !== nothing ? rj.rate : massaction_rate(maj, numjumps)
    c = rj !== nothing ? rj.c : nothing
    u0 = copy(prob.u0)
    p = prob.p

    saveat_times, save_start, save_end = _process_saveat(saveat, tspan, save_start, save_end)

    # Initialize current state and saved history
    u_current = copy(u0)
    u_new = similar(u0)
    t_current = tspan[1]
    if save_start
        usave = [copy(u0)]
        tsave = [tspan[1]]
    else
        usave = typeof(u0)[]
        tsave = typeof(tspan[1])[]
    end
    rate_cache = zeros(float(eltype(u0)), numjumps)
    rate_effective = similar(rate_cache)
    counts = zero(rate_cache)
    du = similar(u0)
    t_end = tspan[2]
    epsilon = alg.epsilon

    # Extract net stoichiometry for state updates
    nu = zeros(float(eltype(u0)), length(u0), numjumps)
    for j in 1:numjumps
        for (spec_idx, stoch) in maj.net_stoch[j]
            nu[spec_idx, j] = stoch
        end
    end
    # Extract reactant stoichiometry for hor and gi
    reactant_stoch = maj.reactant_stoch
    hor = compute_hor(reactant_stoch, numjumps)
    max_hor, max_stoich = precompute_reaction_conditions(
        reactant_stoch, hor, length(u0), numjumps)

    simple_explicit_tau_leaping_loop!(
        prob, alg, u_current, u_new, t_current, t_end, p, rng,
        rate, c, nu, hor, max_hor, max_stoich, numjumps, epsilon,
        dtmin, saveat_times, usave, tsave, du, counts, rate_cache, rate_effective, maj,
        save_end)

    sol = SciMLBase.build_solution(prob, alg, tsave, usave,
        calculate_error = false,
        interp = SciMLBase.ConstantInterpolation(tsave, usave))
    return sol
end

# Residual of the implicit step, written so that the root u_new is the state at
# t + tau. `params` carries a preallocated propensity cache for each of the two
# states so that repeated residual evaluations do not allocate.
#
#   Newton:      u_new = u_current + sum_j nu_j a_j(u_new) tau
#   Trapezoidal: u_new = u_current + sum_j nu_j (a_j(u_current) + a_j(u_new))/2 tau
function implicit_equation!(resid, u_new, params)
    (; u_current, rate_new, rate_current, nu, p, t, tau, rate, numjumps, solver) = params

    rate(rate_new, u_new, p, t + tau)
    resid .= u_new .- u_current

    if solver isa NewtonImplicitSolver
        for j in 1:numjumps
            for spec_idx in axes(nu, 1)
                resid[spec_idx] -= nu[spec_idx, j] * rate_new[j] * tau
            end
        end
    else
        rate(rate_current, u_current, p, t)
        half = one(eltype(rate_new)) / 2
        for j in 1:numjumps
            for spec_idx in axes(nu, 1)
                resid[spec_idx] -= nu[spec_idx, j] * half *
                                   (rate_new[j] + rate_current[j]) * tau
            end
        end
    end
    nothing
end

# Solve one implicit step for the state at t + tau. Returns the new state and
# whether the nonlinear solve converged.
function solve_implicit(u_current, rate_new, rate_current, nu, p, t, tau, rate, numjumps,
        solver)
    u_guess = convert(Vector{float(eltype(u_current))}, u_current)
    params = (; u_current, rate_new, rate_current, nu, p, t, tau, rate, numjumps, solver)
    prob = NonlinearProblem(implicit_equation!, u_guess, params)
    sol = solve(prob, SimpleNewtonRaphson(autodiff = AutoFiniteDiff());
        abstol = 1e-6, reltol = 1e-6)
    return sol.u, SciMLBase.successful_retcode(sol)
end

function simple_implicit_tau_leaping_loop!(
        prob, alg, u_current, u_new, t_current, t_end, p, rng,
        rate, nu, hor, max_hor, max_stoich, numjumps, epsilon,
        dtmin, saveat_times, usave, tsave, du, counts, rate_cache, rate_current, maj,
        solver, save_end)
    save_idx = 1

    # Upper bound carried across iterations. Unlike the explicit loop, whose
    # retries redraw the Poisson counts, the implicit solve is deterministic in
    # (u_current, tau): retrying at an unchanged tau would fail identically, so a
    # rejected step has to shrink this bound to make progress.
    tau_cap = typemax(typeof(t_current))

    while t_current < t_end
        rate(rate_cache, u_current, p, t_current)
        if all(<=(0), rate_cache)  # No reactions can occur, step to final time
            t_current = t_end
            break
        end
        tau = compute_tau(u_current, rate_cache, nu, hor, p, t_current,
            epsilon, rate, dtmin, max_hor, max_stoich, numjumps)
        tau = min(tau, tau_cap, t_end - t_current)
        if !isempty(saveat_times) && save_idx <= length(saveat_times) &&
           t_current + tau > saveat_times[save_idx]
            tau = saveat_times[save_idx] - t_current
        end

        u_predicted, converged = solve_implicit(u_current, rate_cache, rate_current, nu, p,
            t_current, tau, rate, numjumps, solver)
        if !converged
            tau <= dtmin &&
                error("SimpleImplicitTauLeaping failed to converge at t = $t_current " *
                      "with the smallest permitted step dtmin = $dtmin.")
            tau_cap = tau / 2
            continue
        end

        # Sample the leap using the propensities at the implicitly predicted state.
        rate(rate_cache, u_predicted, p, t_current + tau)
        for j in eachindex(counts)
            scaled = rate_cache[j] * tau
            counts[j] = scaled <= zero(scaled) ? zero(eltype(counts)) :
                        pois_rand(rng, scaled)
        end

        du .= 0
        for j in 1:numjumps
            for (spec_idx, stoch) in maj.net_stoch[j]
                du[spec_idx] += stoch * counts[j]
            end
        end
        u_new .= u_current .+ du
        if any(<(0), u_new)
            # Halve tau to avoid negative populations, as per Cao et al. (2006), Section 3.3
            tau <= dtmin && break
            tau_cap = tau / 2
            continue
        end
        t_new = t_current + tau

        # Save state if at a saveat time or if saveat is empty
        if isempty(saveat_times) ||
           (save_idx <= length(saveat_times) && t_new >= saveat_times[save_idx])
            push!(usave, copy(u_new))
            push!(tsave, t_new)
            if !isempty(saveat_times) && t_new >= saveat_times[save_idx]
                save_idx += 1
            end
        end

        u_current .= u_new
        t_current = t_new
        tau_cap = typemax(typeof(t_current))  # release the bound after a good step
    end

    # Save endpoint if requested and not already saved
    if save_end && (isempty(tsave) || tsave[end] != t_end)
        push!(usave, copy(u_current))
        push!(tsave, t_end)
    end
end

function DiffEqBase.solve(jump_prob::JumpProblem, alg::SimpleImplicitTauLeaping;
        seed = nothing,
        dtmin = nothing,
        saveat = nothing, save_start = nothing, save_end = nothing)
    validate_pure_leaping_inputs(jump_prob, alg) ||
        error("SimpleImplicitTauLeaping can only be used with PureLeaping JumpProblem with a MassActionJump.")

    prob = jump_prob.prob
    rng = jump_prob.rng
    tspan = prob.tspan

    if dtmin === nothing
        dtmin = 1e-10 * one(typeof(tspan[2]))
    end

    (seed !== nothing) && seed!(rng, seed)

    maj = jump_prob.massaction_jump
    numjumps = get_num_majumps(maj)
    rate = massaction_rate(maj, numjumps)
    u0 = copy(prob.u0)
    p = prob.p

    saveat_times, save_start, save_end = _process_saveat(saveat, tspan, save_start, save_end)

    # Initialize current state and saved history
    u_current = copy(u0)
    u_new = similar(u0)
    t_current = tspan[1]
    if save_start
        usave = [copy(u0)]
        tsave = [tspan[1]]
    else
        usave = typeof(u0)[]
        tsave = typeof(tspan[1])[]
    end
    rate_cache = zeros(float(eltype(u0)), numjumps)
    rate_current = similar(rate_cache)
    counts = zero(rate_cache)
    du = similar(u0)
    t_end = tspan[2]
    epsilon = alg.epsilon
    solver = alg.solver

    # Extract net stoichiometry for state updates
    nu = zeros(float(eltype(u0)), length(u0), numjumps)
    for j in 1:numjumps
        for (spec_idx, stoch) in maj.net_stoch[j]
            nu[spec_idx, j] = stoch
        end
    end
    # Extract reactant stoichiometry for hor and gi
    reactant_stoch = maj.reactant_stoch
    hor = compute_hor(reactant_stoch, numjumps)
    max_hor, max_stoich = precompute_reaction_conditions(
        reactant_stoch, hor, length(u0), numjumps)

    simple_implicit_tau_leaping_loop!(
        prob, alg, u_current, u_new, t_current, t_end, p, rng,
        rate, nu, hor, max_hor, max_stoich, numjumps, epsilon,
        dtmin, saveat_times, usave, tsave, du, counts, rate_cache, rate_current, maj,
        solver, save_end)

    sol = SciMLBase.build_solution(prob, alg, tsave, usave,
        calculate_error = false,
        interp = SciMLBase.ConstantInterpolation(tsave, usave))
    return sol
end

"""
    EnsembleGPUKernel()
    EnsembleGPUKernel(backend)

Ensemble algorithm marker for GPU execution of tau-leaping ensemble simulations.

## Arguments

  - `backend`: Optional KernelAbstractions-compatible backend. `nothing` requests the
    default backend selected by the extension.

## Fields

  - `backend`: Backend object used by the GPU extension.
  - `cpu_offload`: Fraction of trajectories to offload to CPU execution.

## Returns

  - A `SciMLBase.EnsembleAlgorithm` value for use as the ensemble algorithm argument to
    `solve`.

## Examples

```julia
using JumpProcesses

ensemble_alg = EnsembleGPUKernel()
```
"""
struct EnsembleGPUKernel{Backend} <: SciMLBase.EnsembleAlgorithm
    backend::Backend
    cpu_offload::Float64
end

function EnsembleGPUKernel(backend)
    EnsembleGPUKernel(backend, 0.0)
end

function EnsembleGPUKernel()
    EnsembleGPUKernel(nothing, 0.0)
end
