struct SimpleTauLeaping <: DiffEqBase.AbstractDEAlgorithm end

struct SimpleExplicitTauLeaping{T <: AbstractFloat} <: DiffEqBase.AbstractDEAlgorithm
    epsilon::T  # Error control parameter
end

SimpleExplicitTauLeaping(; epsilon = 0.05) = SimpleExplicitTauLeaping(epsilon)

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

function validate_pure_leaping_inputs(jump_prob::JumpProblem, alg::SimpleExplicitTauLeaping)
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

    sol = DiffEqBase.build_solution(prob, alg, tsave, usave,
        calculate_error = false,
        interp = DiffEqBase.ConstantInterpolation(tsave, usave))
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

    sol = DiffEqBase.build_solution(prob, alg, tsave, usave,
        calculate_error = false,
        interp = DiffEqBase.ConstantInterpolation(tsave, usave))
    return sol
end

struct SimpleAdaptiveTauLeaping <: DiffEqBase.DEAlgorithm
    epsilon::Float64  # Error control parameter
end

SimpleAdaptiveTauLeaping(; epsilon=0.05) = SimpleAdaptiveTauLeaping(epsilon)

function DiffEqBase.solve(jump_prob::JumpProblem, alg::SimpleAdaptiveTauLeaping; seed=nothing)
    @assert isempty(jump_prob.jump_callback.continuous_callbacks)
    @assert isempty(jump_prob.jump_callback.discrete_callbacks)
    prob = jump_prob.prob
    rng = DEFAULT_RNG
    (seed !== nothing) && seed!(rng, seed)

    rj = jump_prob.regular_jump
    rate = rj.rate
    numjumps = rj.numjumps
    c = rj.c
    u0 = copy(prob.u0)
    tspan = prob.tspan
    p = prob.p

    u = [copy(u0)]
    t = [tspan[1]]
    rate_cache = zeros(Float64, numjumps)
    counts = zeros(Int, numjumps)
    du = similar(u0)
    t_end = tspan[2]
    epsilon = alg.epsilon

    nu = compute_stoichiometry(c, u0, numjumps, p, t[1])

    while t[end] < t_end
        u_prev = u[end]
        t_prev = t[end]
        rate(rate_cache, u_prev, p, t_prev)
        tau = compute_tau_explicit(u_prev, rate_cache, nu, p, t_prev, epsilon, rate)
        tau = min(tau, t_end - t_prev)
        counts .= pois_rand.(rng, max.(rate_cache * tau, 0.0))
        c(du, u_prev, p, t_prev, counts, nothing)
        u_new = u_prev + du
        if any(u_new .< 0)
            tau /= 2
            continue
        end
        push!(u, u_new)
        push!(t, t_prev + tau)
    end

    sol = DiffEqBase.build_solution(prob, alg, t, u,
        calculate_error=false,
        interp=DiffEqBase.ConstantInterpolation(t, u))
    return sol
end

struct SimpleImplicitTauLeaping <: DiffEqBase.DEAlgorithm
    epsilon::Float64  # Error control parameter
    nc::Int          # Critical reaction threshold
    nstiff::Float64  # Stiffness threshold for switching
    delta::Float64   # Partial equilibrium threshold
end

SimpleImplicitTauLeaping(; epsilon=0.05, nc=10, nstiff=100.0, delta=0.05) = 
    SimpleImplicitTauLeaping(epsilon, nc, nstiff, delta)

# Compute stoichiometry matrix from c function
function compute_stoichiometry(c, u, numjumps, p, t)
    nu = zeros(Int, length(u), numjumps)
    for j in 1:numjumps
        counts = zeros(numjumps)
        counts[j] = 1
        du = similar(u)
        c(du, u, p, t, counts, nothing)
        nu[:, j] = round.(Int, du)
    end
    return nu
end

# Detect reversible reaction pairs
function find_reversible_pairs(nu)
    pairs = Vector{Tuple{Int,Int}}()
    for j in 1:size(nu, 2)
        for k in (j+1):size(nu, 2)
            if nu[:, j] == -nu[:, k]
                push!(pairs, (j, k))
            end
        end
    end
    return pairs
end

# Compute g_i (approximation from Cao et al., 2006)
function compute_gi(u, nu, i, rate, rate_cache, p, t)
    max_order = 1.0
    for j in 1:size(nu, 2)
        if abs(nu[i, j]) > 0
            rate(rate_cache, u, p, t)
            if rate_cache[j] > 0
                order = 1.0
                if sum(abs.(nu[:, j])) > abs(nu[i, j])
                    order = 2.0
                end
                max_order = max(max_order, order)
            end
        end
    end
    return max_order
end

# Tau-selection for explicit method (Equation 8)
function compute_tau_explicit(u, rate_cache, nu, p, t, epsilon, rate)
    rate(rate_cache, u, p, t)
    mu = zeros(length(u))
    sigma2 = zeros(length(u))
    tau = Inf
    for i in 1:length(u)
        for j in 1:size(nu, 2)
            mu[i] += nu[i, j] * rate_cache[j]
            sigma2[i] += nu[i, j]^2 * rate_cache[j]
        end
        gi = compute_gi(u, nu, i, rate, rate_cache, p, t)
        bound = max(epsilon * u[i] / gi, 1.0)
        mu_term = abs(mu[i]) > 0 ? bound / abs(mu[i]) : Inf
        sigma_term = sigma2[i] > 0 ? bound^2 / sigma2[i] : Inf
        tau = min(tau, mu_term, sigma_term)
    end
    return max(tau, 1e-10)
end

# Partial equilibrium check (Equation 13)
function is_partial_equilibrium(rate_cache, j_plus, j_minus, delta)
    a_plus = rate_cache[j_plus]
    a_minus = rate_cache[j_minus]
    return abs(a_plus - a_minus) <= delta * min(a_plus, a_minus)
end

# Tau-selection for implicit method (Equation 14)
function compute_tau_implicit(u, rate_cache, nu, p, t, epsilon, rate, equilibrium_pairs, delta)
    rate(rate_cache, u, p, t)
    mu = zeros(length(u))
    sigma2 = zeros(length(u))
    non_equilibrium = trues(size(nu, 2))
    for (j_plus, j_minus) in equilibrium_pairs
        if is_partial_equilibrium(rate_cache, j_plus, j_minus, delta)
            non_equilibrium[j_plus] = false
            non_equilibrium[j_minus] = false
        end
    end
    tau = Inf
    for i in 1:length(u)
        for j in 1:size(nu, 2)
            if non_equilibrium[j]
                mu[i] += nu[i, j] * rate_cache[j]
                sigma2[i] += nu[i, j]^2 * rate_cache[j]
            end
        end
        gi = compute_gi(u, nu, i, rate, rate_cache, p, t)
        bound = max(epsilon * u[i] / gi, 1.0)
        mu_term = abs(mu[i]) > 0 ? bound / abs(mu[i]) : Inf
        sigma_term = sigma2[i] > 0 ? bound^2 / sigma2[i] : Inf
        tau = min(tau, mu_term, sigma_term)
    end
    return max(tau, 1e-10)
end

# Identify critical reactions
function identify_critical_reactions(u, rate_cache, nu, nc)
    critical = falses(size(nu, 2))
    for j in 1:size(nu, 2)
        if rate_cache[j] > 0
            Lj = Inf
            for i in 1:length(u)
                if nu[i, j] < 0
                    Lj = min(Lj, floor(Int, u[i] / abs(nu[i, j])))
                end
            end
            if Lj < nc
                critical[j] = true
            end
        end
    end
    return critical
end

# Implicit tau-leaping step using NonlinearSolve
function implicit_tau_step(u_prev, t_prev, tau, rate_cache, counts, nu, p, rate, numjumps)
    # Define the nonlinear system: F(u_new) = u_new - u_prev - sum(nu_j * (counts_j - tau * a_j(u_prev) + tau * a_j(u_new))) = 0
    function f(u_new, params)
        rate_new = zeros(eltype(u_new), numjumps)
        rate(rate_new, u_new, p, t_prev + tau)
        residual = u_new - u_prev
        for j in 1:numjumps
            residual -= nu[:, j] * (counts[j] - tau * rate_cache[j] + tau * rate_new[j])
        end
        return residual
    end
    
    # Initial guess
    u_new = copy(u_prev)
    
    # Solve the nonlinear system
    prob = NonlinearProblem(f, u_new, nothing)
    sol = solve(prob, NewtonRaphson())
    
    # Check for convergence and numerical stability
    if sol.retcode != ReturnCode.Success || any(isnan.(sol.u)) || any(isinf.(sol.u))
        return round.(Int, max.(u_prev, 0.0))  # Revert to previous state
    end
    
    return round.(Int, max.(sol.u, 0.0))
end

# Down-shifting condition (Equation 19)
function use_down_shifting(t, tau_im, tau_ex, a0, t_end)
    return a0 > 0 && t + tau_im >= t_end - 100 * (tau_ex + 1 / a0)
end

function DiffEqBase.solve(jump_prob::JumpProblem, alg::SimpleImplicitTauLeaping; seed=nothing)
    @assert isempty(jump_prob.jump_callback.continuous_callbacks)
    @assert isempty(jump_prob.jump_callback.discrete_callbacks)
    prob = jump_prob.prob
    rng = DEFAULT_RNG
    (seed !== nothing) && seed!(rng, seed)

    rj = jump_prob.regular_jump
    rate = rj.rate
    numjumps = rj.numjumps
    c = rj.c
    u0 = copy(prob.u0)
    tspan = prob.tspan
    p = prob.p
    
    # Initialize storage
    rate_cache = zeros(Float64, numjumps)
    counts = zeros(Int, numjumps)
    du = similar(u0)
    u = [copy(u0)]
    t = [tspan[1]]
    
    # Algorithm parameters
    epsilon = alg.epsilon
    nc = alg.nc
    nstiff = alg.nstiff
    delta = alg.delta
    t_end = tspan[2]
    
    # Compute stoichiometry matrix
    nu = compute_stoichiometry(c, u0, numjumps, p, t[1])
    
    # Detect reversible reaction pairs
    equilibrium_pairs = find_reversible_pairs(nu)
    
    # Main simulation loop
    while t[end] < t_end
        u_prev = u[end]
        t_prev = t[end]
        
        # Compute propensities
        rate(rate_cache, u_prev, p, t_prev)
        
        # Identify critical reactions
        critical = identify_critical_reactions(u_prev, rate_cache, nu, nc)
        
        # Compute tau values
        tau_ex = compute_tau_explicit(u_prev, rate_cache, nu, p, t_prev, epsilon, rate)
        tau_im = compute_tau_implicit(u_prev, rate_cache, nu, p, t_prev, epsilon, rate, equilibrium_pairs, delta)
        
        # Compute critical propensity sum
        ac0 = sum(rate_cache[critical])
        tau2 = ac0 > 0 ? randexp(rng) / ac0 : Inf
        
        # Choose method and stepsize
        a0 = sum(rate_cache)
        use_implicit = a0 > 0 && tau_im > nstiff * tau_ex && !use_down_shifting(t_prev, tau_im, tau_ex, a0, t_end)
        tau1 = use_implicit ? tau_im : tau_ex
        method = use_implicit ? :implicit : :explicit
        
        # Cap tau to prevent large updates
        tau1 = min(tau1, 1.0)
        
        # Check if tau1 is too small
        if a0 > 0 && tau1 < 10 / a0
            # Use SSA for a few steps
            steps = method == :implicit ? 10 : 100
            for _ in 1:steps
                if t_prev >= t_end
                    break
                end
                rate(rate_cache, u_prev, p, t_prev)
                a0 = sum(rate_cache)
                if a0 == 0
                    break
                end
                tau = randexp(rng) / a0
                r = rand(rng) * a0
                cumsum_rate = 0.0
                for j in 1:numjumps
                    cumsum_rate += rate_cache[j]
                    if cumsum_rate > r
                        u_prev += nu[:, j]
                        break
                    end
                end
                t_prev += tau
                push!(u, copy(u_prev))
                push!(t, t_prev)
            end
            continue
        end
        
        # Choose stepsize and compute firings
        if tau2 > tau1
            tau = min(tau1, t_end - t_prev)
            counts .= 0
            for j in 1:numjumps
                if !critical[j]
                    counts[j] = pois_rand(rng, max(rate_cache[j] * tau, 0.0))
                end
            end
            if method == :implicit
                u_new = implicit_tau_step(u_prev, t_prev, tau, rate_cache, counts, nu, p, rate, numjumps)
            else
                c(du, u_prev, p, t_prev, counts, nothing)
                u_new = u_prev + du
            end
        else
            tau = min(tau2, t_end - t_prev)
            counts .= 0
            if ac0 > 0
                r = rand(rng) * ac0
                cumsum_rate = 0.0
                for j in 1:numjumps
                    if critical[j]
                        cumsum_rate += rate_cache[j]
                        if cumsum_rate > r
                            counts[j] = 1
                            break
                        end
                    end
                end
            end
            for j in 1:numjumps
                if !critical[j]
                    counts[j] = pois_rand(rng, max(rate_cache[j] * tau, 0.0))
                end
            end
            if method == :implicit && tau > tau_ex
                u_new = implicit_tau_step(u_prev, t_prev, tau, rate_cache, counts, nu, p, rate, numjumps)
            else
                c(du, u_prev, p, t_prev, counts, nothing)
                u_new = u_prev + du
            end
        end
        
        # Check for negative populations
        if any(u_new .< 0)
            tau1 /= 2
            continue
        end
        
        # Update state and time
        push!(u, u_new)
        push!(t, t_prev + tau)
    end
    
    # Build solution
    sol = DiffEqBase.build_solution(prob, alg, t, u,
        calculate_error = false,
        interp = DiffEqBase.ConstantInterpolation(t, u))
end

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

export SimpleTauLeaping, EnsembleGPUKernel, SimpleAdaptiveTauLeaping, SimpleImplicitTauLeaping
