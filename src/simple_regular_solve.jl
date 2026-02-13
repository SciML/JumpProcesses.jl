struct SimpleTauLeaping <: DiffEqBase.DEAlgorithm end

# Define solver type hierarchy
abstract type AbstractImplicitSolver end
struct NewtonImplicitSolver <: AbstractImplicitSolver end
struct TrapezoidalImplicitSolver <: AbstractImplicitSolver end

struct SimpleImplicitTauLeaping{T <: AbstractFloat} <: DiffEqBase.DEAlgorithm
    epsilon::T  # Error control parameter for tau selection
    solver::AbstractImplicitSolver  # Solver type: Newton or Trapezoidal
end

SimpleImplicitTauLeaping(; epsilon=0.05, solver=NewtonImplicitSolver()) = SimpleImplicitTauLeaping(epsilon, solver)

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

function validate_pure_leaping_inputs(jump_prob::JumpProblem, alg::SimpleImplicitTauLeaping)
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

function DiffEqBase.solve(jump_prob::JumpProblem, alg::SimpleTauLeaping;
        seed = nothing, dt = error("dt is required for SimpleTauLeaping."))
    validate_pure_leaping_inputs(jump_prob, alg) ||
        error("SimpleTauLeaping can only be used with PureLeaping JumpProblems with only RegularJumps.")

    (; prob, rng) = jump_prob
    (seed !== nothing) && Random.seed!(rng, seed)

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
    u = Vector{typeof(prob.u0)}(undef, n)
    u[1] = u0
    t = tspan[1]:dt:tspan[2]

    # iteration variables
    counts = zero(rate_cache) # counts for each variable

    for i in 2:n # iterate over dt-slices
        uprev = u[i - 1]
        tprev = t[i - 1]
        rate(rate_cache, uprev, p, tprev)
        rate_cache .*= dt # multiply by the width of the time interval
        counts .= pois_rand.((rng,), rate_cache) # set counts to the poisson arrivals with our given rates
        c(du, uprev, p, tprev, counts, mark)
        u[i] = du + uprev
    end

    sol = DiffEqBase.build_solution(prob, alg, t, u,
        calculate_error = false,
        interp = DiffEqBase.ConstantInterpolation(t, u))
end

function compute_hor(reactant_stoch, numjumps)
    # Compute the highest order of reaction (HOR) for each reaction j, as per Cao et al. (2006), Section IV.
    # HOR is the sum of stoichiometric coefficients of reactants in reaction j.
    # Extract the element type from reactant_stoch to avoid hardcoding type assumptions.
    stoch_type = eltype(first(first(reactant_stoch)))
    hor = zeros(stoch_type, numjumps)
    for j in 1:numjumps
        order = sum(stoch for (spec_idx, stoch) in reactant_stoch[j]; init=zero(stoch_type))
        if order > 3
            error("Reaction $j has order $order, which is not supported (maximum order is 3).")
        end
        hor[j] = order
    end
    return hor
end

function precompute_reaction_conditions(reactant_stoch, hor, numspecies, numjumps)
    # Precompute reaction conditions for each species i, including:
    # - max_hor: the highest order of reaction (HOR) where species i is a reactant.
    # - max_stoich: the maximum stoichiometry (nu_ij) in reactions with max_hor.
    # Used to optimize compute_gi, as per Cao et al. (2006), Section IV, equation (27).
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

function compute_gi(u, max_hor, max_stoich, i, t)
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
    one_max_hor = one(1 / one(eltype(u)))

    if max_hor[i] == 0  # No reactions involve species i as a reactant
        return one_max_hor
    elseif max_hor[i] == 1
        return one_max_hor
    elseif max_hor[i] == 2
        if max_stoich[i] == 1
            return 2 * one_max_hor
        else  # max_stoich[i] == 2
            return u[i] > one_max_hor ? 
                   2 * one_max_hor + one_max_hor / (u[i] - one_max_hor) : 2 * one_max_hor  # Fallback to 2 if x_i <= 1
        end
    elseif max_hor[i] == 3
        if max_stoich[i] == 1
            return 3 * one_max_hor
        elseif max_stoich[i] == 2
            return u[i] > one_max_hor ? 
                   (3 * one_max_hor / 2) * (2 * one_max_hor + one_max_hor / (u[i] - one_max_hor)) : 3 * one_max_hor  # Fallback to 3 if x_i <= 1
        else  # max_stoich[i] == 3
            return u[i] > 2 * one_max_hor ? 
                   3 * one_max_hor + one_max_hor / (u[i] - one_max_hor) + 2 * one_max_hor / (u[i] - 2 * one_max_hor) : 3 * one_max_hor  # Fallback to 3 if x_i <= 2
        end
    end
    return one_max_hor  # Default case
end

function compute_tau(u, rate_cache, nu, hor, p, t, epsilon, rate, dtmin, max_hor, max_stoich, numjumps)
    # Compute the tau-leaping step-size using equation (20) from Cao et al. (2006):
    # tau = min_{i in I_rs} { max(epsilon * x_i / g_i, 1) / |mu_i(x)|, max(epsilon * x_i / g_i, 1)^2 / sigma_i^2(x) }
    # where mu_i(x) and sigma_i^2(x) are defined in equations (9a) and (9b):
    # mu_i(x) = sum_j nu_ij * a_j(x), sigma_i^2(x) = sum_j nu_ij^2 * a_j(x)
    # I_rs is the set of reactant species (assumed to be all species here, as critical reactions are not specified).
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

# Define residual for implicit equation
# Newton: u_new = u_current + sum_j nu_j * a_j(u_new) * tau (Cao et al., 2004)
# Trapezoidal: u_new = u_current + sum_j nu_j * (a_j(u_current) + a_j(u_new))/2 * tau
function implicit_equation!(resid, u_new, params)
    u_current, rate_cache, nu, p, t, tau, rate, numjumps, solver = params
    rate(rate_cache, u_new, p, t + tau)
    resid .= u_new .- u_current
    for j in 1:numjumps
        for spec_idx in 1:size(nu, 1)
            if isa(solver, NewtonImplicitSolver)
                resid[spec_idx] -= nu[spec_idx, j] * rate_cache[j] * tau  # Cao et al. (2004)
            else  # TrapezoidalImplicitSolver
                rate_current = similar(rate_cache)
                rate(rate_current, u_current, p, t)
                resid[spec_idx] -= nu[spec_idx, j] * 0.5 * (rate_cache[j] + rate_current[j]) * tau
            end
        end
    end
    resid .= max.(resid, -u_new)  # Ensure non-negative solution
end

# Solve implicit equation using SimpleNonlinearSolve
function solve_implicit(u_current, rate_cache, nu, p, t, tau, rate, numjumps, solver)
    u_new = convert(Vector{Float64}, u_current)
    prob = NonlinearProblem(implicit_equation!, u_new, (u_current, rate_cache, nu, p, t, tau, rate, numjumps, solver))
    sol = solve(prob, SimpleNewtonRaphson(autodiff=AutoFiniteDiff()); abstol=1e-6, reltol=1e-6)
    return sol.u, sol.retcode == ReturnCode.Success
end

# Function to generate a mass action rate function
function massaction_rate(maj, numjumps)
    return (out, u, p, t) -> begin
        for j in 1:numjumps
            out[j] = evalrxrate(u, j, maj)
        end
    end
end

function simple_implicit_tau_leaping_loop!(
        prob, alg, u_current, t_current, t_end, p, rng,
        rate, nu, hor, max_hor, max_stoich, numjumps, epsilon,
        dtmin, saveat_times, usave, tsave, du, counts, rate_cache,
        maj, solver)
    save_idx = 1

    while t_current < t_end
        rate(rate_cache, u_current, p, t_current)
        tau = compute_tau(u_current, rate_cache, nu, hor, p, t_current, epsilon, rate, dtmin, max_hor, max_stoich, numjumps)
        tau = min(tau, t_end - t_current)
        if !isempty(saveat_times) && save_idx <= length(saveat_times) && t_current + tau > saveat_times[save_idx]
            tau = saveat_times[save_idx] - t_current
        end

        u_new_float, converged = solve_implicit(u_current, rate_cache, nu, p, t_current, tau, rate, numjumps, solver)
        if !converged
            tau /= 2
            continue
        end

        rate(rate_cache, u_new_float, p, t_current + tau)
        counts .= pois_rand.(rng, max.(rate_cache * tau, 0.0))
        du .= 0
        for j in 1:numjumps
            for (spec_idx, stoch) in maj.net_stoch[j]
                du[spec_idx] += stoch * counts[j]
            end
        end
        u_new = u_current + du

        if any(<(0), u_new)
            # Halve tau to avoid negative populations, as per Cao et al. (2006), Section 3.3
            tau /= 2
            continue
        end
        # Ensure non-negativity, as per Cao et al. (2006), Section 3.3
        for i in eachindex(u_new)
            u_new[i] = max(u_new[i], 0)
        end
        t_new = t_current + tau

        if isempty(saveat_times) || (save_idx <= length(saveat_times) && t_new >= saveat_times[save_idx])
            push!(usave, copy(u_new))
            push!(tsave, t_new)
            if !isempty(saveat_times) && t_new >= saveat_times[save_idx]
                save_idx += 1
            end
        end

        u_current = u_new
        t_current = t_new
    end
end

function DiffEqBase.solve(jump_prob::JumpProblem, alg::SimpleImplicitTauLeaping; 
        seed = nothing,
        dtmin = nothing,
        saveat = nothing)
    validate_pure_leaping_inputs(jump_prob, alg) ||
        error("SimpleImplicitTauLeaping can only be used with PureLeaping JumpProblem with a MassActionJump.")

    (; prob, rng) = jump_prob
    (seed !== nothing) && Random.seed!(rng, seed)

    maj = jump_prob.massaction_jump
    numjumps = get_num_majumps(maj)
    rj = jump_prob.regular_jump
    # Extract rates
    rate = rj !== nothing ? rj.rate : massaction_rate(maj, numjumps)
    c = rj !== nothing ? rj.c : nothing
    u0 = copy(prob.u0)
    tspan = prob.tspan
    p = prob.p

    if dtmin === nothing
        dtmin = 1e-10 * one(typeof(tspan[2]))
    end

    u_current = copy(u0)
    t_current = tspan[1]
    usave = [copy(u0)]
    tsave = [tspan[1]]
    rate_cache = zeros(float(eltype(u0)), numjumps)
    counts = zero(rate_cache)
    du = similar(u0)
    t_end = tspan[2]
    epsilon = alg.epsilon
    solver = alg.solver

    nu = zeros(float(eltype(u0)), length(u0), numjumps)
    for j in 1:numjumps
        for (spec_idx, stoch) in maj.net_stoch[j]
            nu[spec_idx, j] = stoch
        end
    end
    reactant_stoch = maj.reactant_stoch
    hor = compute_hor(reactant_stoch, numjumps)
    max_hor, max_stoich = precompute_reaction_conditions(reactant_stoch, hor, length(u0), numjumps)

    # Set up saveat_times
    if isnothing(saveat)
        saveat_times = Vector{typeof(tspan[1])}()
    elseif saveat isa Number
        saveat_times = collect(range(tspan[1], tspan[2], step=saveat))
    else
        saveat_times = collect(saveat)
    end
    simple_implicit_tau_leaping_loop!(
        prob, alg, u_current, t_current, t_end, p, rng,
        rate, nu, hor, max_hor, max_stoich, numjumps, epsilon,
        dtmin, saveat_times, usave, tsave, du, counts, rate_cache,
        maj, solver)

    sol = DiffEqBase.build_solution(prob, alg, tsave, usave,
        calculate_error=false,
        interp=DiffEqBase.ConstantInterpolation(tsave, usave))
    return sol
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
