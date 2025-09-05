struct SimpleTauLeaping <: DiffEqBase.DEAlgorithm end

# Define solver type hierarchy
abstract type AbstractImplicitSolver end
struct NewtonImplicitSolver <: AbstractImplicitSolver end
struct TrapezoidalImplicitSolver <: AbstractImplicitSolver end

# Adaptive tau-leaping solver
struct SimpleAdaptiveTauLeaping{T <: AbstractFloat} <: DiffEqBase.DEAlgorithm
    epsilon::T  # Error control parameter for tau selection
    solver::AbstractImplicitSolver  # Solver type for implicit method
end

# Stiffness detection threshold is computed dynamically as epsilon * sum(u), where u is the current state,
# as inspired by Cao et al. (2007), Section III.B, which uses the ratio of propensity functions to identify
# fast and slow time scales in stiff systems, scaled by system size for robustness.
SimpleAdaptiveTauLeaping(; epsilon=0.05, solver=NewtonImplicitSolver()) = 
    SimpleAdaptiveTauLeaping(epsilon, solver)

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

# Validation for adaptive tau-leaping
function validate_pure_leaping_inputs(jump_prob::JumpProblem, alg::SimpleAdaptiveTauLeaping)
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
        error("SimpleTauLeaping can only be used with PureLeaping JumpProblems with only non-RegularJumps.")
    
    @unpack prob, rng = jump_prob
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
    hor = zeros(Int, numjumps)
    for j in 1:numjumps
        order = sum(stoch for (spec_idx, stoch) in reactant_stoch[j]; init=0)
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
    max_hor = zeros(Int, numspecies)
    max_stoich = zeros(Int, numspecies)
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
    if max_hor[i] == 0  # No reactions involve species i as a reactant
        return 1.0
    elseif max_hor[i] == 1
        return 1.0
    elseif max_hor[i] == 2
        if max_stoich[i] == 1
            return 2.0
        elseif max_stoich[i] == 2
            return u[i] > 1 ? 2.0 + 1.0 / (u[i] - 1) : 2.0  # Fallback to 2.0 if x_i <= 1
        end
    elseif max_hor[i] == 3
        if max_stoich[i] == 1
            return 3.0
        elseif max_stoich[i] == 2
            return u[i] > 1 ? 1.5 * (2.0 + 1.0 / (u[i] - 1)) : 3.0  # Fallback to 3.0 if x_i <= 1
        elseif max_stoich[i] == 3
            return u[i] > 2 ? 3.0 + 1.0 / (u[i] - 1) + 2.0 / (u[i] - 2) : 3.0  # Fallback to 3.0 if x_i <= 2
        end
    end
    return 1.0  # Default case
end

function compute_tau(u, rate_cache, nu, hor, p, t, epsilon, rate, dtmin, max_hor, max_stoich, numjumps)
    # Compute the tau-leaping step-size using equation (20) from Cao et al. (2006):
    # tau = min_{i in I_rs} { max(epsilon * x_i / g_i, 1) / |mu_i(x)|, max(epsilon * x_i / g_i, 1)^2 / sigma_i^2(x) }
    # where mu_i(x) and sigma_i^2(x) are defined in equations (9a) and (9b):
    # mu_i(x) = sum_j nu_ij * a_j(x), sigma_i^2(x) = sum_j nu_ij^2 * a_j(x)
    # I_rs is the set of reactant species (assumed to be all species here, as critical reactions are not specified).
    rate(rate_cache, u, p, t)
    if all(==(0.0), rate_cache)  # Handle case where all rates are zero
        return dtmin
    end
    tau = Inf
    for i in 1:length(u)
        mu = zero(eltype(u))
        sigma2 = zero(eltype(u))
        for j in 1:size(nu, 2)
            mu += nu[i, j] * rate_cache[j] # Equation (9a)
            sigma2 += nu[i, j]^2 * rate_cache[j] # Equation (9b)
        end
        gi = compute_gi(u, max_hor, max_stoich, i, t)
        bound = max(epsilon * u[i] / gi, 1.0) # max(epsilon * x_i / g_i, 1)
        mu_term = abs(mu) > 0 ? bound / abs(mu) : Inf # First term in equation (8)
        sigma_term = sigma2 > 0 ? bound^2 / sigma2 : Inf # Second term in equation (8)
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

# Stiffness detection based on propensity ratio scaled by system size
# Reference: Cao et al. (2007), Section III.B, using ratio of propensity functions to detect stiffness
function is_stiff(rate_cache, u, epsilon)
    non_zero_rates = [rate for rate in rate_cache if rate > 0]
    if length(non_zero_rates) <= 1
        return false  # Use explicit method if no or only one non-zero propensity
    end
    max_rate = maximum(non_zero_rates)
    min_rate = minimum(non_zero_rates)
    threshold = epsilon * sum(u)  # Dynamic threshold based on system size
    return max_rate / min_rate > threshold
end

# Adaptive tau-leaping solver
# Reference: Cao et al. (2007) for adaptive strategy, Cao et al. (2004) for implicit method, Cao et al. (2006) for tau selection
function DiffEqBase.solve(jump_prob::JumpProblem, alg::SimpleAdaptiveTauLeaping; 
        seed = nothing,
        dtmin = 1e-10,
        saveat = nothing)
    validate_pure_leaping_inputs(jump_prob, alg) ||
        error("SimpleAdaptiveTauLeaping can only be used with PureLeaping JumpProblem with a MassActionJump.")

    @unpack prob, rng = jump_prob
    (seed !== nothing) && seed!(rng, seed)

    maj = jump_prob.massaction_jump
    numjumps = get_num_majumps(maj)
    rate = (out, u, p, t) -> begin
        for j in 1:numjumps
            out[j] = evalrxrate(u, j, maj)
        end
    end
    u0 = copy(prob.u0)
    tspan = prob.tspan
    p = prob.p

    u_current = copy(u0)
    t_current = tspan[1]
    usave = [copy(u0)]
    tsave = [tspan[1]]
    rate_cache = zeros(Float64, numjumps)
    counts = zeros(Int64, numjumps)
    du = similar(u0)
    t_end = tspan[2]
    epsilon = alg.epsilon
    solver = alg.solver

    nu = zeros(Int64, length(u0), numjumps)
    for j in 1:numjumps
        for (spec_idx, stoch) in maj.net_stoch[j]
            nu[spec_idx, j] = stoch
        end
    end
    reactant_stoch = maj.reactant_stoch
    hor = compute_hor(reactant_stoch, numjumps)
    max_hor, max_stoich = precompute_reaction_conditions(reactant_stoch, hor, length(u0), numjumps)

    saveat_times = isnothing(saveat) ? Vector{Float64}() : 
                   (saveat isa Number ? collect(range(tspan[1], tspan[2], step=saveat)) : collect(saveat))
    save_idx = 1

    while t_current < t_end
        rate(rate_cache, u_current, p, t_current)
        tau = compute_tau(u_current, rate_cache, nu, hor, p, t_current, epsilon, rate, dtmin, max_hor, max_stoich, numjumps)
        tau = min(tau, t_end - t_current)
        if !isempty(saveat_times) && save_idx <= length(saveat_times) && t_current + tau > saveat_times[save_idx]
            tau = saveat_times[save_idx] - t_current
        end

        # Stiffness detection using dynamic propensity ratio threshold
        use_implicit = is_stiff(rate_cache, u_current, epsilon)

        if use_implicit
            # Implicit tau-leaping (Cao et al., 2004)
            u_new_float, converged = solve_implicit(u_current, rate_cache, nu, p, t_current, tau, rate, numjumps, solver)
            if !converged
                tau /= 2
                continue
            end
            rate(rate_cache, u_new_float, p, t_current + tau)
            counts .= pois_rand.(rng, max.(rate_cache * tau, 0.0))  # Cao et al. (2004): Poisson sampling
            du .= zero(eltype(u_current))
            for j in 1:numjumps
                for (spec_idx, stoch) in maj.net_stoch[j]
                    du[spec_idx] += stoch * counts[j]
                end
            end
            u_new = u_current + du  # Cao et al. (2004): Final state update
        else
            # Explicit tau-leaping (Cao et al., 2006)
            counts .= pois_rand.(rng, max.(rate_cache * tau, 0.0))  # Equation (8): Poisson sampling
            du .= zero(eltype(u_current))
            for j in 1:numjumps
                for (spec_idx, stoch) in maj.net_stoch[j]
                    du[spec_idx] += stoch * counts[j]
                end
            end
            u_new = u_current + du
        end

        if any(<(0), u_new)
            tau /= 2
            continue
        end
        t_new = t_current + tau

        if isempty(saveat_times) || (save_idx <= length(saveat_times) && t_new >= saveat_times[save_idx])
            push!(usave, copy(ceil.(u_new)))  # Ensure integer solutions for molecular populations
            push!(tsave, t_new)
            if !isempty(saveat_times) && t_new >= saveat_times[save_idx]
                save_idx += 1
            end
        end

        u_current = u_new
        t_current = t_new
    end

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
