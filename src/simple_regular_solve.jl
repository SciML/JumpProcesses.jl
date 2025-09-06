struct SimpleTauLeaping <: DiffEqBase.DEAlgorithm end

# Define solver type hierarchy
abstract type AbstractImplicitSolver end
struct NewtonImplicitSolver <: AbstractImplicitSolver end
struct TrapezoidalImplicitSolver <: AbstractImplicitSolver end

# Adaptive tau-leaping solver
struct SimpleAdaptiveTauLeaping{T <: AbstractFloat} <: DiffEqBase.DEAlgorithm
    epsilon::T  # Error control parameter for tau selection
    solver::AbstractImplicitSolver  # Solver type for implicit method
    eigenvalue_check::Bool  # Enable eigenvalue-based stiffness detection
    stiffness_ratio_threshold::T # # Stiffness ratio threshold
    implicit_epsilon_factor::T  # Scaling factor for implicit tau-selection
end

# Stiffness detection uses a dynamic threshold epsilon * sum(u) for propensity ratios,
# as inspired by Cao et al. (2007), Section III.B. Optional eigenvalue-based check
# uses the Jacobian's eigenvalue ratio. implicit_epsilon_factor=10.0 relaxes tau-selection
# for implicit tau-leaping, per Cao et al. (2007), Section III.A.
SimpleAdaptiveTauLeaping(; epsilon=0.05, solver=NewtonImplicitSolver(), eigenvalue_check=false, stiffness_ratio_threshold=1e4, implicit_epsilon_factor=10.0) = 
    SimpleAdaptiveTauLeaping(epsilon, solver, eigenvalue_check, stiffness_ratio_threshold, implicit_epsilon_factor)

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

# Compute highest order of reaction (HOR)
# Reference: Cao et al. (2006), J. Chem. Phys. 124, 044109, Section IV
function compute_hor(reactant_stoch, numjumps)
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

# Precompute max_hor and max_stoich for g_i calculation
# Reference: Cao et al. (2006), Section IV, equation (27)
function precompute_reaction_conditions(reactant_stoch, hor, numspecies, numjumps)
    max_hor = zeros(Int, numspecies)
    max_stoich = zeros(Int, numspecies)
    for j in 1:numjumps
        for (spec_idx, stoch) in reactant_stoch[j]
            if stoch > 0
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

# Compute g_i to bound propensity changes
# Reference: Cao et al. (2006), Section IV, equation (27)
function compute_gi(u, max_hor, max_stoich, i, t)
    if max_hor[i] == 0
        return 1.0
    elseif max_hor[i] == 1
        return 1.0
    elseif max_hor[i] == 2
        if max_stoich[i] == 1
            return 2.0
        elseif max_stoich[i] == 2
            return u[i] > 1 ? 2.0 + 1.0 / (u[i] - 1) : 2.0
        end
    elseif max_hor[i] == 3
        if max_stoich[i] == 1
            return 3.0
        elseif max_stoich[i] == 2
            return u[i] > 1 ? 1.5 * (2.0 + 1.0 / (u[i] - 1)) : 3.0
        elseif max_stoich[i] == 3
            return u[i] > 2 ? 3.0 + 1.0 / (u[i] - 1) + 2.0 / (u[i] - 2) : 3.0
        end
    end
    return 1.0
end

# Compute tau for explicit tau-leaping
# Reference: Cao et al. (2006), equation (8), using equations (9a) and (9b)
function compute_tau(u, rate_cache, nu, hor, p, t, epsilon, rate, dtmin, max_hor, max_stoich, numjumps)
    rate(rate_cache, u, p, t)
    if all(==(0.0), rate_cache)
        return dtmin
    end
    tau = Inf
    for i in 1:length(u)
        mu = zero(eltype(rate_cache))  # Equation (9a): mu_i(x) = sum_j nu_ij * a_j(x)
        sigma2 = zero(eltype(rate_cache))  # Equation (9b): sigma_i^2(x) = sum_j nu_ij^2 * a_j(x)
        for j in 1:size(nu, 2)
            mu += nu[i, j] * rate_cache[j]
            sigma2 += nu[i, j]^2 * rate_cache[j]
        end
        gi = compute_gi(u, max_hor, max_stoich, i, t)  # Equation (27)
        bound = max(epsilon * max(u[i], 0.0) / gi, 1.0)
        mu_term = abs(mu) > 0 ? bound / abs(mu) : Inf  # First term in equation (8)
        sigma_term = sigma2 > 0 ? bound^2 / sigma2 : Inf  # Second term in equation (8)
        tau = min(tau, mu_term, sigma_term)  # Equation (8)
    end
    return max(tau, dtmin)
end

# Compute tau for implicit tau-leaping with relaxed error control
# Reference: Cao et al. (2007), J. Chem. Phys. 126, 224101, Section III.A, using relaxed epsilon for larger steps
function compute_tau_implicit(u, rate_cache, nu, hor, p, t, epsilon, rate, dtmin, max_hor, max_stoich, numjumps, implicit_epsilon_factor)
    tau_explicit = compute_tau(u, rate_cache, nu, hor, p, t, epsilon, rate, dtmin, max_hor, max_stoich, numjumps)
    u_predict = float.(copy(u))  # Initialize as Float64 to handle fractional updates
    rate(rate_cache, u, p, t)
    for j in 1:numjumps
        for spec_idx in 1:size(nu, 1)
            u_predict[spec_idx] += nu[spec_idx, j] * rate_cache[j] * tau_explicit
        end
    end
    u_predict = max.(u_predict, 0.0)
    
    relaxed_epsilon = epsilon * implicit_epsilon_factor
    # Reuse compute_tau with predicted state, time, and relaxed epsilon
    tau = compute_tau(u_predict, rate_cache, nu, hor, p, t + tau_explicit, relaxed_epsilon, rate, dtmin, max_hor, max_stoich, numjumps)
    return max(tau, dtmin)
end

# Define residual for implicit equation
# Reference: Cao et al. (2004), J. Chem. Phys. 121, 4059
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

# Solve implicit equation
function solve_implicit(u_current, rate_cache, nu, p, t, tau, rate, numjumps, solver)
    u_new = float.(copy(u_current))
    prob = NonlinearProblem(implicit_equation!, u_new, (u_current, rate_cache, nu, p, t, tau, rate, numjumps, solver))
    sol = solve(prob, SimpleNewtonRaphson(autodiff=AutoFiniteDiff()); abstol=1e-6, reltol=1e-6)
    return sol.u, sol.retcode == ReturnCode.Success
end

# Compute Jacobian for eigenvalue-based stiffness detection
# Reference: Cao et al. (2007), Section III.B
function compute_jacobian(u, rate, numjumps, numspecies, p, t)
    J = zeros(numjumps, numspecies)
    rate_cache = zeros(numjumps)
    rate(rate_cache, u, p, t)
    h = 1e-6
    for i in 1:numspecies
        u_plus = float.(copy(u))
        u_plus[i] += h
        rate_plus = zeros(numjumps)
        rate(rate_plus, u_plus, p, t)
        for j in 1:numjumps
            J[j, i] = (rate_plus[j] - rate_cache[j]) / h
        end
    end
    return J
end

# Stiffness detection using propensity ratio or eigenvalues
# Reference: Cao et al. (2007), Section III.B
function is_stiff(rate_cache, u, epsilon, eigenvalue_check, stiffness_ratio_threshold, p, t, rate, numjumps, numspecies)
    non_zero_rates = [rate for rate in rate_cache if rate > 0]
    if length(non_zero_rates) <= 1
        return false
    end
    if eigenvalue_check
        J = compute_jacobian(u, rate, numjumps, numspecies, p, t)
        eigvals = real.(LinearAlgebra.eigvals(J))
        non_zero_eigvals = [abs(λ) for λ in eigvals if abs(λ) > 1e-10]
        if length(non_zero_eigvals) <= 1
            return false
        end
        max_eig = maximum(non_zero_eigvals)
        min_eig = minimum(non_zero_eigvals)
        return max_eig / min_eig > stiffness_ratio_threshold  # Stiffness ratio threshold, Petzold (1983), SIAM J. Sci. Stat. Comput. 4(1), 136–148
    else
        max_rate = maximum(non_zero_rates)
        min_rate = minimum(non_zero_rates)
        threshold = epsilon * sum(u)
        return max_rate / min_rate > threshold  # Propensity ratio threshold, Cao et al. (2007), J. Chem. Phys. 126, 224101, Section III.B
    end
end

# Adaptive tau-leaping solver
# Reference: Cao et al. (2007), Cao et al. (2004), Cao et al. (2006)
function DiffEqBase.solve(jump_prob::JumpProblem, alg::SimpleAdaptiveTauLeaping; seed=nothing, dtmin=1e-10, saveat=nothing)
    validate_pure_leaping_inputs(jump_prob, alg) || error("SimpleAdaptiveTauLeaping can only be used with PureLeaping JumpProblem with a MassActionJump.")

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
    eigenvalue_check = alg.eigenvalue_check
    stiffness_ratio_threshold = alg.stiffness_ratio_threshold
    implicit_epsilon_factor = alg.implicit_epsilon_factor

    nu = zeros(Int64, length(u0), numjumps)
    for j in 1:numjumps
        for (spec_idx, stoch) in maj.net_stoch[j]
            nu[spec_idx, j] = stoch
        end
    end
    reactant_stoch = maj.reactant_stoch
    hor = compute_hor(reactant_stoch, numjumps)
    max_hor, max_stoich = precompute_reaction_conditions(reactant_stoch, hor, length(u0), numjumps)
    numspecies = length(u0)

    saveat_times = isnothing(saveat) ? Vector{Float64}() : 
                   (saveat isa Number ? collect(range(tspan[1], tspan[2], step=saveat)) : collect(saveat))
    save_idx = 1

    while t_current < t_end
        rate(rate_cache, u_current, p, t_current)
        use_implicit = is_stiff(rate_cache, u_current, epsilon, eigenvalue_check, stiffness_ratio_threshold, p, t_current, rate, numjumps, numspecies)
        tau = use_implicit ? 
              compute_tau_implicit(u_current, rate_cache, nu, hor, p, t_current, epsilon, rate, dtmin, max_hor, max_stoich, numjumps, implicit_epsilon_factor) :
              compute_tau(u_current, rate_cache, nu, hor, p, t_current, epsilon, rate, dtmin, max_hor, max_stoich, numjumps)
        tau = min(tau, t_end - t_current)
        if !isempty(saveat_times) && save_idx <= length(saveat_times) && t_current + tau > saveat_times[save_idx]
            tau = saveat_times[save_idx] - t_current
        end

        if use_implicit
            u_new_float, converged = solve_implicit(u_current, rate_cache, nu, p, t_current, tau, rate, numjumps, solver)
            if !converged
                tau /= 2
                continue
            end
            rate(rate_cache, u_new_float, p, t_current + tau)
            counts .= pois_rand.(rng, max.(rate_cache * tau, 0.0))  # Cao et al. (2004)
            du .= zero(eltype(u_current))
            for j in 1:numjumps
                for spec_idx in 1:size(nu, 1)
                    if nu[spec_idx, j] != 0
                        du[spec_idx] += nu[spec_idx, j] * counts[j]
                    end
                end
            end
            u_new = u_current + du  # Cao et al. (2004)
        else
            counts .= pois_rand.(rng, max.(rate_cache * tau, 0.0))  # Cao et al. (2006), equation (8)
            du .= zero(eltype(u_current))
            for j in 1:numjumps
                for spec_idx in 1:size(nu, 1)
                    if nu[spec_idx, j] != 0
                        du[spec_idx] += nu[spec_idx, j] * counts[j]
                    end
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
            push!(usave, copy(u_new))  # Ensure integer solutions
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
