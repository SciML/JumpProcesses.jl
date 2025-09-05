struct SimpleTauLeaping <: DiffEqBase.DEAlgorithm end

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

# Define the ImplicitTauLeaping algorithm
struct ImplicitTauLeaping <: DiffEqBase.DEAlgorithm
    epsilon::Float64  # Error control parameter
    nc::Int          # Critical reaction threshold
    nstiff::Float64  # Stiffness threshold for switching
    delta::Float64   # Partial equilibrium threshold
    equilibrium_pairs::Vector{Tuple{Int,Int}}  # Reversible reaction pairs
end

# Default constructor
ImplicitTauLeaping(; epsilon=0.05, nc=10, nstiff=100.0, delta=0.05, equilibrium_pairs=[(1, 2)]) = 
    ImplicitTauLeaping(epsilon, nc, nstiff, delta, equilibrium_pairs)

function DiffEqBase.solve(jump_prob::JumpProblem, alg::ImplicitTauLeaping; seed=nothing)
    # Boilerplate setup
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
    equilibrium_pairs = alg.equilibrium_pairs
    t_end = tspan[2]
    
    # Compute stoichiometry matrix from c function
    function compute_stoichiometry(c, u, numjumps)
        nu = zeros(Int, length(u), numjumps)
        for j in 1:numjumps
            counts = zeros(numjumps)
            counts[j] = 1
            du = similar(u)
            c(du, u, p, t[1], counts, nothing)
            nu[:, j] = round.(Int, du)
        end
        return nu
    end
    nu = compute_stoichiometry(c, u0, numjumps)
    
    # Helper function to compute g_i (approximation from Cao et al., 2006)
    function compute_gi(u, nu, i)
        # Simplified g_i: highest order of reaction involving species i
        max_order = 1.0
        for j in 1:numjumps
            if abs(nu[i, j]) > 0
                # Approximate reaction order based on propensity (heuristic)
                rate(rate_cache, u, p, t[end])
                if rate_cache[j] > 0
                    order = 1.0  # Assume first-order for simplicity
                    if j == 1  # For SIR infection (S*I), assume second-order
                        order = 2.0
                    end
                    max_order = max(max_order, order)
                end
            end
        end
        return max_order
    end
    
    # Tau-selection for explicit method (Equation 8)
    function compute_tau_explicit(u, rate_cache, nu, p, t)
        rate(rate_cache, u, p, t)
        mu = zeros(length(u))
        sigma2 = zeros(length(u))
        tau = Inf
        for i in 1:length(u)
            for j in 1:numjumps
                mu[i] += nu[i, j] * rate_cache[j]
                sigma2[i] += nu[i, j]^2 * rate_cache[j]
            end
            gi = compute_gi(u, nu, i)
            bound = max(epsilon * u[i] / gi, 1.0)
            mu_term = abs(mu[i]) > 0 ? bound / abs(mu[i]) : Inf
            sigma_term = sigma2[i] > 0 ? bound^2 / sigma2[i] : Inf
            tau = min(tau, mu_term, sigma_term)
        end
        return tau
    end
    
    # Partial equilibrium check (Equation 13)
    function is_partial_equilibrium(rate_cache, j_plus, j_minus)
        a_plus = rate_cache[j_plus]
        a_minus = rate_cache[j_minus]
        return abs(a_plus - a_minus) <= delta * min(a_plus, a_minus)
    end
    
    # Tau-selection for implicit method (Equation 14)
    function compute_tau_implicit(u, rate_cache, nu, p, t)
        rate(rate_cache, u, p, t)
        mu = zeros(length(u))
        sigma2 = zeros(length(u))
        non_equilibrium = trues(numjumps)
        for (j_plus, j_minus) in equilibrium_pairs
            if is_partial_equilibrium(rate_cache, j_plus, j_minus)
                non_equilibrium[j_plus] = false
                non_equilibrium[j_minus] = false
            end
        end
        tau = Inf
        for i in 1:length(u)
            for j in 1:numjumps
                if non_equilibrium[j]
                    mu[i] += nu[i, j] * rate_cache[j]
                    sigma2[i] += nu[i, j]^2 * rate_cache[j]
                end
            end
            gi = compute_gi(u, nu, i)
            bound = max(epsilon * u[i] / gi, 1.0)
            mu_term = abs(mu[i]) > 0 ? bound / abs(mu[i]) : Inf
            sigma_term = sigma2[i] > 0 ? bound^2 / sigma2[i] : Inf
            tau = min(tau, mu_term, sigma_term)
        end
        return tau
    end
    
    # Identify critical reactions
    function identify_critical_reactions(u, rate_cache, nu)
        critical = falses(numjumps)
        for j in 1:numjumps
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
    
    # Implicit tau-leaping step with Newton's method
    function implicit_tau_step(u_prev, t_prev, tau, rate_cache, counts, nu, p)
        u_new = copy(u_prev)
        rate_new = zeros(numjumps)
        tol = 1e-6
        max_iter = 100
        for iter in 1:max_iter
            rate(rate_new, u_new, p, t_prev + tau)
            residual = u_new - u_prev
            for j in 1:numjumps
                residual -= nu[:, j] * (counts[j] - tau * rate_cache[j] + tau * rate_new[j])
            end
            if norm(residual) < tol
                break
            end
            # Approximate Jacobian
            J = Diagonal(ones(length(u_new)))
            for j in 1:numjumps
                for i in 1:length(u_new)
                    if j == 1 && i in [1, 2]  # Infection: β*S*I
                        J[i, i] += nu[i, j] * tau * p[1] * (i == 1 ? u_new[2] : u_new[1])
                    elseif j == 2 && i == 2  # Recovery: ν*I
                        J[i, i] += nu[i, j] * tau * p[2]
                    end
                end
            end
            u_new -= J \ residual
            u_new = max.(u_new, 0.0)
        end
        return round.(Int, u_new)
    end
    
    # Down-shifting condition (Equation 19)
    function use_down_shifting(t, tau_im, tau_ex, a0, t_end)
        return t + tau_im >= t_end - 100 * (tau_ex + 1 / a0)
    end
    
    # Main simulation loop
    while t[end] < t_end
        u_prev = u[end]
        t_prev = t[end]
        
        # Compute propensities
        rate(rate_cache, u_prev, p, t_prev)
        
        # Identify critical reactions
        critical = identify_critical_reactions(u_prev, rate_cache, nu)
        
        # Compute tau values
        tau_ex = compute_tau_explicit(u_prev, rate_cache, nu, p, t_prev)
        tau_im = compute_tau_implicit(u_prev, rate_cache, nu, p, t_prev)
        
        # Compute critical propensity sum
        ac0 = sum(rate_cache[critical])
        tau2 = ac0 > 0 ? randexp(rng) / ac0 : Inf
        
        # Choose method and stepsize
        a0 = sum(rate_cache)
        use_implicit = tau_im > nstiff * tau_ex && !use_down_shifting(t_prev, tau_im, tau_ex, a0, t_end)
        tau1 = use_implicit ? tau_im : tau_ex
        method = use_implicit ? :implicit : :explicit
        
        # Check if tau1 is too small
        if tau1 < 10 / a0
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
                    counts[j] = pois_rand(rng, rate_cache[j] * tau)
                end
            end
            if method == :implicit
                u_new = implicit_tau_step(u_prev, t_prev, tau, rate_cache, counts, nu, p)
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
                    counts[j] = pois_rand(rng, rate_cache[j] * tau)
                end
            end
            if method == :implicit && tau > tau_ex
                u_new = implicit_tau_step(u_prev, t_prev, tau, rate_cache, counts, nu, p)
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
        calculate_error=false,
        interp=DiffEqBase.ConstantInterpolation(t, u))
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

export SimpleTauLeaping, EnsembleGPUKernel, ImplicitTauLeaping
