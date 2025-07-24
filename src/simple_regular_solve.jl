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

# Define ImplicitTauLeaping algorithm
struct ImplicitTauLeaping <: DiffEqBase.DEAlgorithm
    epsilon::Float64  # Error control parameter
    nc::Int          # Critical reaction threshold
    nstiff::Int      # Stiffness threshold multiplier
    delta::Float64   # Partial equilibrium threshold
end

ImplicitTauLeaping(; epsilon=0.05, nc=10, nstiff=100, delta=0.05) = ImplicitTauLeaping(epsilon, nc, nstiff, delta)

function DiffEqBase.solve(jump_prob::JumpProblem, alg::ImplicitTauLeaping;
        seed = nothing,
        dt = error("dt is required for ImplicitTauLeaping."),
        kwargs...)

    # Boilerplate from SimpleTauLeaping
    @assert isempty(jump_prob.jump_callback.continuous_callbacks)
    @assert isempty(jump_prob.jump_callback.discrete_callbacks)
    prob = jump_prob.prob
    rng = DEFAULT_RNG
    (seed !== nothing) && seed!(rng, seed)

    rj = jump_prob.regular_jump
    rate = rj.rate  # rate(out, u, p, t)
    numjumps = rj.numjumps
    c = rj.c  # c(dc, u, p, t, counts, mark)
    reversible_pairs = get(kwargs, :reversible_pairs, Tuple{Int,Int}[])

    if !isnothing(rj.mark_dist)
        error("Mark distributions are currently not supported in ImplicitTauLeaping")
    end

    # Initialize state and buffers
    u0 = copy(prob.u0)
    p = prob.p
    tspan = prob.tspan
    state_dim = length(u0)
    dt = Float64(dt)

    # Compute stoichiometry matrix
    nu = zeros(Int, state_dim, numjumps)
    for j in 1:numjumps
        dc = zeros(state_dim)
        c(dc, u0, p, 0.0, [i == j ? 1 : 0 for i in 1:numjumps], nothing)
        nu[:, j] = dc
    end

    # Initialize solution arrays
    n = Int((tspan[2] - tspan[1]) / dt) + 1
    u = Vector{typeof(u0)}(undef, n)
    u[1] = u0
    t = range(tspan[1], tspan[2], length=n)

    # Buffers for iteration
    current_u = copy(u0)
    rate_cache = zeros(Float64, numjumps)
    counts = zeros(Float64, numjumps)
    local_dc = zeros(Float64, state_dim)
    I_rs = 1:state_dim
    g = ones(state_dim)  # Scaling factor for tau-leaping

    function compute_tau_explicit(u, rate, nu, num_jumps, epsilon, g, J_ncr, I_rs, p)
        rate_cache = zeros(eltype(u), num_jumps)
        rate(rate_cache, u, p, 0.0)
        
        mu = zeros(eltype(u), length(u))
        sigma2 = zeros(eltype(u), length(u))
        
        for i in I_rs
            mu[i] = sum(nu[i,j] * rate_cache[j] for j in J_ncr; init=0.0)
            sigma2[i] = sum(nu[i,j]^2 * rate_cache[j] for j in J_ncr; init=0.0)
        end
        
        tau = Inf
        for i in I_rs
            denom_mu = max(epsilon * u[i] / g[i], 1.0)
            denom_sigma = denom_mu^2
            if abs(mu[i]) > 0
                tau = min(tau, denom_mu / abs(mu[i]))
            end
            if sigma2[i] > 0
                tau = min(tau, denom_sigma / sigma2[i])
            end
        end
        return tau
    end

    function compute_tau_implicit(u, rate, nu, num_jumps, epsilon, g, J_necr, I_rs, p)
        rate_cache = zeros(eltype(u), num_jumps)
        rate(rate_cache, u, p, 0.0)
        
        mu = zeros(eltype(u), length(u))
        sigma2 = zeros(eltype(u), length(u))
        
        for i in I_rs
            mu[i] = sum(nu[i,j] * rate_cache[j] for j in J_necr; init=0.0)
            sigma2[i] = sum(nu[i,j]^2 * rate_cache[j] for j in J_necr; init=0.0)
        end
        
        tau = Inf
        for i in I_rs
            denom_mu = max(epsilon * u[i] / g[i], 1.0)
            denom_sigma = denom_mu^2
            if abs(mu[i]) > 0
                tau = min(tau, denom_mu / abs(mu[i]))
            end
            if sigma2[i] > 0
                tau = min(tau, denom_sigma / sigma2[i])
            end
        end
        return isinf(tau) ? 1e6 : tau
    end

    function identify_critical_reactions(u, nu, num_jumps, nc)
        L = zeros(Int, num_jumps)
        J_critical = Int[]
        
        for j in 1:num_jumps
            min_val = Inf
            for i in 1:length(u)
                if nu[i,j] < 0
                    val = floor(u[i] / abs(nu[i,j]))
                    min_val = min(min_val, val)
                end
            end
            L[j] = min_val == Inf ? typemax(Int) : Int(min_val)
            if L[j] < nc
                push!(J_critical, j)
            end
        end
        J_ncr = setdiff(1:num_jumps, J_critical)
        return J_critical, J_ncr
    end

    function check_partial_equilibrium(rate_cache, reversible_pairs, delta)
        J_equilibrium = Int[]
        for (j_plus, j_minus) in reversible_pairs
            a_plus = rate_cache[j_plus]
            a_minus = rate_cache[j_minus]
            if abs(a_plus - a_minus) <= delta * min(a_plus, a_minus)
                push!(J_equilibrium, j_plus, j_minus)
            end
        end
        return J_equilibrium
    end

    function newton_solve!(x_new, x, rate, nu, rate_cache, counts, p, t, tau, max_iter=10, tol=1e-6)
        state_dim = length(x)
        num_jumps = length(counts)
        
        for iter in 1:max_iter
            rate(rate_cache, x_new, p, t)
            rate_cache .*= tau
            
            residual = x_new .- x
            for j in 1:num_jumps
                residual .-= nu[:,j] * (counts[j] - rate_cache[j] + tau * rate_cache[j])
            end
            
            if norm(residual) < tol
                break
            end
            
            J = zeros(eltype(x), state_dim, state_dim)
            for j in 1:num_jumps
                for i in 1:state_dim
                    for k in 1:state_dim
                        J[i,k] += nu[i,j] * nu[k,j] * rate_cache[j]
                    end
                end
            end
            J = I - tau * J
            
            delta_x = J \ residual
            x_new .-= delta_x
            
            if norm(delta_x) < tol
                break
            end
        end
        return x_new
    end

    # Main solver loop
    for i in 2:n
        tprev = t[i - 1]
        J_critical, J_ncr = identify_critical_reactions(current_u, nu, numjumps, alg.nc)

        rate(rate_cache, current_u, p, tprev)
        a0_critical = sum(rate_cache[j] for j in J_critical; init=0.0)

        J_equilibrium = check_partial_equilibrium(rate_cache, reversible_pairs, alg.delta)
        J_necr = setdiff(J_ncr, J_equilibrium)

        tau_ex = compute_tau_explicit(current_u, rate, nu, numjumps, alg.epsilon, g, J_ncr, I_rs, p)
        tau_im = compute_tau_implicit(current_u, rate, nu, numjumps, alg.epsilon, g, J_necr, I_rs, p)

        tau2 = a0_critical > 0 ? -log(rand(rng)) / a0_critical : Inf
        use_implicit = tau_im > alg.nstiff * tau_ex
        tau1 = use_implicit ? tau_im : tau_ex

        if tau1 < 10 / sum(rate_cache; init=0.0)
            a0 = sum(rate_cache; init=0.0)
            if a0 > 0
                tau = -log(rand(rng)) / a0
                r = rand(rng) * a0
                cumsum_a = 0.0
                jc = 1
                for k in 1:numjumps
                    cumsum_a += rate_cache[k]
                    if cumsum_a > r
                        jc = k
                        break
                    end
                end
                current_u .+= nu[:,jc]
            else
                tau = dt
            end
        else
            tau = min(tau1, tau2, dt)
            if tau == tau2
                if a0_critical > 0
                    r = rand(rng) * a0_critical
                    cumsum_a = 0.0
                    jc = !isempty(J_critical) ? J_critical[1] : 1
                    for k in J_critical
                        cumsum_a += rate_cache[k]
                        if cumsum_a > r
                            jc = k
                            break
                        end
                    end
                    counts .= 0
                    counts[jc] = 1
                    if use_implicit && tau > tau_ex
                        for k in J_ncr
                            counts[k] = pois_rand(rng, rate_cache[k] * tau)
                        end
                        c(local_dc, current_u, p, tprev, counts, nothing)
                        current_u .= newton_solve!(current_u .+ local_dc, current_u, rate, nu, rate_cache, counts, p, tprev, tau)
                    else
                        for k in J_ncr
                            counts[k] = pois_rand(rng, rate_cache[k] * tau)
                        end
                        c(local_dc, current_u, p, tprev, counts, nothing)
                        current_u .+= local_dc
                    end
                else
                    tau = tau1
                    if use_implicit
                        for k in 1:numjumps
                            counts[k] = pois_rand(rng, rate_cache[k] * tau)
                        end
                        c(local_dc, current_u, p, tprev, counts, nothing)
                        current_u .= newton_solve!(current_u .+ local_dc, current_u, rate, nu, rate_cache, counts, p, tprev, tau)
                    else
                        for k in 1:numjumps
                            counts[k] = pois_rand(rng, rate_cache[k] * tau)
                        end
                        c(local_dc, current_u, p, tprev, counts, nothing)
                        current_u .+= local_dc
                    end
                end
            else
                counts .= 0
                if use_implicit
                    for k in J_ncr
                        counts[k] = pois_rand(rng, rate_cache[k] * tau)
                    end
                    c(local_dc, current_u, p, tprev, counts, nothing)
                    current_u .= newton_solve!(current_u .+ local_dc, current_u, rate, nu, rate_cache, counts, p, tprev, tau)
                else
                    for k in J_ncr
                        counts[k] = pois_rand(rng, rate_cache[k] * tau)
                    end
                    c(local_dc, current_u, p, tprev, counts, nothing)
                    current_u .+= local_dc
                end
            end
        end

        if any(current_u .< 0)
            tau1 /= 2
            continue
        end

        u[i] = copy(current_u)
    end

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

export SimpleTauLeaping, EnsembleGPUKernel, ImplicitTauLeaping
