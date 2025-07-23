function SciMLBase.__solve(ensembleprob::SciMLBase.AbstractEnsembleProblem, 
        alg::ImplicitTauLeaping,
        ensemblealg::EnsembleGPUKernel;
        trajectories,
        seed = nothing,
        dt = error("dt is required for ImplicitTauLeaping."),
        kwargs...)

    if trajectories == 1
        return SciMLBase.__solve(ensembleprob, alg, EnsembleSerial(); trajectories=1,
            seed, dt, kwargs...)
    end

    backend = ensemblealg.backend === nothing ? CPU() : ensemblealg.backend

    jump_prob = ensembleprob.prob

    @assert isempty(jump_prob.jump_callback.continuous_callbacks)
    @assert isempty(jump_prob.jump_callback.discrete_callbacks)
    prob = jump_prob.prob
    
    probs = [remake(jump_prob) for _ in 1:trajectories]

    ts, us = vectorized_solve(probs, jump_prob, alg; backend, trajectories, seed, dt, kwargs...)

    _ts = Array(ts)
    _us = Array(us)

    time = @elapsed sol = [begin
        ts = @view _ts[:, i]
        us = @view _us[:, :, i]
        sol_idx = findlast(x -> x != probs[i].prob.tspan[1], ts)
        if sol_idx === nothing
            @error "No solution found" tspan=probs[i].tspan[1] ts
            error("Batch solve failed")
        end
        @views ensembleprob.output_func(
            SciMLBase.build_solution(probs[i].prob,
                alg,
                ts[1:sol_idx],
                [us[j, :] for j in 1:sol_idx],
                k = nothing,
                stats = nothing,
                calculate_error = false,
                retcode = sol_idx != length(ts) ? ReturnCode.Terminated : ReturnCode.Success),
            i)[1]
    end for i in eachindex(probs)]
    return SciMLBase.EnsembleSolution(sol, time, true)
end

struct TrajectoryDataImplicit{U <: StaticArray, P, T}
    u0::U
    p::P
    tspan::Tuple{T, T}
end

struct JumpDataImplicit{R, C, V}
    rate::R
    c::C
    nu::V
    numjumps::Int
end

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
    
    x_temp = copy(x_new)
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

@kernel function implicit_tau_leaping_kernel(@Const(probs_data), _us, _ts, dt, @Const(rj_data),
                                            current_u_buf, rate_cache_buf, counts_buf, local_dc_buf,
                                            seed::UInt64, alg::ImplicitTauLeaping, reversible_pairs)
    i = @index(Global, Linear)

    @inbounds begin
        current_u = view(current_u_buf, :, i)
        rate_cache = view(rate_cache_buf, :, i)
        counts = view(counts_buf, :, i)
        local_dc = view(local_dc_buf, :, i)
    end

    @inbounds prob_data = probs_data[i]
    u0 = prob_data.u0
    p = prob_data.p
    tspan = prob_data.tspan

    rate = rj_data.rate
    num_jumps = rj_data.numjumps
    c = rj_data.c
    nu = rj_data.nu

    @inbounds for k in 1:length(u0)
        current_u[k] = u0[k]
    end

    n = Int((tspan[2] - tspan[1]) / dt) + 1
    state_dim = length(u0)

    ts_view = @inbounds view(_ts, :, i)
    us_view = @inbounds view(_us, :, :, i)

    @inbounds ts_view[1] = tspan[1]
    @inbounds for k in 1:state_dim
        us_view[1, k] = current_u[k]
    end

    rng = Random.Xoshiro(seed + i)

    I_rs = 1:state_dim
    g = ones(state_dim)

    for j in 2:n
        tprev = tspan[1] + (j-2) * dt

        J_critical, J_ncr = identify_critical_reactions(current_u, nu, num_jumps, alg.nc)

        rate(rate_cache, current_u, p, tprev)
        a0_critical = sum(rate_cache[j] for j in J_critical; init=0.0)

        J_equilibrium = check_partial_equilibrium(rate_cache, reversible_pairs, alg.delta)
        J_necr = setdiff(J_ncr, J_equilibrium)

        tau_ex = compute_tau_explicit(current_u, rate, nu, num_jumps, alg.epsilon, g, J_ncr, I_rs, p)
        tau_im = compute_tau_implicit(current_u, rate, nu, num_jumps, alg.epsilon, g, J_necr, I_rs, p)

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
                for k in 1:num_jumps
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
                    jc = J_critical[1]
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
                            counts[k] = poisson_rand(rate_cache[k] * tau, rng)
                        end
                        c(local_dc, current_u, p, tprev, counts, nothing)
                        current_u .= newton_solve!(current_u .+ local_dc, current_u, rate, nu, rate_cache, counts, p, tprev, tau)
                    else
                        for k in J_ncr
                            counts[k] = poisson_rand(rate_cache[k] * tau, rng)
                        end
                        c(local_dc, current_u, p, tprev, counts, nothing)
                        current_u .+= local_dc
                    end
                else
                    tau = tau1
                    if use_implicit
                        for k in 1:num_jumps
                            counts[k] = poisson_rand(rate_cache[k] * tau, rng)
                        end
                        c(local_dc, current_u, p, tprev, counts, nothing)
                        current_u .= newton_solve!(current_u .+ local_dc, current_u, rate, nu, rate_cache, counts, p, tprev, tau)
                    else
                        for k in 1:num_jumps
                            counts[k] = poisson_rand(rate_cache[k] * tau, rng)
                        end
                        c(local_dc, current_u, p, tprev, counts, nothing)
                        current_u .+= local_dc
                    end
                end
            else
                counts .= 0
                if use_implicit
                    for k in J_ncr
                        counts[k] = poisson_rand(rate_cache[k] * tau, rng)
                    end
                    c(local_dc, current_u, p, tprev, counts, nothing)
                    current_u .= newton_solve!(current_u .+ local_dc, current_u, rate, nu, rate_cache, counts, p, tprev, tau)
                else
                    for k in J_ncr
                        counts[k] = poisson_rand(rate_cache[k] * tau, rng)
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

        @inbounds for k in 1:state_dim
            us_view[j, k] = current_u[k]
        end
        @inbounds ts_view[j] = tspan[1] + (j-1) * dt
    end
end

function vectorized_solve(probs, prob::JumpProblem, alg::ImplicitTauLeaping; backend, trajectories, seed, dt, kwargs...)
    rj = prob.regular_jump
    nu = zeros(Int, length(prob.prob.u0), rj.numjumps)
    for j in 1:rj.numjumps
        dc = zeros(length(prob.prob.u0))
        rj.c(dc, prob.prob.u0, prob.prob.p, 0.0, [i == j ? 1 : 0 for i in 1:rj.numjumps], nothing)
        nu[:,j] = dc
    end
    rj_data = JumpDataImplicit(rj.rate, rj.c, nu, rj.numjumps)

    probs_data = [TrajectoryDataImplicit(SA{eltype(p.prob.u0)}[p.prob.u0...], p.prob.p, p.prob.tspan) for p in probs]

    probs_data_gpu = adapt(backend, probs_data)
    rj_data_gpu = adapt(backend, rj_data)

    state_dim = length(first(probs_data).u0)
    tspan = prob.prob.tspan
    dt = Float64(dt)
    n_steps = Int((tspan[2] - tspan[1]) / dt) + 1
    n_trajectories = length(probs)
    num_jumps = rj_data.numjumps

    @assert state_dim > 0 "Dimension of state must be positive"
    @assert num_jumps >= 0 "Number of jumps must be positive"

    ts = allocate(backend, eltype(prob.prob.tspan), (n_steps, n_trajectories))
    us = allocate(backend, eltype(prob.prob.u0), (n_steps, state_dim, n_trajectories))

    current_u_buf = allocate(backend, eltype(prob.prob.u0), (state_dim, n_trajectories))
    rate_cache_buf = allocate(backend, eltype(prob.prob.u0), (num_jumps, n_trajectories))
    counts_buf = allocate(backend, eltype(prob.prob.u0), (num_jumps, n_trajectories))
    local_dc_buf = allocate(backend, eltype(prob.prob.u0), (state_dim, n_trajectories))

    @kernel function init_buffers_kernel(@Const(probs_data), current_u_buf)
        i = @index(Global, Linear)
        @inbounds u0 = probs_data[i].u0
        @inbounds for k in 1:length(u0)
            current_u_buf[k, i] = u0[k]
        end
    end
    init_kernel = init_buffers_kernel(backend)
    init_event = init_kernel(probs_data_gpu, current_u_buf; ndrange=n_trajectories)
    KernelAbstractions.synchronize(backend)

    seed = seed === nothing ? UInt64(12345) : UInt64(seed)
    reversible_pairs = get(kwargs, :reversible_pairs, Tuple{Int,Int}[])

    kernel = implicit_tau_leaping_kernel(backend)
    main_event = kernel(probs_data_gpu, us, ts, dt, rj_data_gpu,
                        current_u_buf, rate_cache_buf, counts_buf, local_dc_buf, seed, alg, reversible_pairs;
                        ndrange=n_trajectories)
    KernelAbstractions.synchronize(backend)

    return ts, us
end

@inline function poisson_rand(lambda, rng)
    L = exp(-lambda)
    k = 0
    p = 1.0
    while p > L
        k += 1
        p *= rand(rng)
    end
    return k - 1
end
