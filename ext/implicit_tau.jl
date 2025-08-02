# Ensemble solver for ImplicitTauLeaping
function SciMLBase.__solve(ensembleprob::SciMLBase.AbstractEnsembleProblem, 
        alg::ImplicitTauLeaping,
        ensemblealg::EnsembleGPUKernel;
        trajectories,
        seed = nothing,
        max_steps = 10000,
        kwargs...)

    if trajectories == 1
        return SciMLBase.__solve(ensembleprob, alg, EnsembleSerial(); trajectories = 1,
            seed, kwargs...)
    end

    ensemblealg.backend === nothing ? backend = CPU() : 
    backend = ensemblealg.backend

    jump_prob = ensembleprob.prob
    
    probs = [remake(jump_prob) for _ in 1:trajectories]
    # Debug: Verify p in probs
    for i in 1:trajectories
        @assert typeof(probs[i].prob.p) == NTuple{4, Float64} "p in probs[$i] must be NTuple{4, Float64}, got $(typeof(probs[i].prob.p)), p = $(probs[i].prob.p)"
    end

    ts, us = vectorized_solve(probs, jump_prob, alg; backend, trajectories, seed, max_steps)

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
                                        retcode = sol_idx !=
                                                length(ts) ?
                                                ReturnCode.Terminated :
                                                ReturnCode.Success),
                                    i)[1]
                            end
                            for i in eachindex(probs)]
    return SciMLBase.EnsembleSolution(sol, time, true)
end

# Structs for trajectory and jump data
struct ImplicitTauLeapingTrajectoryData{U <: StaticArray, P, T}
    u0::U
    p::P
    tspan::Tuple{T, T}
end

struct ImplicitTauLeapingJumpData{R, C, N}
    rate::R
    c::C
    numjumps::Int
    nu::N
end

struct ImplicitTauLeapingData
    epsilon::Float64
    nc::Int
    nstiff::Float64
    delta::Float64
end

# ImplicitTauLeaping kernel
@kernel function implicit_tau_leaping_kernel(@Const(probs_data), _us, _ts, @Const(rj_data), @Const(alg_data),
                                            current_u_buf, rate_cache_buf, counts_buf, local_dc_buf,
                                            mu_buf, sigma2_buf, critical_buf, rate_new_buf, residual_buf, J_buf,
                                            seed::UInt64, max_steps)
    i = @index(Global, Linear)

    # Thread-local buffers
    @inbounds begin
        current_u = view(current_u_buf, :, i)
        rate_cache = view(rate_cache_buf, :, i)
        counts = view(counts_buf, :, i)
        local_dc = view(local_dc_buf, :, i)
        mu = view(mu_buf, :, i)
        sigma2 = view(sigma2_buf, :, i)
        critical = view(critical_buf, :, i)
        rate_new = view(rate_new_buf, :, i)
        residual = view(residual_buf, :, i)
        J = view(J_buf, :, :, i)
    end

    # Problem data
    @inbounds prob_data = probs_data[i]
    u0 = prob_data.u0
    p = prob_data.p
    tspan = prob_data.tspan
    t_end = tspan[2]

    # Jump data
    rate = rj_data.rate
    num_jumps = rj_data.numjumps
    c = rj_data.c
    nu = rj_data.nu

    # Algorithm parameters
    epsilon = alg_data.epsilon
    nc = alg_data.nc
    nstiff = alg_data.nstiff
    delta = alg_data.delta

    # Initialize state
    @inbounds for k in 1:length(u0)
        current_u[k] = u0[k]
    end
    state_dim = length(u0)

    # Output arrays
    ts_view = @inbounds view(_ts, :, i)
    us_view = @inbounds view(_us, :, :, i)
    @inbounds ts_view[1] = tspan[1]
    @inbounds for k in 1:state_dim
        us_view[1, k] = current_u[k]
    end

    # Debug: Check parameter type
    if i == 1
        @assert typeof(p) == NTuple{4, Float64} "p must be a tuple of 4 Float64 values, got $(typeof(p)), p = $p"
        @show p
        @show typeof(rate)
    end

    # Find reversible pairs
    equilibrium_pairs = Tuple{Int,Int}[]
    for j in 1:num_jumps
        for k in (j+1):num_jumps
            if all(nu[l, j] == -nu[l, k] for l in 1:state_dim)
                push!(equilibrium_pairs, (j, k))
            end
        end
    end

    # Helper functions
    function compute_gi(u, t)
        max_order = 1.0
        for j in 1:num_jumps
            if any(abs.(nu[:, j]) .> 0)
                # Debug: Check p and rate before calling
                if i == 1
                    @show p
                    @show typeof(rate)
                end
                rate(rate_cache, u, p, t)
                if rate_cache[j] > 0
                    order = sum(abs.(nu[:, j])) > abs(nu[findfirst(abs.(nu[:, j]) .> 0), j]) ? 2.0 : 1.0
                    max_order = max(max_order, order)
                end
            end
        end
        max_order
    end

    function compute_tau_explicit(u, t)
        # Debug: Check p and rate before calling
        if i == 1
            @show p
            @show typeof(rate)
        end
        rate(rate_cache, u, p, t)
        mu .= 0.0
        sigma2 .= 0.0
        tau = Inf
        for l in 1:state_dim
            for j in 1:num_jumps
                mu[l] += nu[l, j] * rate_cache[j]
                sigma2[l] += nu[l, j]^2 * rate_cache[j]
            end
            gi = compute_gi(u, t)
            bound = max(epsilon * u[l] / gi, 1.0)
            mu_term = abs(mu[l]) > 0 ? bound / abs(mu[l]) : Inf
            sigma_term = sigma2[l] > 0 ? bound^2 / sigma2[l] : Inf
            tau = min(tau, mu_term, sigma_term)
        end
        tau
    end

    function is_partial_equilibrium(rate_cache, j_plus, j_minus)
        a_plus = rate_cache[j_plus]
        a_minus = rate_cache[j_minus]
        abs(a_plus - a_minus) <= delta * min(a_plus, a_minus)
    end

    function compute_tau_implicit(u, t)
        # Debug: Check p and rate before calling
        if i == 1
            @show p
            @show typeof(rate)
        end
        rate(rate_cache, u, p, t)
        mu .= 0.0
        sigma2 .= 0.0
        non_equilibrium = trues(num_jumps)
        for (j_plus, j_minus) in equilibrium_pairs
            if is_partial_equilibrium(rate_cache, j_plus, j_minus)
                non_equilibrium[j_plus] = false
                non_equilibrium[j_minus] = false
            end
        end
        tau = Inf
        for l in 1:state_dim
            for j in 1:num_jumps
                if non_equilibrium[j]
                    mu[l] += nu[l, j] * rate_cache[j]
                    sigma2[l] += nu[l, j]^2 * rate_cache[j]
                end
            end
            gi = compute_gi(u, t)
            bound = max(epsilon * u[l] / gi, 1.0)
            mu_term = abs(mu[l]) > 0 ? bound / abs(mu[l]) : Inf
            sigma_term = sigma2[l] > 0 ? bound^2 / sigma2[l] : Inf
            tau = min(tau, mu_term, sigma_term)
        end
        tau
    end

    function identify_critical_reactions(u)
        critical .= false
        for j in 1:num_jumps
            if rate_cache[j] > 0
                Lj = Inf
                for l in 1:state_dim
                    if nu[l, j] < 0
                        Lj = min(Lj, floor(Int, u[l] / abs(nu[l, j])))
                    end
                end
                if Lj < nc
                    critical[j] = true
                end
            end
        end
    end

    function implicit_tau_step(u_prev, t_prev, tau, rate_cache, counts, u_new)
        u_new .= u_prev
        tol = 1e-6
        max_iter = 100
        for iter in 1:max_iter
            # Debug: Check p and rate before calling
            if i == 1
                @show p
                @show typeof(rate)
            end
            rate(rate_new, u_new, p, t_prev + tau)
            residual .= u_new .- u_prev
            for j in 1:num_jumps
                for l in 1:state_dim
                    residual[l] -= nu[l, j] * (counts[j] - tau * rate_cache[j] + tau * rate_new[j])
                end
            end
            if norm(residual) < tol
                break
            end
            J .= 0.0
            for l in 1:state_dim
                J[l, l] = 1.0
                for j in 1:num_jumps
                    if rate_new[j] > 0
                        J[l, l] += nu[l, j] * tau * rate_new[j] / max(u_new[l], 1.0)
                    end
                end
            end
            u_new .-= J \ residual
            u_new .= max.(u_new, 0.0)
        end
        u_new .= round.(Int, u_new)
        # Debug: Check p and c before calling
        if i == 1
            @assert typeof(p) == NTuple{4, Float64} "p must be a tuple of 4 Float64 values before c, got $(typeof(p)), p = $p"
            @show p
            @show typeof(c)
        end
        c(local_dc, u_new, p, t_prev + tau, counts, nothing)
        u_new .+= local_dc
    end

    function use_down_shifting(t, tau_im, tau_ex, a0)
        t + tau_im >= t_end - 100 * (tau_ex + 1 / a0)
    end

    # Thread-local RNG
    local rng_state = seed ⊻ UInt64(i)
    function thread_rand()
        rng_state = (1103515245 * rng_state + 12345) & 0x7fffffff
        rng_state / 0x7fffffff
    end
    function thread_randexp()
        -log(thread_rand())
    end
    function thread_poisson(lambda)
        L = exp(-lambda)
        k = 0
        p = 1.0
        while p > L
            k += 1
            p *= thread_rand()
        end
        k - 1
    end

    # Main simulation loop
    step = 1
    t = tspan[1]
    while t < t_end && step < max_steps
        step += 1
        # Debug: Check p and rate before calling
        if i == 1
            @show p
            @show typeof(rate)
        end
        rate(rate_cache, current_u, p, t)
        identify_critical_reactions(current_u)
        tau_ex = compute_tau_explicit(current_u, t)
        tau_im = compute_tau_implicit(current_u, t)
        ac0 = sum(rate_cache[critical])
        tau2 = ac0 > 0 ? thread_randexp() / ac0 : Inf
        a0 = sum(rate_cache)
        use_implicit = a0 > 0 && tau_im > nstiff * tau_ex && !use_down_shifting(t, tau_im, tau_ex, a0)
        tau1 = use_implicit ? tau_im : tau_ex

        if a0 > 0 && tau1 < 10 / a0
            steps = use_implicit ? 10 : 100
            for _ in 1:steps
                if t >= t_end
                    break
                end
                rate(rate_cache, current_u, p, t)
                a0 = sum(rate_cache)
                if a0 == 0
                    break
                end
                tau = thread_randexp() / a0
                r = thread_rand() * a0
                cumsum_rate = 0.0
                for j in 1:num_jumps
                    cumsum_rate += rate_cache[j]
                    if cumsum_rate > r
                        current_u .+= nu[:, j]
                        break
                    end
                end
                t += tau
                if step <= max_steps
                    @inbounds ts_view[step] = t
                    @inbounds for k in 1:state_dim
                        us_view[step, k] = current_u[k]
                    end
                    step += 1
                end
            end
            continue
        end

        if tau2 > tau1
            tau = min(1.0, t_end - t)
            counts .= 0
            for j in 1:num_jumps
                if !critical[j]
                    counts[j] = thread_poisson(rate_cache[j] * tau)
                end
            end
            if use_implicit
                implicit_tau_step(current_u, t, tau, rate_cache, counts, current_u)
            else
                c(local_dc, current_u, p, t, counts, nothing)
                current_u .+= local_dc
            end
        else
            tau = min(1.0, t_end - t)
            counts .= 0
            if ac0 > 0
                r = thread_rand() * ac0
                cumsum_rate = 0.0
                for j in 1:num_jumps
                    if critical[j]
                        cumsum_rate += rate_cache[j]
                        if cumsum_rate > r
                            counts[j] = 1
                            break
                        end
                    end
                end
            end
            for j in 1:num_jumps
                if !critical[j]
                    counts[j] = thread_poisson(rate_cache[j] * tau)
                end
            end
            if use_implicit && tau > tau_ex
                implicit_tau_step(current_u, t, tau, rate_cache, counts, current_u)
            else
                c(local_dc, current_u, p, t, counts, nothing)
                current_u .+= local_dc
            end
        end

        if any(current_u .< 0)
            tau1 /= 2
            continue
        end

        t += tau
        if step <= max_steps
            @inbounds ts_view[step] = t
            @inbounds for k in 1:state_dim
                us_view[step, k] = current_u[k]
            end
        end
    end
end

# Vectorized solve for ImplicitTauLeaping
function vectorized_solve(probs, prob::JumpProblem, alg::ImplicitTauLeaping; backend, trajectories, seed, max_steps)
    rj = prob.regular_jump
    state_dim = length(prob.prob.u0)
    p_correct = prob.prob.p  # Store correct p
    nu = let c = rj.c, u0 = prob.prob.u0, numjumps = rj.numjumps
        nu = zeros(Int, state_dim, numjumps)
        for j in 1:numjumps
            counts = zeros(numjumps)
            counts[j] = 1
            du = similar(u0)
            c(du, u0, p_correct, prob.prob.tspan[1], counts, nothing)
            nu[:, j] = round.(Int, du)
        end
        nu
    end
    # Explicitly bind p_correct to both c and rate
    c_fixed = (du, u, p, t, counts, mark) -> rj.c(du, u, p_correct, t, counts, mark)
    rate_fixed = (out, u, p, t) -> rj.rate(out, u, p_correct, t)
    rj_data = ImplicitTauLeapingJumpData(rate_fixed, c_fixed, rj.numjumps, nu)
    alg_data = ImplicitTauLeapingData(alg.epsilon, alg.nc, alg.nstiff, alg.delta)

    probs_data = [ImplicitTauLeapingTrajectoryData(SA{eltype(p.prob.u0)}[p.prob.u0...], p_correct, p.prob.tspan) for p in probs]

    probs_data_gpu = adapt(backend, probs_data)
    rj_data_gpu = adapt(backend, rj_data)
    alg_data_gpu = adapt(backend, alg_data)

    tspan = prob.prob.tspan
    num_jumps = rj_data.numjumps

    @assert state_dim > 0 "Dimension of state must be positive"
    @assert num_jumps >= 0 "Number of jumps must be positive"

    ts = allocate(backend, eltype(prob.prob.tspan), (max_steps, trajectories))
    us = allocate(backend, eltype(prob.prob.u0), (max_steps, state_dim, trajectories))

    current_u_buf = allocate(backend, eltype(prob.prob.u0), (state_dim, trajectories))
    rate_cache_buf = allocate(backend, eltype(prob.prob.u0), (num_jumps, trajectories))
    counts_buf = allocate(backend, Int, (num_jumps, trajectories))
    local_dc_buf = allocate(backend, eltype(prob.prob.u0), (state_dim, trajectories))
    mu_buf = allocate(backend, eltype(prob.prob.u0), (state_dim, trajectories))
    sigma2_buf = allocate(backend, eltype(prob.prob.u0), (state_dim, trajectories))
    critical_buf = allocate(backend, Bool, (num_jumps, trajectories))
    rate_new_buf = allocate(backend, eltype(prob.prob.u0), (num_jumps, trajectories))
    residual_buf = allocate(backend, eltype(prob.prob.u0), (state_dim, trajectories))
    J_buf = allocate(backend, eltype(prob.prob.u0), (state_dim, state_dim, trajectories))

    @kernel function init_buffers_kernel(@Const(probs_data), current_u_buf)
        i = @index(Global, Linear)
        @inbounds u0 = probs_data[i].u0
        @inbounds for k in 1:length(u0)
            current_u_buf[k, i] = u0[k]
        end
    end
    init_kernel = init_buffers_kernel(backend)
    init_event = init_kernel(probs_data_gpu, current_u_buf; ndrange=trajectories)
    KernelAbstractions.synchronize(backend)

    seed = seed === nothing ? UInt64(12345) : UInt64(seed)

    # Debug: Verify parameters before kernel launch
    @assert all(typeof(p.prob.p) == NTuple{4, Float64} for p in probs) "All problems must have p as NTuple{4, Float64}"
    @show typeof(probs[1].prob.p)
    @show probs[1].prob.p
    @show typeof(rj_data.rate)
    @show typeof(rj_data.c)

    kernel = implicit_tau_leaping_kernel(backend)
    main_event = kernel(probs_data_gpu, us, ts, rj_data_gpu, alg_data_gpu,
                        current_u_buf, rate_cache_buf, counts_buf, local_dc_buf,
                        mu_buf, sigma2_buf, critical_buf, rate_new_buf, residual_buf, J_buf,
                        seed, max_steps; ndrange=trajectories)
    KernelAbstractions.synchronize(backend)

    return ts, us
end
