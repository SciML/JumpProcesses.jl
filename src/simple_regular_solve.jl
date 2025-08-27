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

struct SimpleAdaptiveTauLeaping{T <: AbstractFloat} <: DiffEqBase.DEAlgorithm
    epsilon::T  # Error control parameter
end

SimpleAdaptiveTauLeaping(; epsilon=0.05) = SimpleAdaptiveTauLeaping(epsilon)

function compute_gi(u, nu, hor, i)
    max_order = 1.0
    for j in 1:size(nu, 2)
        if abs(nu[i, j]) > 0
            max_order = max(max_order, float(hor[j]))
        end
    end
    return max_order
end

function compute_tau_explicit(u, rate_cache, nu, hor, p, t, epsilon, rate, dtmin)
    rate(rate_cache, u, p, t)
    mu = zeros(length(u))
    sigma2 = zeros(length(u))
    tau = Inf
    for i in 1:length(u)
        for j in 1:size(nu, 2)
            mu[i] += nu[i, j] * rate_cache[j]
            sigma2[i] += nu[i, j]^2 * rate_cache[j]
        end
        gi = compute_gi(u, nu, hor, i)
        bound = max(epsilon * u[i] / gi, 1.0)
        mu_term = abs(mu[i]) > 0 ? bound / abs(mu[i]) : Inf
        sigma_term = sigma2[i] > 0 ? bound^2 / sigma2[i] : Inf
        tau = min(tau, mu_term, sigma_term)
    end
    return max(tau, dtmin)
end

function DiffEqBase.solve(jump_prob::JumpProblem, alg::SimpleAdaptiveTauLeaping; 
        seed = nothing,
        dtmin = 1e-10,
        saveat = nothing)
    validate_pure_leaping_inputs(jump_prob, alg) ||
        error("SimpleTauLeaping can only be used with PureLeaping JumpProblems with only non-RegularJumps.")
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

    # Compute initial stoichiometry and HOR
    nu = zeros(Int, length(u0), numjumps)
    counts_temp = zeros(Int, numjumps)
    for j in 1:numjumps
        fill!(counts_temp, 0)
        counts_temp[j] = 1
        c(du, u0, p, t[1], counts_temp, nothing)
        nu[:, j] = du
    end
    hor = zeros(Int, size(nu, 2))
    for j in 1:size(nu, 2)
        hor[j] = sum(abs.(nu[:, j])) > maximum(abs.(nu[:, j])) ? 2 : 1
    end

    saveat_times = isnothing(saveat) ? Vector{typeof(t)}() : saveat isa Number ? collect(range(tspan[1], tspan[2], step=saveat)) : collect(saveat)
    save_idx = 1

    while t[end] < t_end
        u_prev = u[end]
        t_prev = t[end]
        # Recompute stoichiometry
        for j in 1:numjumps
            fill!(counts_temp, 0)
            counts_temp[j] = 1
            c(du, u_prev, p, t_prev, counts_temp, nothing)
            nu[:, j] = du
        end
        rate(rate_cache, u_prev, p, t_prev)
        tau = compute_tau_explicit(u_prev, rate_cache, nu, hor, p, t_prev, epsilon, rate, dtmin)
        tau = min(tau, t_end - t_prev)
        if !isempty(saveat_times)
            if save_idx <= length(saveat_times) && t_prev + tau > saveat_times[save_idx]
                tau = saveat_times[save_idx] - t_prev
            end
        end
        counts .= pois_rand.(rng, max.(rate_cache * tau, 0.0))
        c(du, u_prev, p, t_prev, counts, nothing)
        u_new = u_prev + du
        if any(<(0), u_new)
            # Halve tau to avoid negative populations, as per Cao et al. (2006, J. Chem. Phys., DOI: 10.1063/1.2159468)
            tau /= 2
            continue
        end
        u_new = max.(u_new, 0)  # Ensure non-negative states
        push!(u, u_new)
        push!(t, t_prev + tau)
        if !isempty(saveat_times) && save_idx <= length(saveat_times) && t[end] >= saveat_times[save_idx]
            save_idx += 1
        end
    end

    # Interpolate to saveat times if specified
    if !isempty(saveat_times)
        t_out = saveat_times
        u_out = [u[end]]
        for t_save in saveat_times
            idx = findlast(ti -> ti <= t_save, t)
            push!(u_out, u[idx])
        end
        t = t_out
        u = u_out[2:end]
    end

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

export SimpleTauLeaping, EnsembleGPUKernel, SimpleAdaptiveTauLeaping
