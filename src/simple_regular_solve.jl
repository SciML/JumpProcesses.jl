struct SimpleTauLeaping <: DiffEqBase.DEAlgorithm end

struct SimpleAdaptiveTauLeaping{T <: AbstractFloat} <: DiffEqBase.DEAlgorithm
    epsilon::T  # Error control parameter
end

SimpleAdaptiveTauLeaping(; epsilon=0.05) = SimpleAdaptiveTauLeaping(epsilon)

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

function compute_hor(nu)
    hor = zeros(Int, size(nu, 2))
    for j in 1:size(nu, 2)
        order = sum(abs(stoich) for stoich in nu[:, j] if stoich < 0; init=0)
        if order > 3
            error("Reaction $j has order $order, which is not supported (maximum order is 3).")
        end
        hor[j] = order
    end
    return hor
end

function compute_gi(u, nu, hor, i)
    max_gi = 1
    for j in 1:size(nu, 2)
        if nu[i, j] < 0  # Species i is a substrate
            if hor[j] == 1
                max_gi = max(max_gi, 1)
            elseif hor[j] == 2 || hor[j] == 3
                stoich = abs(nu[i, j])
                if stoich >= 2
                    gi = 2 / stoich + 1 / (stoich - 1)
                    max_gi = max(max_gi, ceil(Int, gi))
                elseif stoich == 1
                    max_gi = max(max_gi, hor[j])
                end
            end
        end
    end
    return max_gi
end

function compute_tau_explicit(u, rate_cache, nu, hor, p, t, epsilon, rate, dtmin)
    rate(rate_cache, u, p, t)
    tau = Inf
    for i in 1:length(u)
        mu = zero(eltype(u))
        sigma2 = zero(eltype(u))
        for j in 1:size(nu, 2)
            mu += nu[i, j] * rate_cache[j]
            sigma2 += nu[i, j]^2 * rate_cache[j]
        end
        gi = compute_gi(u, nu, hor, i)
        bound = max(epsilon * u[i] / gi, 1.0)
        mu_term = abs(mu) > 0 ? bound / abs(mu) : Inf
        sigma_term = sigma2 > 0 ? bound^2 / sigma2 : Inf
        tau = min(tau, mu_term, sigma_term)
    end
    return max(tau, dtmin)
end

function DiffEqBase.solve(jump_prob::JumpProblem, alg::SimpleAdaptiveTauLeaping; 
        seed = nothing,
        dtmin = 1e-10,
        saveat = nothing)
    validate_pure_leaping_inputs(jump_prob, alg) ||
        error("SimpleAdaptiveTauLeaping can only be used with PureLeaping JumpProblem with a MassActionJump.")
    prob = jump_prob.prob
    rng = DEFAULT_RNG
    (seed !== nothing) && seed!(rng, seed)

    maj = jump_prob.massaction_jump
    numjumps = get_num_majumps(maj)
    rj = jump_prob.regular_jump
    # Extract rates
    rate = rj !== nothing ? rj.rate :
        (out, u, p, t) -> begin
            for j in 1:numjumps
                out[j] = evalrxrate(u, j, maj)
            end
        end
    c = rj !== nothing ? rj.c : nothing
    u0 = copy(prob.u0)
    tspan = prob.tspan
    p = prob.p

    # Initialize current state and saved history
    u_current = copy(u0)
    t_current = tspan[1]
    usave = [copy(u0)]
    tsave = [tspan[1]]
    rate_cache = zeros(float(eltype(u0)), numjumps)
    counts = zero(rate_cache)
    du = similar(u0)
    t_end = tspan[2]
    epsilon = alg.epsilon

    # Extract stoichiometry once from MassActionJump
    nu = zeros(float(eltype(u0)), length(u0), numjumps)
    for j in 1:numjumps
        for (spec_idx, stoich) in maj.net_stoch[j]
            nu[spec_idx, j] = stoich
        end
    end
    hor = compute_hor(nu)

    # Set up saveat_times
    saveat_times = nothing
    if isnothing(saveat)
        saveat_times = Vector{typeof(tspan[1])}()
    elseif saveat isa Number
        saveat_times = collect(range(tspan[1], tspan[2], step=saveat))
    else
        saveat_times = collect(saveat)
    end

    save_idx = 1

    while t_current < t_end
        rate(rate_cache, u_current, p, t_current)
        tau = compute_tau_explicit(u_current, rate_cache, nu, hor, p, t_current, epsilon, rate, dtmin)
        tau = min(tau, t_end - t_current)
        if !isempty(saveat_times) && save_idx <= length(saveat_times) && t_current + tau > saveat_times[save_idx]
            tau = saveat_times[save_idx] - t_current
        end
        counts .= pois_rand.(rng, max.(rate_cache * tau, 0.0))
        du .= 0
        if c !== nothing
            c(du, u_current, p, t_current, counts, nothing)
        else
            for j in 1:numjumps
                for (spec_idx, stoich) in maj.net_stoch[j]
                    du[spec_idx] += stoich * counts[j]
                end
            end
        end
        u_new = u_current + du
        if any(<(0), u_new)
            # Halve tau to avoid negative populations, as per Cao et al. (2006, J. Chem. Phys., DOI: 10.1063/1.2159468)
            tau /= 2
            continue
        end
        for i in eachindex(u_new)
            u_new[i] = max(u_new[i], 0)
        end
        t_new = t_current + tau

        # Save state if at a saveat time or if saveat is empty
        if isempty(saveat_times) || (save_idx <= length(saveat_times) && t_new >= saveat_times[save_idx])
            push!(usave, u_new)
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

export SimpleTauLeaping, EnsembleGPUKernel, SimpleAdaptiveTauLeaping
