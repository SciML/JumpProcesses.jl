struct SimpleTauLeaping <: DiffEqBase.DEAlgorithm end

function DiffEqBase.solve(jump_prob::JumpProblem, alg::SimpleTauLeaping;
        seed = nothing,
        dt = error("dt is required for SimpleTauLeaping."))

    # boilerplate from SimpleTauLeaping method
    @assert isempty(jump_prob.jump_callback.continuous_callbacks) # still needs to be a regular jump
    @assert isempty(jump_prob.jump_callback.discrete_callbacks)
    prob = jump_prob.prob
    seed === nothing ? rng = Xorshifts.Xoroshiro128Plus() :
    rng = Xorshifts.Xoroshiro128Plus(seed)

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

export SimpleTauLeaping

struct BinomialTauLeaping <: DiffEqBase.DEAlgorithm end

function determine_reaction_type(eq_expr::Expr)
    expr_str = string(eq_expr)
    if occursin("* u[", expr_str, 2) && occursin(" * ", expr_str)
        i = parse(Int, match(r"u\[(\d+)\]", string(eq_expr)).captures[1])
        return "Homodimer formation", i

    elseif occursin("* u[", expr_str, 2)
        i = parse.(Int, match(r"u\[(\d+)\] \* u\[(\d+)\]", string(eq_expr)).captures)
        return "Second-order reaction", i
    
    elseif occursin("* u[", expr_str)
        i = parse(Int, match(r"u\[(\d+)\]", string(eq_expr)).captures[1])
        return "First-order reaction", i
    else
        return "Unknown"
    end
end

# Determine N_j, dependent on the reaction type
function N_j(rate::Vector{Expr}, uprev::Vector{Int})
    N_j_values = Vector{Int}(undef, length(rate))

    for (eq_idx, eq_expr) in enumerate(rate)
        reaction_type, i = determine_reaction_type(eq_expr)

        if reaction_type == "First-order reaction"
            N_j_values[eq_idx] = uprev[i]

        elseif reaction_type == "Second-order reaction"
            i, j = i[1], i[2]
            N_j_values[eq_idx] = min(uprev[i], uprev[j])

        elseif reaction_type == "Homodimer formation"
            i = get_species_index(eq_expr)
            N_j_values[eq_idx] = state[i] >= 2 ? floor(0.5 * state[i]) : 0

        else # for unknown reaction types, we set N_j = 0
            N_j_values[eq_idx] = 0

        end
    end

    return N_j_values
end

function DiffEqBase.solve(jump_prob::JumpProblem, alg::BinomialTauLeaping;
        seed = nothing,
        dt = error("dt is required for BinomialTauLeaping."))

    # boilerplate from BinomialTauLeaping method
    @assert isempty(jump_prob.jump_callback.continuous_callbacks) # still needs to be a regular jump
    @assert isempty(jump_prob.jump_callback.discrete_callbacks)
    prob = jump_prob.prob
    seed === nothing ? rng = Xorshifts.Xoroshiro128Plus() :
    rng = Xorshifts.Xoroshiro128Plus(seed)

    rj = jump_prob.regular_jump
    rate = rj.rate # rate function rate(out,u,p,t)
    numjumps = rj.numjumps # used for size information (# of jump processes)
    c = rj.c # matrix-free operator c(u_buffer, uprev, tprev, counts, p, mark)

    if !isnothing(rj.mark_dist) == nothing # https://github.com/JuliaDiffEq/DifferentialEquations.jl/issues/250
        error("Mark distributions are currently not supported in BinomialTauLeaping")
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
        N_j = N_j(rate_cache, uprev)
        
        counts .= rand(rng, Binomial(N_j, rate_cache/N_j))
        c(du, uprev, p, tprev, counts, mark)
        u[i] = du + uprev
    end

    sol = DiffEqBase.build_solution(prob, alg, t, u,
        calculate_error = false,
        interp = DiffEqBase.ConstantInterpolation(t, u))
end

export BinomialTauLeaping

