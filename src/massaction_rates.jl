###############################################################################
# Stochiometry for a given reaction is a vector of pairs mapping species id to
# stochiometric coefficient.
###############################################################################

@inline function evalrxrate(speciesvec::AbstractVector{T}, rxidx,
        majump::MassActionJump{U})::R where {T <: Integer, R, U <: AbstractVector{R}}
    val = one(T)
    @inbounds for specstoch in majump.reactant_stoch[rxidx]
        specpop = speciesvec[specstoch[1]]
        val *= specpop
        @inbounds for k in 2:specstoch[2]
            specpop -= one(specpop)
            val *= specpop
        end
    end

    @inbounds return val * majump.scaled_rates[rxidx]
end

@inline function evalrxrate(speciesvec::AbstractVector{T}, rxidx,
        majump::MassActionJump{U})::R where {T <: Real, R, U <: AbstractVector{R}}
    val = one(T)
    @inbounds for specstoch in majump.reactant_stoch[rxidx]
        specpop = speciesvec[specstoch[1]]
        val *= specpop
        @inbounds for k in 2:specstoch[2]
            specpop -= one(specpop)
            val *= specpop
        end
        # we need to check the smallest rate law term is positive
        # i.e. for an order k reaction: x - k + 1 > 0 
        (specpop <= 0) && return zero(R)
    end

    @inbounds return val * majump.scaled_rates[rxidx]
end

@inline function executerx!(speciesvec::AbstractVector{T}, rxidx::S,
        majump::M) where {T, S, M <: AbstractMassActionJump}
    @inbounds net_stoch = majump.net_stoch[rxidx]
    @inbounds for specstoch in net_stoch
        speciesvec[specstoch[1]] += specstoch[2]
    end
    nothing
end

@inline function executerx(speciesvec::SVector{T}, rxidx::S,
        majump::M) where {T, S, M <: AbstractMassActionJump}
    @inbounds net_stoch = majump.net_stoch[rxidx]
    @inbounds for specstoch in net_stoch
        speciesvec = setindex(speciesvec, speciesvec[specstoch[1]] + specstoch[2],
            specstoch[1])
    end
    speciesvec

    #=
    map(net_stoch) do stoch
        @inbounds speciesvec[stoch[1]] + stoch[2]
    end
    =#
end

function scalerates!(unscaled_rates::AbstractVector{U},
        stochmat::AbstractVector{V}) where {U, S, T, W <: Pair{S, T},
        V <: AbstractVector{W}}
    @inbounds for i in eachindex(unscaled_rates)
        coef = one(T)
        @inbounds for specstoch in stochmat[i]
            coef *= factorial(specstoch[2])
        end
        unscaled_rates[i] /= coef
    end
    nothing
end

function scalerates!(unscaled_rates::AbstractMatrix{U},
        stochmat::AbstractVector{V}) where {U, S, T, W <: Pair{S, T},
        V <: AbstractVector{W}}
    @inbounds for i in size(unscaled_rates, 1)
        coef = one(T)
        @inbounds for specstoch in stochmat[i]
            coef *= factorial(specstoch[2])
        end
        unscaled_rates[i, :] /= coef
    end
    nothing
end

function scalerate(unscaled_rate::U,
        stochmat::AbstractVector{Pair{S, T}}) where {U <: Number, S, T}
    coef = one(T)
    @inbounds for specstoch in stochmat
        coef *= factorial(specstoch[2])
    end
    unscaled_rate /= coef
end

###############################################################################
# dependency graph when MassActionJump uses pairs to represent (species,stoich)
###############################################################################

# map from species to reactions depending on that species
# uses a Vector instead of a Set as the latter requires isEqual,
# and by using an underlying Dict can be slower for small numbers
# of dependencies
function var_to_jumps_map(numspec, ma_jumps::AbstractMassActionJump)
    numrxs = get_num_majumps(ma_jumps)

    # map from a species to reactions that depend on it
    spec_to_dep_rxs = [Vector{Int}() for n in 1:numspec]
    for rx in 1:numrxs
        for (spec, stoch) in ma_jumps.reactant_stoch[rx]
            push!(spec_to_dep_rxs[spec], rx)
        end
    end

    foreach(s -> unique!(sort!(s)), spec_to_dep_rxs)
    spec_to_dep_rxs
end

"""
make a map from reactions to dependent species
"""
function jump_to_vars_map(majumps)
    [[s for (s, c) in majumps.net_stoch[i]] for i in 1:get_num_majumps(majumps)]
end

# dependency graph is a map from a reaction to a vector of reactions
# that should depend on species it changes
function make_dependency_graph(numspec, ma_jumps::AbstractMassActionJump)
    numrxs = get_num_majumps(ma_jumps)
    spec_to_dep_rxs = var_to_jumps_map(numspec, ma_jumps)

    # create map from rx to reactions depending on it
    dep_graph = [Vector{Int}() for n in 1:numrxs]
    for rx in 1:numrxs

        # rx changes spec, hence rxs depending on spec depend on rx
        for (spec, stoch) in ma_jumps.net_stoch[rx]
            for dependent_rx in spec_to_dep_rxs[spec]
                push!(dep_graph[rx], dependent_rx)
            end
        end
    end

    add_self_dependencies!(dep_graph, dosort = false)
    foreach(deps -> unique!(sort!(deps)), dep_graph)
    dep_graph
end

# update dependency graph to make sure jumps depend on themselves
function add_self_dependencies!(dg; dosort = true)
    for (i, jump_deps) in enumerate(dg)
        if !any(y -> isequal(y, i), jump_deps)
            push!(jump_deps, i)
            dosort && sort!(jump_deps)
        end
    end
end

@inline massaction_data(jump::MassActionJump) =
    (jump.scaled_rates, jump.reactant_stoch, jump.net_stoch)
@inline massaction_data(jump::MassActionJump{<:Number}) =
    ((jump.scaled_rates,), (jump.reactant_stoch,), (jump.net_stoch,))

"""
    massaction_rates!(rates, jump::MassActionJump, u)

Write each reaction propensity at populations `u` into `rates`. Uses the stored,
combinatorially scaled rate constants and falling factorial rate laws. A reaction
with insufficient reactants has zero propensity. `rates` must have one entry per
reaction and must not alias `u` or the jump's data. Rate constants must already
be initialized; for parameter-dependent rates, use the mass-action jump stored
in a `JumpProblem`.
"""
function massaction_rates!(rates, jump::MassActionJump, u)
    constants, reactants, _ = massaction_data(jump)
    for j in eachindex(constants)
        rates[j] = massaction_propensity(constants, reactants, u, j)
    end
    return rates
end

@inline function massaction_propensity(constants, reactants, u, j)
    rate = one(eltype(u))
    for (species, order) in reactants[j]
        population = u[species]
        population <= order - 1 && return zero(rate)
        for k in 0:(order - 1)
            rate *= population - k
        end
    end
    return rate * constants[j]
end

"""
    massaction_stoichiometry_mul!(du, jump::MassActionJump, counts)

Overwrite `du` with the net stoichiometry times the reaction vector `counts`.
`counts` may contain reaction counts or propensities and must have one entry per
reaction. `du` must have one entry per species and must not alias `counts` or the
jump's data.
"""
function massaction_stoichiometry_mul!(du, jump::MassActionJump, counts)
    fill!(du, zero(eltype(du)))
    _, _, stoichiometry = massaction_data(jump)
    for j in eachindex(stoichiometry)
        for (species, coefficient) in stoichiometry[j]
            du[species] += coefficient * counts[j]
        end
    end
    return du
end

"""
    massaction_drift!(du, jump::MassActionJump, u)

Overwrite `du` with the mass-action drift, the net stoichiometry times the
propensities at `u`. Evaluates and accumulates each reaction without allocating
an intermediate propensity vector. Supports automatic differentiation through
`u` when `du` can store the resulting scalar type. `du` must have one entry per
species and must not alias `u` or the jump's data. Initialize rate constants as
for [`massaction_rates!`](@ref).
"""
function massaction_drift!(du, jump::MassActionJump, u)
    fill!(du, zero(eltype(du)))
    constants, reactants, stoichiometry = massaction_data(jump)
    for j in eachindex(constants)
        rate = massaction_propensity(constants, reactants, u, j)
        for (species, coefficient) in stoichiometry[j]
            du[species] += coefficient * rate
        end
    end
    return du
end

leaping_rates!(out, jump::MassActionJump, u, p, t) = massaction_rates!(out, jump, u)
leaping_rates!(out, jump::RegularJump, u, p, t) = jump.rate(out, u, p, t)
leaping_change!(du, jump::MassActionJump, u, p, t, counts, mark) =
    massaction_stoichiometry_mul!(du, jump, counts)
leaping_change!(du, jump::RegularJump, u, p, t, counts, mark) =
    jump.c(du, u, p, t, counts, mark)
leaping_num_jumps(jump::MassActionJump) = get_num_majumps(jump)
leaping_num_jumps(jump::RegularJump) = jump.numjumps
