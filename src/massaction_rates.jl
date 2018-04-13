###############################################################################
# Stochiometry for a given reaction is a vector of pairs mapping species id to
# stochiometric coefficient.
###############################################################################

@fastmath function evalrxrate(speciesvec::AbstractVector{T}, rateconst,
                              stochmat::AbstractVector{Pair{S,V}})::typeof(rateconst) where {T,S,V}
    val = one(T)

    @inbounds for specstoch in stochmat
        specpop = speciesvec[specstoch[1]]
        val    *= specpop
        @inbounds for k = 2:specstoch[2]
            specpop -= one(specpop)
            val     *= specpop
        end
    end

     rateconst * val
end

@inline @fastmath function executerx!(speciesvec::AbstractVector{T},
                                      net_stoch::AbstractVector{Pair{S,V}}) where {T,S,V}
    @inbounds for specstoch in net_stoch
        speciesvec[specstoch[1]] += specstoch[2]
    end
    nothing
end


function scalerates!(unscaled_rates::Vector{U}, stochmat::Vector{Vector{Pair{S,T}}}) where {U,S,T}
    @inbounds for i in eachindex(unscaled_rates)
        coef = one(T)
        @inbounds for specstoch in stochmat[i]
            coef *= factorial(specstoch[2])
        end
        unscaled_rates[i] /= coef
    end
    nothing
end

function scalerate(unscaled_rate::U, stochmat::Vector{Pair{S,T}}) where {U <: Number, S, T}
    coef = one(T)
    @inbounds for specstoch in stochmat
        coef *= factorial(specstoch[2])
    end
    unscaled_rate /= coef
end
