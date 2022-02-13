struct SpatialMassActionJump{F,S,U,V} <: AbstractMassActionJump
    uniform_rates::Vector{F} # reactions that are uniform in space
    spatial_rates::Matrix{F} # reactions whose rate depends on the site
    reactant_stoch::S
    net_stoch::U
    param_mapper::V
end

"""
uniform rates go first in ordering
"""
function SpatialMassActionJump{F,S,U,V}(uniform_rates::Vector{F}, spatial_rates::Matrix{F}, reactant_stoch::S, net_stoch::U, param_mapper::V, scale_rates::Bool, useiszero::Bool, nocopy::Bool) where {F, S, U, V}
    uniform_rates = nocopy ? uniform_rates : copy(uniform_rates)
    spatial_rates = nocopy ? spatial_rates : copy(spatial_rates)
    reactant_stoch = nocopy ? reactant_stoch : copy(reactant_stoch)
    for i in eachindex(reactant_stoch)
      if useiszero && (length(reactant_stoch[i]) == 1) && iszero(reactant_stoch[i][1][1])
        reactant_stoch[i] = typeof(reactant_stoch[i])()
      end
    end

    if scale_rates && !isempty(uniform_rates)
      scalerates!(uniform_rates, reactant_stoch)
    end
    if scale_rates && !isempty(spatial_rates)
      scalerates!(spatial_rates, reactant_stoch)
    end
    new(uniform_rates, spatial_rates, reactant_stoch, net_stoch, param_mapper)
  end

# rx_coefficients[i,j] is the coefficient of reaction i at site j by which the reaction rate will be multiplied
function SpatialMassActionJump(ma_jumps::MassActionJump{T,S,U,V}, rx_coefficients::Matrix) where {T,S,U,V}
    @assert size(rx_coefficients, 1) == get_num_majumps(ma_jumps)
    spatial_rates = copy(rx_coefficients)
    for i in 1:size(spatial_rates, 1)
        spatial_rates[i,:] *= ma_jumps.scaled_rates[i]
    end
    F = eltype(spatial_rates)
    SpatialMassActionJump{F, S, U, V}(zeros(F, 0), spatial_rates, ma_jumps.reactant_stoch, ma_jumps.net_stoch, ma_jumps.param_mapper)
end

function SpatialMassActionJump(ma_jumps::MassActionJump{T,S,U,V}) where {T,S,U,V}
    F = eltype{ma_jumps.scaled_rates}
    SpatialMassActionJump{F, S, U, V}(ma_jumps.scaled_rates, zeros(F, 0), ma_jumps.reactant_stoch, ma_jumps.net_stoch, ma_jumps.param_mapper)
end

get_num_majumps(spatial_majump::SpatialMassActionJump) = length(spatial_majump.uniform_rates) + size(spatial_majump.spatial_rates, 1)
using_params(spatial_majump::SpatialMassActionJump) = false

function rate_at_site(rx, site, spatial_majump::SpatialMassActionJump)
    num_unif_rxs = length(spatial_majump.uniform_rates[rx])
    rx <= num_unif_rxs ? spatial_majump.uniform_rates[rx] : spatial_majump.spatial_rates[rx-num_unif_rxs, site]
end

function evalrxrate(speciesmat, rxidx::S, majump::SpatialMassActionJump, site::Int) where {T,S}
    val = one(T)
    @inbounds for specstoch in majump.reactant_stoch[rxidx]
        specpop = speciesmat[specstoch[1], site]
        val    *= specpop
        @inbounds for k = 2:specstoch[2]
            specpop -= one(specpop)
            val     *= specpop
        end
    end

    @inbounds return val * rate_at_site(rxidx, site, majump)
end