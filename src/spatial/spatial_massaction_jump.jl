struct SpatialMassActionJump{T,S,U,V} <: AbstractJump
    scaled_rates::T
    reactant_stoch::S
    net_stoch::U
    param_mapper::V
  

# rx_coefficients[i,j] is the coefficient of reaction i at site j by which the reaction rate will be multiplied
function SpatialMassActionJump(ma_jumps::MassActionJump{T,S,U,V}, rx_coefficients::Matrix)
    @assert size(rx_coefficients, 1) == get_num_majumps(ma_jumps)
    scaled_rates = copy(rx_coefficients)
    for i in 1:size(scaled_rates, 1) # scale rates
        scaled_rates[i,:] *= ma_jumps.scaled_rates[i]
    end
    SpatialMassActionJump{typeof(scaled_rates), S, U, V}(scaled_rates, ma_jumps.reactant_stoch, ma_jumps.net_stoch, ma_jumps.param_mapper)
end

function SpatialMassActionJump(ma_jumps::MassActionJump{T,S,U,V}, rx_coefficients::Nothing)
    SpatialMassActionJump{T, S, U, V}(ma_jumps.scaled_rates, ma_jumps.reactant_stoch, ma_jumps.net_stoch, ma_jumps.param_mapper)
end
    
get_num_majumps(spatial_majump::SpatialMassActionJump) = size(spatial_majump.scaled_rates, 1)
rate_at_site(rx, site, spatial_majump::SpatialMassActionJump{T,S,U,V} where T <: Matrix) = spatial_majump.scaled_rates[rx, site]
rate_at_site(rx, site, spatial_majump::SpatialMassActionJump{T,S,U,V} where T <: Vector) = spatial_majump.scaled_rates[rx]

function evalrxrate(speciesvec::AbstractVector{T}, rxidx::S, majump::SpatialMassActionJump, site::Int)
    val = one(T)
    @inbounds for specstoch in majump.reactant_stoch[rxidx]
        specpop = speciesvec[specstoch[1]]
        val    *= specpop
        @inbounds for k = 2:specstoch[2]
            specpop -= one(specpop)
            val     *= specpop
        end
    end

    @inbounds return val * rate_at_site(rx, site, majump)
end