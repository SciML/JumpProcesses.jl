struct SpatialMassActionJump{A <: Union{AbstractVector, Nothing},
                             B <: Union{AbstractMatrix, Nothing}, S, U, V} <:
       AbstractMassActionJump
    uniform_rates::A # reactions that are uniform in space
    spatial_rates::B # reactions whose rate depends on the site
    reactant_stoch::S
    net_stoch::U
    param_mapper::V

    """
    uniform rates go first in ordering
    """
    function SpatialMassActionJump{A, B, S, U, V}(uniform_rates::A, spatial_rates::B,
                                                  reactant_stoch::S, net_stoch::U,
                                                  param_mapper::V, scale_rates::Bool,
                                                  useiszero::Bool,
                                                  nocopy::Bool) where {
                                                                       A <:
                                                                       Union{AbstractVector,
                                                                             Nothing},
                                                                       B <:
                                                                       Union{AbstractMatrix,
                                                                             Nothing}, S, U,
                                                                       V}
        uniform_rates = (nocopy || isnothing(uniform_rates)) ? uniform_rates :
                        copy(uniform_rates)
        spatial_rates = (nocopy || isnothing(spatial_rates)) ? spatial_rates :
                        copy(spatial_rates)
        reactant_stoch = nocopy ? reactant_stoch : copy(reactant_stoch)
        for i in eachindex(reactant_stoch)
            if useiszero && (length(reactant_stoch[i]) == 1) &&
               iszero(reactant_stoch[i][1][1])
                reactant_stoch[i] = typeof(reactant_stoch[i])()
            end
        end
        num_unif_rates = isnothing(uniform_rates) ? 0 : length(uniform_rates)
        if scale_rates && num_unif_rates > 0
            scalerates!(uniform_rates, reactant_stoch)
        end
        if scale_rates && !isnothing(spatial_rates) && !isempty(spatial_rates)
            scalerates!(spatial_rates, reactant_stoch[(num_unif_rates + 1):end])
        end
        new(uniform_rates, spatial_rates, reactant_stoch, net_stoch, param_mapper)
    end
end

################ Constructors ##################

function SpatialMassActionJump(urates::A, srates::B, rs::S, ns::U, pmapper::V;
                               scale_rates = true, useiszero = true,
                               nocopy = false) where {A <: Union{AbstractVector, Nothing},
                                                      B <: Union{AbstractMatrix, Nothing},
                                                      S, U, V}
    SpatialMassActionJump{A, B, S, U, V}(urates, srates, rs, ns, pmapper, scale_rates,
                                         useiszero, nocopy)
end
function SpatialMassActionJump(urates::A, srates::B, rs, ns; scale_rates = true,
                               useiszero = true,
                               nocopy = false) where {A <: Union{AbstractVector, Nothing},
                                                      B <: Union{AbstractMatrix, Nothing}}
    SpatialMassActionJump(urates, srates, rs, ns, nothing; scale_rates = scale_rates,
                          useiszero = useiszero, nocopy = nocopy)
end

function SpatialMassActionJump(srates::B, rs, ns, pmapper; scale_rates = true,
                               useiszero = true,
                               nocopy = false) where {B <: Union{AbstractMatrix, Nothing}}
    SpatialMassActionJump(nothing, srates, rs, ns, pmapper; scale_rates = scale_rates,
                          useiszero = useiszero, nocopy = nocopy)
end
function SpatialMassActionJump(srates::B, rs, ns; scale_rates = true, useiszero = true,
                               nocopy = false) where {B <: Union{AbstractMatrix, Nothing}}
    SpatialMassActionJump(nothing, srates, rs, ns, nothing; scale_rates = scale_rates,
                          useiszero = useiszero, nocopy = nocopy)
end

function SpatialMassActionJump(urates::A, rs, ns, pmapper; scale_rates = true,
                               useiszero = true,
                               nocopy = false) where {A <: Union{AbstractVector, Nothing}}
    SpatialMassActionJump(urates, nothing, rs, ns, pmapper; scale_rates = scale_rates,
                          useiszero = useiszero, nocopy = nocopy)
end
function SpatialMassActionJump(urates::A, rs, ns; scale_rates = true, useiszero = true,
                               nocopy = false) where {A <: Union{AbstractVector, Nothing}}
    SpatialMassActionJump(urates, nothing, rs, ns, nothing; scale_rates = scale_rates,
                          useiszero = useiszero, nocopy = nocopy)
end

function SpatialMassActionJump(ma_jumps::MassActionJump{T, S, U, V}; scale_rates = true,
                               useiszero = true, nocopy = false) where {T, S, U, V}
    SpatialMassActionJump(ma_jumps.scaled_rates, ma_jumps.reactant_stoch,
                          ma_jumps.net_stoch, ma_jumps.param_mapper;
                          scale_rates = scale_rates, useiszero = useiszero, nocopy = nocopy)
end

##############################################

function get_num_majumps(spatial_majump::SpatialMassActionJump{Nothing, Nothing, S, U, V}) where {
                                                                                                  S,
                                                                                                  U,
                                                                                                  V
                                                                                                  }
    0
end
function get_num_majumps(spatial_majump::SpatialMassActionJump{Nothing, B, S, U, V}) where {
                                                                                            B,
                                                                                            S,
                                                                                            U,
                                                                                            V
                                                                                            }
    size(spatial_majump.spatial_rates, 1)
end
function get_num_majumps(spatial_majump::SpatialMassActionJump{A, Nothing, S, U, V}) where {
                                                                                            A,
                                                                                            S,
                                                                                            U,
                                                                                            V
                                                                                            }
    length(spatial_majump.uniform_rates)
end
function get_num_majumps(spatial_majump::SpatialMassActionJump{A, B, S, U, V}) where {
                                                                                      A <:
                                                                                      AbstractVector,
                                                                                      B <:
                                                                                      AbstractMatrix,
                                                                                      S, U,
                                                                                      V}
    length(spatial_majump.uniform_rates) + size(spatial_majump.spatial_rates, 1)
end
using_params(spatial_majump::SpatialMassActionJump) = false

function rate_at_site(rx, site,
                      spatial_majump::SpatialMassActionJump{Nothing, B, S, U, V}) where {B,
                                                                                         S,
                                                                                         U,
                                                                                         V}
    spatial_majump.spatial_rates[rx, site]
end
function rate_at_site(rx, site,
                      spatial_majump::SpatialMassActionJump{A, Nothing, S, U, V}) where {A,
                                                                                         S,
                                                                                         U,
                                                                                         V}
    spatial_majump.uniform_rates[rx]
end
function rate_at_site(rx, site,
                      spatial_majump::SpatialMassActionJump{A, B, S, U, V}) where {
                                                                                   A <:
                                                                                   AbstractVector,
                                                                                   B <:
                                                                                   AbstractMatrix,
                                                                                   S, U, V}
    num_unif_rxs = length(spatial_majump.uniform_rates)
    rx <= num_unif_rxs ? spatial_majump.uniform_rates[rx] :
    spatial_majump.spatial_rates[rx - num_unif_rxs, site]
end

function evalrxrate(speciesmat::AbstractMatrix{T}, rxidx::S, majump::SpatialMassActionJump,
                    site::Int) where {T, S}
    val = one(T)
    @inbounds for specstoch in majump.reactant_stoch[rxidx]
        specpop = speciesmat[specstoch[1], site]
        val *= specpop
        @inbounds for k in 2:specstoch[2]
            specpop -= one(specpop)
            val *= specpop
        end
    end
    @inbounds return val * rate_at_site(rxidx, site, majump)
end
