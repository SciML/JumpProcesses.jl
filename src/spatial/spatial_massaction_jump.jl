const AVecOrNothing = Union{AbstractVector, Nothing}
const AMatOrNothing = Union{AbstractMatrix, Nothing}

struct SpatialMassActionJump{A <: AVecOrNothing, B <: AMatOrNothing, S, U, V} <:
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
            nocopy::Bool) where {A <: AVecOrNothing,
            B <: AMatOrNothing,
            S, U, V}
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
        nocopy = false) where {A <: AVecOrNothing,
        B <: AMatOrNothing, S, U, V}
    SpatialMassActionJump{A, B, S, U, V}(urates, srates, rs, ns, pmapper, scale_rates,
        useiszero, nocopy)
end
function SpatialMassActionJump(urates::A, srates::B, rs, ns; scale_rates = true,
        useiszero = true,
        nocopy = false) where {A <: AVecOrNothing,
        B <: AMatOrNothing}
    SpatialMassActionJump(urates, srates, rs, ns, nothing; scale_rates = scale_rates,
        useiszero = useiszero, nocopy = nocopy)
end

function SpatialMassActionJump(srates::B, rs, ns, pmapper; scale_rates = true,
        useiszero = true,
        nocopy = false) where {B <: AMatOrNothing}
    SpatialMassActionJump(nothing, srates, rs, ns, pmapper; scale_rates = scale_rates,
        useiszero = useiszero, nocopy = nocopy)
end
function SpatialMassActionJump(srates::B, rs, ns; scale_rates = true, useiszero = true,
        nocopy = false) where {B <: AMatOrNothing}
    SpatialMassActionJump(nothing, srates, rs, ns, nothing; scale_rates = scale_rates,
        useiszero = useiszero, nocopy = nocopy)
end

function SpatialMassActionJump(urates::A, rs, ns, pmapper; scale_rates = true,
        useiszero = true,
        nocopy = false) where {A <: AVecOrNothing}
    SpatialMassActionJump(urates, nothing, rs, ns, pmapper; scale_rates = scale_rates,
        useiszero = useiszero, nocopy = nocopy)
end
function SpatialMassActionJump(urates::A, rs, ns; scale_rates = true, useiszero = true,
        nocopy = false) where {A <: AVecOrNothing}
    SpatialMassActionJump(urates, nothing, rs, ns, nothing; scale_rates = scale_rates,
        useiszero = useiszero, nocopy = nocopy)
end

# scale_rates defaults to false since ma_jumps.scaled_rates are already scaled;
# passing true would double-scale.
function SpatialMassActionJump(ma_jumps::MassActionJump{T, S, U, V}; scale_rates = false,
        useiszero = true, nocopy = false) where {T, S, U, V}
    SpatialMassActionJump(ma_jumps.scaled_rates, ma_jumps.reactant_stoch,
        ma_jumps.net_stoch, ma_jumps.param_mapper;
        scale_rates = scale_rates, useiszero = useiszero, nocopy = nocopy)
end

##############################################

function get_num_majumps(smaj::SpatialMassActionJump{
        Nothing, Nothing, S, U, V}) where
        {S, U, V}
    0
end
function get_num_majumps(smaj::SpatialMassActionJump{
        Nothing, B, S, U, V}) where
        {B, S, U, V}
    size(smaj.spatial_rates, 1)
end
function get_num_majumps(smaj::SpatialMassActionJump{
        A, Nothing, S, U, V}) where
        {A, S, U, V}
    length(smaj.uniform_rates)
end
function get_num_majumps(smaj::SpatialMassActionJump{
        A, B, S, U, V}) where
        {A <: AbstractVector, B <: AbstractMatrix, S, U, V}
    length(smaj.uniform_rates) + size(smaj.spatial_rates, 1)
end
using_params(smaj::SpatialMassActionJump) = false

function rate_at_site(rx, site,
        smaj::SpatialMassActionJump{Nothing, B, S, U, V}) where {B, S, U, V}
    smaj.spatial_rates[rx, site]
end
function rate_at_site(rx, site,
        smaj::SpatialMassActionJump{A, Nothing, S, U, V}) where {A, S, U, V}
    smaj.uniform_rates[rx]
end
function rate_at_site(rx, site,
        smaj::SpatialMassActionJump{A, B, S, U, V}) where
        {A <: AbstractVector, B <: AbstractMatrix, S, U, V}
    num_unif_rxs = length(smaj.uniform_rates)
    rx <= num_unif_rxs ? smaj.uniform_rates[rx] :
    smaj.spatial_rates[rx - num_unif_rxs, site]
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
