const AVecOrNothing = Union{AbstractVector, Nothing}
const AMatOrNothing = Union{AbstractMatrix, Nothing}

"""
    SpatialMassActionJump(uniform_rates, spatial_rates, reactant_stoch, net_stoch,
        param_mapper = nothing; scale_rates = true, useiszero = true, nocopy = false)
    SpatialMassActionJump(spatial_rates, reactant_stoch, net_stoch, param_mapper = nothing;
        kwargs...)
    SpatialMassActionJump(uniform_rates, reactant_stoch, net_stoch, param_mapper = nothing;
        kwargs...)
    SpatialMassActionJump(ma_jumps::MassActionJump; scale_rates = false, kwargs...)

Represent mass-action reactions whose rate constants may be uniform across sites, vary by
site, or include both uniform and spatially varying reaction channels.

Uniform reactions are ordered before spatially varying reactions. For a spatial rate matrix,
rows index reactions and columns index sites.

## Arguments

  - `uniform_rates`: Vector of rate constants for reactions that use the same rate at every
    site, or `nothing`.
  - `spatial_rates`: Matrix of rate constants for site-dependent reactions, or `nothing`.
  - `reactant_stoch`: Reactant stoichiometry for each reaction, using the same pair-vector
    representation as [`MassActionJump`](@ref).
  - `net_stoch`: Net stoichiometry for each reaction.
  - `param_mapper`: Optional function mapping problem parameters to rate constants.
  - `ma_jumps`: Existing [`MassActionJump`](@ref) to reinterpret as spatial mass-action
    reactions.

## Keyword Arguments

  - `scale_rates`: Whether to divide rates by factorial stoichiometry factors. Defaults to
    `true` for raw rates and `false` when constructing from an existing `MassActionJump`.
  - `useiszero`: Whether a single zero reactant entry is treated as an empty reactant list.
  - `nocopy`: Whether to store input arrays directly instead of copying them.

## Fields

  - `uniform_rates`: Uniform reaction rates, or `nothing`.
  - `spatial_rates`: Site-dependent reaction-rate matrix, or `nothing`.
  - `reactant_stoch`: Reactant stoichiometry by reaction.
  - `net_stoch`: Net stoichiometry by reaction.
  - `param_mapper`: Optional parameter-to-rate mapping.

## Examples

```julia
using JumpProcesses

rates = [0.1 0.2 0.3]
reactant_stoch = [[1 => 1]]
net_stoch = [[1 => -1]]
smaj = SpatialMassActionJump(rates, reactant_stoch, net_stoch)
get_num_majumps(smaj) == 1
```
"""
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
