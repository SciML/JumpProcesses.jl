# define new aggregator algorithms here and specify their properties

# Direct type methods

"""
Gillespie's Direct method. `ConstantRateJump` rates and affects are stored in
tuples. Fastest for a small (total) number of `ConstantRateJump`s or
`MassActionJump`s (~10). For larger numbers of possible jumps, use other
methods.

Daniel T. Gillespie, A general method for numerically simulating the stochastic
time evolution of coupled chemical reactions, Journal of Computational Physics,
22 (4), 403–434 (1976). doi:10.1016/0021-9991(76)90041-3.
"""
struct Direct <: AbstractAggregatorAlgorithm end

"""
Gillespie's Direct method. `ConstantRateJump` rates and affects are stored via
`FunctionWrappers`, which is more performant than `Direct` for very large
numbers of `ConstantRateJump`s. However, for such large numbers of jump
different classes of aggregators are usually much more performant (i.e.
`SortingDirect`, `DirectCR`, `RSSA` or `RSSACR`).

Daniel T. Gillespie, A general method for numerically simulating the stochastic
time evolution of coupled chemical reactions, Journal of Computational Physics,
22 (4), 403–434 (1976). doi:10.1016/0021-9991(76)90041-3.
"""
struct DirectFW <: AbstractAggregatorAlgorithm end

"""
The Composition-Rejection Direct method. Performs best relative to other methods
for systems with large numbers of jumps with special structure (for example a
linear chain of reactions, or jumps corresponding to particles hopping on a grid
or graph).

A. Slepoy, A.P. Thompson and S.J. Plimpton, A constant-time kinetic Monte
Carlo algorithm for simulation of large biochemical reaction networks, Journal
of Chemical Physics, 128 (20), 205101 (2008). doi:10.1063/1.2919546.

S. Mauch and M. Stalzer, Efficient formulations for exact stochastic
simulation of chemical systems, ACM Transactions on Computational Biology and
Bioinformatics, 8 (1), 27-35 (2010). doi:10.1109/TCBB.2009.47.
"""
struct DirectCR <: AbstractAggregatorAlgorithm end

"""
The Sorting Direct method. Often the fastest algorithm for smaller to moderate
sized systems (tens of jumps), or systems where a few jumps occur much more
frequently than others.

J. M. McCollum, G. D. Peterson, C. D. Cox, M. L. Simpson and N. F. Samatova, The
sorting direct method for stochastic simulation of biochemical systems with
varying reaction execution behavior, Computational Biology and Chemistry, 30
(1), 39049 (2006). doi:10.1016/j.compbiolchem.2005.10.007.
"""
struct SortingDirect <: AbstractAggregatorAlgorithm end

"""
The Rejection SSA method. One of the best methods for systems with hundreds to
many thousands of jumps (along with `RSSACR`) and sparse dependency graphs.

V. H. Thanh, C. Priami and R. Zunino, Efficient rejection-based simulation of
biochemical reactions with stochastic noise and delays, Journal of Chemical
Physics, 141 (13), 134116 (2014). doi:10.1063/1.4896985

V. H. Thanh, R. Zunino and C. Priami, On the rejection-based algorithm for
simulation and analysis of large-scale reaction networks, Journal of Chemical
Physics, 142 (24), 244106 (2015). doi:10.1063/1.4922923.
"""
struct RSSA <: AbstractAggregatorAlgorithm end

"""
The Rejection SSA Composition-Rejection method. Often the best performer for
systems with tens of thousands of jumps and sparse dependency graphs.

V. H. Thanh, R. Zunino, and C. Priami, Efficient constant-time complexity
algorithm for stochastic simulation of large reaction networks, IEEE/ACM
Transactions on Computational Biology and Bioinformatics, 14 (3), 657-667
(2017). doi:10.1109/TCBB.2016.2530066.
"""
struct RSSACR <: AbstractAggregatorAlgorithm end

"""
A rejection-based direct method.
"""
struct RDirect <: AbstractAggregatorAlgorithm end

# NRM-based methods

"""
Gillespie's First Reaction Method. Should not be used for practical applications
due to slow performance relative to all other methods.

Daniel T. Gillespie, A general method for numerically simulating the stochastic
time evolution of coupled chemical reactions, Journal of Computational Physics,
22 (4), 403–434 (1976). doi:10.1016/0021-9991(76)90041-3.
"""
struct FRM <: AbstractAggregatorAlgorithm end

"""
Gillespie's First Reaction Method with `FunctionWrappers` for handling
`ConstantRateJump`s. Should not be used for practical applications
due to slow performance relative to all other methods.

Daniel T. Gillespie, A general method for numerically simulating the stochastic
time evolution of coupled chemical reactions, Journal of Computational Physics,
22 (4), 403–434 (1976). doi:10.1016/0021-9991(76)90041-3.
"""
struct FRMFW <: AbstractAggregatorAlgorithm end

"""
The Next Reaction Method. Can significantly outperform Direct for systems with
large numbers of jumps and sparse dependency graphs, but is usually slower than
one of `DirectCR`, `RSSA`, or `RSSACR` for such systems.

M. A. Gibson and J. Bruck, Efficient exact stochastic simulation of chemical
systems with many species and many channels, Journal of Physical Chemistry A,
104 (9), 1876-1889 (2000). doi:10.1021/jp993732q.
"""
struct NRM <: AbstractAggregatorAlgorithm end

"""
An improvement of the COEVOLVE algorithm for simulating any compound jump
process that evolves through time. This method handles variable intensity
rates with user-defined bounds and inter-dependent processes. It reduces to
NRM when rates are constant. As opposed to COEVOLVE, this method syncs the
thinning procedure with the stepper which allows it to handle dependencies on
continuous dynamics.

G. A. Zagatti, S. A. Isaacson, C. Rackauckas, V. Ilin, S.-K. Ng and S. Bressan,
Extending JumpProcess.jl for fast point process simulation with time-varying
intensities, arXiv. doi:10.48550/arXiv.2306.06992.

M. Farajtabar, Y. Wang, M. Gomez-Rodriguez, S. Li, H. Zha, and L. Song,
COEVOLVE: a joint point process model for information diffusion and network
evolution, Journal of Machine Learning Research 18(1), 1305–1353 (2017). doi:
10.5555/3122009.3122050.
"""
struct Coevolve <: AbstractAggregatorAlgorithm end

"""
A constant-complexity NRM method. Stores next reaction times in a table with a specified bin width.

Kevin R. Sanft and Hans G. Othmer, Constant-complexity stochastic simulation
algorithm with optimal binning,  Journal of Chemical Physics 143, 074108
(2015). doi: 10.1063/1.4928635.
"""
struct CCNRM <: AbstractAggregatorAlgorithm end

# spatial methods

"""
The Next Subvolume Method for spatial jump process simulations. Usually slower
than `DirectCRDirect`. Uses an indexed priority queue tree structure to
determine where on the grid/graph the next jump occurs, and then the `Direct`
method to determine which jump at the given location occurs.

Elf, Johan and Ehrenberg, M, Spontaneous separation of bi-stable biochemical
systems into spatial domains of opposite phases,Systems Biology, 1(2), 230-236
(2004). doi:10.1049/sb:20045021.
"""
struct NSM <: AbstractAggregatorAlgorithm end

"""
The Direct Composition-Rejection Direct method. Uses the `DirectCR` method to
determine where on the grid/graph a jump occurs, and the `Direct` method to
determine which jump occurs at the sampled location.

Kevin R. Sanft and Hans G. Othmer, Constant-complexity stochastic simulation
algorithm with optimal binning,  Journal of Chemical Physics 143, 074108
(2015). doi: 10.1063/1.4928635.
"""
struct DirectCRDirect <: AbstractAggregatorAlgorithm end

struct DirectCRRSSA <: AbstractAggregatorAlgorithm end

const JUMP_AGGREGATORS = (Direct(), DirectFW(), DirectCR(), SortingDirect(), RSSA(), FRM(),
    FRMFW(), NRM(), RSSACR(), RDirect(), Coevolve(), CCNRM())

# For JumpProblem construction without an aggregator
struct NullAggregator <: AbstractAggregatorAlgorithm end

# true if aggregator requires a jump dependency graph
needs_depgraph(aggregator::AbstractAggregatorAlgorithm) = false
needs_depgraph(aggregator::DirectCR) = true
needs_depgraph(aggregator::SortingDirect) = true
needs_depgraph(aggregator::NRM) = true
needs_depgraph(aggregator::CCNRM) = true
needs_depgraph(aggregator::RDirect) = true
needs_depgraph(aggregator::Coevolve) = true

# true if aggregator requires a map from solution variable to dependent jumps.
# It is implicitly assumed these aggregators also require the reverse map, from
# jumps to variables they update.
needs_vartojumps_map(aggregator::AbstractAggregatorAlgorithm) = false
needs_vartojumps_map(aggregator::RSSA) = true
needs_vartojumps_map(aggregator::RSSACR) = true

# true if aggregator supports variable rates
supports_variablerates(aggregator::AbstractAggregatorAlgorithm) = false
supports_variablerates(aggregator::Coevolve) = true

# true if aggregator supports hops, e.g. diffusion
is_spatial(aggregator::AbstractAggregatorAlgorithm) = false
is_spatial(aggregator::NSM) = true
is_spatial(aggregator::DirectCRDirect) = true
is_spatial(aggregator::DirectCRRSSA) = true

# return the fastest aggregator out of the available ones
function select_aggregator(jumps::JumpSet; vartojumps_map = nothing,
        jumptovars_map = nothing, dep_graph = nothing, spatial_system = nothing,
        hopping_constants = nothing)

    # detect if a spatial SSA should be used
    !isnothing(spatial_system) && !isnothing(hopping_constants) && return DirectCRDirect

    # if variable rate jumps are present, return one of the two SSAs that support them
    if num_vrjs(jumps) > 0
        (num_bndvrjs(jumps) > 0) && return Coevolve
        return Direct
    end

    # if the number of jumps is small, return the Direct
    num_jumps(jumps) < USE_DIRECT_THRESHOLD && return Direct

    # if there are only massaction jumps, we can build the species-jumps dependency graphs
    can_build_dgs = num_crjs(jumps) == 0 && num_vrjs(jumps) == 0
    have_species_to_jumps_dgs = !isnothing(vartojumps_map) && !isnothing(jumptovars_map)

    # if we have the species-jumps dgs or can build them, use a Rejection-based methods
    if can_build_dgs || have_species_to_jumps_dgs
        (num_jumps(jumps) < USE_RSSA_THRESHOLD) && return RSSA
        return RSSACR
    elseif !isnothing(dep_graph)   # if only have a normal dg
        (num_jumps(jumps) < USE_SORTINGDIRECT_THRESHOLD) && return SortingDirect
        return DirectCR
    else
        return Direct
    end
end
