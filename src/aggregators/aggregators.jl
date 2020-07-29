# define new aggregator algorithms here and specify their properties

"""
Gillespie, Daniel T. (1976). A General Method for Numerically Simulating the
Stochastic Time Evolution of Coupled Chemical Reactions. Journal of
Computational Physics. 22 (4): 403–434. doi:10.1016/0021-9991(76)90041-3.
"""
struct Direct <: AbstractAggregatorAlgorithm end

"""
Gillespie, Daniel T. (1976). A General Method for Numerically Simulating the
Stochastic Time Evolution of Coupled Chemical Reactions. Journal of
Computational Physics. 22 (4): 403–434. doi:10.1016/0021-9991(76)90041-3.
"""
struct DirectFW <: AbstractAggregatorAlgorithm end

"""
- A. Slepoy, A.P. Thompson and S.J. Plimpton, A constant-time kinetic Monte
  Carlo algorithm for simulation of large biochemical reaction networks, Journal
  of Chemical Physics, 128 (20), 205101 (2008). doi:10.1063/1.2919546

- S. Mauch and M. Stalzer, Efficient formulations for exact stochastic
  simulation of chemical systems, ACM Transactions on Computational Biology and
  Bioinformatics, 8 (1), 27-35 (2010). doi:10.1109/TCBB.2009.47
"""
struct DirectCR <: AbstractAggregatorAlgorithm end

"""
J. M. McCollum, G. D. Peterson, C. D. Cox, M. L. Simpson and N. F. Samatova, The
  sorting direct method for stochastic simulation of biochemical systems with
  varying reaction execution behavior, Computational Biology and Chemistry,
  30 (1), 39049 (2006). doi:10.1016/j.compbiolchem.2005.10.007
"""
struct SortingDirect <: AbstractAggregatorAlgorithm end

"""
- V. H. Thanh, C. Priami and R. Zunino, Efficient rejection-based simulation of
  biochemical reactions with stochastic noise and delays, Journal of Chemical
  Physics, 141 (13), 134116 (2014). doi:10.1063/1.4896985

- V. H. Thanh, R. Zunino and C. Priami, On the rejection-based algorithm for
  simulation and analysis of large-scale reaction networks, Journal of Chemical
  Physics, 142 (24), 244106 (2015). doi:10.1063/1.4922923
"""
struct RSSA <: AbstractAggregatorAlgorithm end

"""
Gillespie, Daniel T. (1976). A General Method for Numerically Simulating the
Stochastic Time Evolution of Coupled Chemical Reactions. Journal of
Computational Physics. 22 (4): 403–434. doi:10.1016/0021-9991(76)90041-3.
"""
struct FRM <: AbstractAggregatorAlgorithm end

"""
Gillespie, Daniel T. (1976). A General Method for Numerically Simulating the
Stochastic Time Evolution of Coupled Chemical Reactions. Journal of
Computational Physics. 22 (4): 403–434. doi:10.1016/0021-9991(76)90041-3.
"""
struct FRMFW <: AbstractAggregatorAlgorithm end

"""
M. A. Gibson and J. Bruck, Efficient exact stochastic simulation of chemical
systems with many species and many channels, Journal of Physical Chemistry A,
104 (9), 1876-1889 (2000). doi:10.1021/jp993732q
"""
struct NRM <: AbstractAggregatorAlgorithm end
struct RSSACR <: AbstractAggregatorAlgorithm end
struct RDirect <: AbstractAggregatorAlgorithm end
struct WellMixedSpatial  <: AbstractSpatialAggregatorAlgorithm
    WellMixedSSA :: AbstractAggregatorAlgorithm
end


const JUMP_AGGREGATORS = (Direct(),DirectFW(),DirectCR(),SortingDirect(),RSSA(),FRM(),FRMFW(),NRM(),RSSACR(), RDirect())

# For JumpProblem construction without an aggregator
struct NullAggregator <: AbstractAggregatorAlgorithm end

# true if aggregator requires a jump dependency graph
needs_depgraph(aggregator::AbstractAggregatorAlgorithm) = false
needs_depgraph(aggregator::DirectCR) = true
needs_depgraph(aggregator::SortingDirect) = true
needs_depgraph(aggregator::NRM) = true
needs_depgraph(aggregator::RDirect) = true

# true if aggregator requires a map from solution variable to dependent jumps.
# It is implicitly assumed these aggregators also require the reverse map, from
# jumps to variables they update.
needs_vartojumps_map(aggregator::AbstractAggregatorAlgorithm) = false
needs_vartojumps_map(aggregator::RSSA) = true
needs_vartojumps_map(aggregator::RSSACR) = true
