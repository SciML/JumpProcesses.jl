# define new aggregator algorithms here and specify their properties

struct Direct <: AbstractAggregatorAlgorithm end
struct DirectFW <: AbstractAggregatorAlgorithm end
struct DirectCR <: AbstractAggregatorAlgorithm end
struct SortingDirect <: AbstractAggregatorAlgorithm end
struct RSSA <: AbstractAggregatorAlgorithm end
struct FRM <: AbstractAggregatorAlgorithm end
struct FRMFW <: AbstractAggregatorAlgorithm end
struct NRM <: AbstractAggregatorAlgorithm end
struct RSSACR <: AbstractAggregatorAlgorithm end
struct RDirect <: AbstractAggregatorAlgorithm end


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
