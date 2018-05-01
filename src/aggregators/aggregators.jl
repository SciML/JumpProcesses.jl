struct Direct <: AbstractAggregatorAlgorithm end
struct DirectFW <: AbstractAggregatorAlgorithm end 
struct FRM <: AbstractAggregatorAlgorithm end
struct FRMFW <: AbstractAggregatorAlgorithm end
struct SortingDirect <: AbstractAggregatorAlgorithm end

# For JumpProblem construction without an aggregator
struct NullAggregator <: AbstractAggregatorAlgorithm end

# true if aggregator requires a dependency graph
needs_depgraph(aggregator::AbstractAggregatorAlgorithm) = false
needs_depgraph(aggregator::SortingDirect) = true
