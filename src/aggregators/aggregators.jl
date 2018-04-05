struct Direct <: AbstractAggregatorAlgorithm end
struct DirectManyJumps <: AbstractAggregatorAlgorithm end 

# For JumpProblem construction without an aggregator
struct NullAggregator <: AbstractAggregatorAlgorithm end
