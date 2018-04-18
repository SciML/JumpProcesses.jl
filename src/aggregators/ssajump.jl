"""
An aggregator interface for SSA-like algorithms.
  
### Required Fields
- `next_jump`
- `next_jump_time`
- `end_time`
- `cur_rates`
- `sum_rate`
- `ma_jumps`
- `rates`
- `affects!`
- `save_positions`
- `rng`
"""
abstract type AbstractSSAJumpAggregator <: AbstractJumpAggregator end

########### The following routines should be templates for all SSAs ###########

# # condition for jump to occur
# @inline function (p::SSAJumpAggregator)(u, t, integrator) 
#   p.next_jump_time == t
# end

# # executing jump at the next jump time
# function (p::SSAJumpAggregator)(integrator) 
#   execute_jumps!(p, integrator, integrator.u, integrator.p, integrator.t)
#   generate_jumps!(p, integrator, integrator.u, integrator.p, integrator.t)
#   register_next_jump_time!(integrator, p, integrator.t)
#   nothing
# end

# # setting up a new simulation
# function (p::SSAJumpAggregator)(dj, u, t, integrator) # initialize
#   initialize!(p, integrator, u, integrator.p, t)
#   register_next_jump_time!(integrator, p, integrator.t)
#   nothing
# end

############################## Generic Routines ###############################

@inline function register_next_jump_time!(integrator, p::AbstractSSAJumpAggregator, t)
  if p.next_jump_time < p.end_time
    add_tstop!(integrator, p.next_jump_time)
  end
  nothing
end

# helper routine for setting up standard fields of SSA jump aggregations
function build_jump_aggregation(jump_agg_type, u, p, t, end_time, ma_jumps, rates, 
                                affects!, save_positions, rng; kwargs...)

  # mass action jumps
  majumps = ma_jumps
  if majumps == nothing
    majumps = MassActionJump(Vector{typeof(t)}(),
                             Vector{Vector{Pair{Int,eltype(u)}}}(),
                             Vector{Vector{Pair{Int,eltype(u)}}}())
  end

  # current jump rates, allows mass action rates and constant jumps
  cur_rates = Vector{typeof(t)}(length(majumps.scaled_rates) + length(rates))

  sum_rate = zero(typeof(t))
  next_jump = 0
  next_jump_time = typemax(typeof(t))
  jump_agg_type(next_jump, next_jump_time, end_time, cur_rates, sum_rate, 
                majumps, rates, affects!, save_positions, rng; kwargs...)
end

DiscreteCallback(c::AbstractSSAJumpAggregator) =DiscreteCallback(c, c, initialize = c, save_positions = c.save_positions)


