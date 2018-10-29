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

DiscreteCallback(c::AbstractSSAJumpAggregator) = DiscreteCallback(c, c, initialize = c, save_positions = c.save_positions)

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
  cur_rates = Vector{typeof(t)}(undef,get_num_majumps(majumps) + length(rates))

  sum_rate = zero(typeof(t))
  next_jump = 0
  next_jump_time = typemax(typeof(t))
  jump_agg_type(next_jump, next_jump_time, end_time, cur_rates, sum_rate,
                majumps, rates, affects!, save_positions, rng; kwargs...)
end

# reevaluate all rates and total rate
function fill_rates_and_sum!(p::AbstractSSAJumpAggregator, u, params, t)
  sum_rate = zero(typeof(p.sum_rate))

  # mass action jumps
  majumps   = p.ma_jumps
  cur_rates = p.cur_rates
  @inbounds for i in 1:get_num_majumps(majumps)
      cur_rates[i] = evalrxrate(u, i, majumps)
      sum_rate    += cur_rates[i]
  end

  # constant rates
  rates = p.rates
  idx   = get_num_majumps(majumps) + 1
  @inbounds for rate in rates
      cur_rates[idx] = rate(u, params, t)
      sum_rate += cur_rates[idx]
      idx += 1
  end

  p.sum_rate = sum_rate
  nothing
end

# recalculate jump rates for jumps that depend on the just executed jump
# requires dependency graph
function update_dependent_rates!(p::AbstractSSAJumpAggregator, u, params, t)
  @inbounds dep_rxs = p.dep_gr[p.next_jump]
  num_majumps = get_num_majumps(p.ma_jumps)
  cur_rates   = p.cur_rates
  sum_rate    = p.sum_rate
  majumps     = p.ma_jumps
  @inbounds for rx in dep_rxs
      sum_rate -= cur_rates[rx]
      if rx <= num_majumps
          @inbounds cur_rates[rx] = evalrxrate(u, rx, majumps)
      else
          @inbounds cur_rates[rx] = p.rates[rx-num_majumps](u, params, t)
      end
      sum_rate += cur_rates[rx]
  end

  p.sum_rate = sum_rate
  nothing
end


########## bracket interval routines for rejection methods ############
inline get_spec_brackets(uval, δ) = (one(eltype(δ)) - δ) * uval, (one(eltype(δ)) + δ) * uval
inline get_majump_brackets(ulow, uhigh, k, majumps) = evalrxrate(ulow, k, majumps), evalrxrate(uhigh, k, majumps)

# for constant rate jumps we must check the ordering of the bracket values`
inline function get_cjump_brackets(ulow, uhigh, rate, params, t)
    rlow  = rate(ulow, params, t)
    rhigh = rate(uhigh, params, t)
    if rlow > rhigh
        return rhigh,rlow
    else
        return rlow,rhigh
    end
end
