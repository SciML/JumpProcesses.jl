"""
An aggregator interface for SSA-like algorithms.
  
### Required Fields
- `next_jump`
- `next_jump_time`
- `end_time`
- `cur_rates`
- `sum_rate`
- `rates`
- `affects!`
- `save_positions`
- `rng`

### Required Methods
- `(p::SSAJumpAggregator)(dj,u,t,integrator)`: an initialization functor
- `aggregate(aggregator::AbstractAggregatorAlgorithm,u,p,t,end_time,constant_jumps,save_positions)`
- `generate_jump(p, integrator)`:
"""
abstract type SSAJumpAggregator <: AbstractJumpAggregator end

##### defaults #####

@inline retrieve_jump(p::SSAJumpAggregator) = p.next_jump_time,p.next_jump

# forbidden; see: https://github.com/JuliaLang/julia/issues/14919
# @inline function (p::SSAJumpAggregator)(u,t,integrator) # condition
#   p.next_jump_time==t
# end

# function (p::SSAJumpAggregator)(integrator) # affect!
#   ttnj, i = retrieve_jump(p)
#   @inbounds p.affects![i](integrator)
#   generate_jump!(p,integrator.u,integrator.p,integrator.t)
#   if p.next_jump_time < p.end_time
#     add_tstop!(integrator,p.next_jump_time)
#   end
#   nothing
# end

DiscreteCallback(c::SSAJumpAggregator) = DiscreteCallback(c,c,initialize=c,save_positions=c.save_positions)

##### required methods #####

generate_jump!(p::SSAJumpAggregator,u,params,t) = nothing

# (p::SSAJumpAggregator)(dj,u,t,integrator) = nothing # initialize

aggregate(aggregator::AbstractAggregatorAlgorithm,u,p,t,end_time,constant_jumps,save_positions) = nothing

##### helper functions for updating rates #####

@inline function fill_cur_rates(u,p,t,cur_rates,idx,rate,rates...)
  @inbounds cur_rates[idx] = rate(u,p,t)
  idx += 1
  fill_cur_rates(u,p,t,cur_rates,idx,rates...)
end

@inline function fill_cur_rates(u,p,t,cur_rates,idx,rate)
  @inbounds cur_rates[idx] = rate(u,p,t)
  nothing
end

function cur_rates_as_cumsum(u,p,t,rates,cur_rates)
  @inbounds fill_cur_rates(u,p,t,cur_rates,1,rates...)
  sum_rate = sum(cur_rates)
  @fastmath normalizer = 1/sum_rate
  @inbounds cur_rates[1] = normalizer*cur_rates[1]
  @inbounds for i in 2:length(cur_rates) # normalize for choice, cumsum
    cur_rates[i] = normalizer*cur_rates[i] + cur_rates[i-1]
  end
  sum_rate
end

##### helper functions for sampling jump times #####

@inline randexp_ziggurat(sum_rate) = randexp() / sum_rate
@inline randexp_inverse(sum_rate) = -log(rand()) / sum_rate

##### helper functions for sampling jump indices #####
@inline randidx_bisection(cur_rates,rng_val) = searchsortedfirst(cur_rates,rng_val)

##### helper functions for coupled sampling #####
