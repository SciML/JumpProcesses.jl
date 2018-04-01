using FunctionWrappers

type DirectJumpAggregation{T,F1,F2} <: AbstractJumpAggregator
  next_jump::T
  end_time::T
  cur_rates::Vector{T}
  sum_rate::T
  rates::F1
  affects!::F2
  save_positions::Tuple{Bool,Bool}
end

@inline function (p::DirectJumpAggregation)(u,t,integrator) # condition
  p.next_jump==t
end

function (p::DirectJumpAggregation)(integrator) # affect!
  rng_val = rand() * p.sum_rate
  i = searchsortedfirst(p.cur_rates, rng_val)
  @inbounds p.affects![i](integrator)
  p.sum_rate,ttnj = time_to_next_jump(integrator.u,integrator.p,integrator.t,p.rates,p.cur_rates)
  p.next_jump = integrator.t + ttnj
  if p.next_jump < p.end_time
    add_tstop!(integrator,p.next_jump)
  end
  nothing
end

function (p::DirectJumpAggregation)(dj,u,t,integrator) # initialize
  sum_rate,next_jump = time_to_next_jump(u,integrator.p,t,p.rates,p.cur_rates)
  p.sum_rate = sum_rate
  p.next_jump = t + next_jump
  if p.next_jump < p.end_time
    add_tstop!(integrator,p.next_jump)
  end
  nothing
end


################# Tuple based time to next jump ##################
function time_to_next_jump(u, p, t, rates::Tuple, cur_rates) 
  @inbounds fill_cur_rates(u,p,t,cur_rates,1,rates...)
  @inbounds for i in 2:length(cur_rates) # cumsum
    cur_rates[i] = cur_rates[i] + cur_rates[i-1]
  end
  @inbounds sum_rate = cur_rates[end]
  sum_rate,randexp()/sum_rate
end

@inline function fill_cur_rates(u,p,t,cur_rates,idx,rate,rates...)
  @inbounds cur_rates[idx] = rate(u,p,t)
  idx += 1
  fill_cur_rates(u,p,t,cur_rates,idx,rates...)
end

@inline function fill_cur_rates(u,p,t,cur_rates,idx,rate)
  @inbounds cur_rates[idx] = rate(u,p,t)
  nothing
end
##################################################################

############ FunctionWrapper based time to next jump #############
function time_to_next_jump(u, p, t, rates::Vector{T}, cur_rates) where T
  prev_rate = zero(t)
  new_rate  = zero(t)
  @inbounds for i in 1:length(rates)
      new_rate     = rates[i](u, p, t)
      cur_rates[i] = new_rate + prev_rate
      prev_rate    = cur_rates[i]
  end
  @inbounds sum_rate = cur_rates[end]
  sum_rate,randexp()/sum_rate
end
##################################################################

function getRatesAffectsAsTuples(constant_jumps)
  rates    = ((c.rate for c in constant_jumps)...)
  affects! = ((c.affect! for c in constant_jumps)...)

  return rates, affects!
end

function getRatesAffectsAsFWrappers(u, p, t, constant_jumps)
  RateWrapper   = FunctionWrappers.FunctionWrapper{typeof(t),Tuple{typeof(u), typeof(p), typeof(t)}}
  rates         = [RateWrapper(c.rate) for c in constant_jumps]
  AffectWrapper = FunctionWrappers.FunctionWrapper{Void,Tuple{Any}}
  affects!      = [AffectWrapper(x->(c.affect!(x);nothing)) for c in constant_jumps]

  return rates, affects!
end


# when to switch from tuples for rates/affects to FunctionWrappers
const TUPLE_TO_FWRAPPER_CUTOFF = 10

@inline function aggregate(aggregator::Direct,u,p,t,end_time,constant_jumps,save_positions)

  # decide if representing rates/affects as tuples or function wrappers
  if length(constant_jumps) < TUPLE_TO_FWRAPPER_CUTOFF
    rates,affects! = getRatesAffectsAsTuples(constant_jumps)
  else
    rates,affects! = getRatesAffectsAsFWrappers(u,p,t,constant_jumps)
  end

  cur_rates = Vector{typeof(t)}(length(rates))
  sum_rate,next_jump = time_to_next_jump(u,p,t,rates,cur_rates)
  DirectJumpAggregation(next_jump, end_time, cur_rates, sum_rate, rates, affects!, save_positions)
end

DiscreteCallback(c::DirectJumpAggregation) = DiscreteCallback(c,c,initialize=c,save_positions=c.save_positions)
