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
  rng_val = rand()
  i = searchsortedfirst(p.cur_rates,rng_val)
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

function time_to_next_jump(u,p,t,rates,cur_rates)
  @inbounds fill_cur_rates(u,p,t,cur_rates,1,rates...)
  sum_rate = sum(cur_rates)
  @fastmath normalizer = 1/sum_rate
  @inbounds cur_rates[1] = normalizer*cur_rates[1]
  @inbounds for i in 2:length(cur_rates) # normalize for choice, cumsum
    cur_rates[i] = normalizer*cur_rates[i] + cur_rates[i-1]
  end
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

@inline function aggregate(aggregator::Direct,u,p,t,end_time,constant_jumps,save_positions)
  rates = ((c.rate for c in constant_jumps)...)
  affects! = ((c.affect! for c in constant_jumps)...)
  cur_rates = Vector{Float64}(length(rates))
  sum_rate,next_jump = time_to_next_jump(u,p,t,rates,cur_rates)
  DirectJumpAggregation(next_jump,end_time,cur_rates,
    sum_rate,rates,affects!,save_positions)
end

DiscreteCallback(c::DirectJumpAggregation) = DiscreteCallback(c,c,initialize=c,save_positions=c.save_positions)
