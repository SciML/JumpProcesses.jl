type DirectJumpAggregation{T,F1,F2} <: SSAJumpAggregator
  next_jump::Int
  next_jump_time::T
  end_time::T
  cur_rates::Vector{T}
  sum_rate::T
  rates::F1
  affects!::F2
  save_positions::Tuple{Bool,Bool}
end

@inline function (p::DirectJumpAggregation)(u,t,integrator) # condition
  p.next_jump_time==t
end

function (p::DirectJumpAggregation)(integrator) # affect!
  ttnj, i = retrieve_jump(p)
  @inbounds p.affects![i](integrator)
  generate_jump!(p,integrator.u,integrator.p,integrator.t)
  if p.next_jump_time < p.end_time
    add_tstop!(integrator,p.next_jump_time)
  end
  nothing
end

function generate_jump!(p::DirectJumpAggregation,u,params,t)
  # update the jump rates
  sum_rate = cur_rates_as_cumsum(u,params,t,p.rates,p.cur_rates)
  # determine next jump index
  i = randidx_bisection(p.cur_rates, rand())
  # determine next jump time
  ttnj = randexp_ziggurat(sum_rate)
  # mutate fields
  p.sum_rate = sum_rate
  p.next_jump = i
  p.next_jump_time = t + ttnj
  nothing
end

function (p::DirectJumpAggregation)(dj,u,t,integrator) # initialize
  generate_jump!(p,u,integrator.p,t)
  if p.next_jump_time < p.end_time
    add_tstop!(integrator,p.next_jump_time)
  end
  nothing
end

@inline function aggregate(aggregator::Direct,u,p,t,end_time,constant_jumps,save_positions)
  rates = ((c.rate for c in constant_jumps)...)
  affects! = ((c.affect! for c in constant_jumps)...)
  cur_rates = Vector{Float64}(length(rates))
  sum_rate = zero(Float64)
  next_jump = 0
  next_jump_time = typemax(Float64)
  DirectJumpAggregation(next_jump,next_jump_time,end_time,cur_rates,
    sum_rate,rates,affects!,save_positions)
end
