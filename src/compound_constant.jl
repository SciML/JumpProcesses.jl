type CompoundConstantRateJump{T,F1,F2} <: AbstractJump
  next_jump::T
  end_time::T
  cur_rates::Vector{T}
  sum_rate::T
  rates::F1
  affects!::F2
  save_positions::Tuple{Bool,Bool}
end

function (p::CompoundConstantRateJump)(t,u,integrator) # condition
  p.next_jump==t
end

function (p::CompoundConstantRateJump)(integrator) # affect!
  rng_val = rand()
  i = findfirst(x -> x<=rng_val,p.cur_rates) + 1
  p.affects![i](integrator)
  p.sum_rate,ttnj = time_to_next_jump(integrator.t,integrator.u,p.rates,p.cur_rates)
  p.next_jump = integrator.t + ttnj
  if p.next_jump < p.end_time
    add_tstop!(integrator,p.next_jump)
  end
end

function time_to_next_jump(t,u,rates,cur_rates)
  fill_cur_rates(t,u,cur_rates,1,rates...)
  sum_rate = sum(cur_rates)
  cur_rates[1] = cur_rates[1]/sum_rate
  for i in 2:length(cur_rates) # normalize for choice, cumsum
    cur_rates[i] = cur_rates[i]/sum_rate + cur_rates[i-1]
  end
  sum_rate,randexp()/sum_rate
end

function fill_cur_rates(t,u,cur_rates,idx,rate,rates...)
  cur_rates[idx] = rate(t,u)
  idx += 1
  fill_cur_rates(t,u,cur_rates,idx,rates...)
end

function fill_cur_rates(t,u,cur_rates,idx,rate)
  cur_rates[idx] = rate(t,u)
  nothing
end

function CompoundConstantRateJump(t,u,end_time,constant_jumps;save_positions=(true,true))
  rates = ((c.rate for c in constant_jumps)...)
  affects! = ((c.affect! for c in constant_jumps)...)
  cur_rates = Vector{Float64}(length(rates))
  sum_rate,next_jump = time_to_next_jump(t,u,rates,cur_rates)
  CompoundConstantRateJump(next_jump,end_time,cur_rates,
    sum_rate,rates,affects!,save_positions)
end

DiscreteCallback(c::CompoundConstantRateJump) = DiscreteCallback(c,c,c.save_positions)
