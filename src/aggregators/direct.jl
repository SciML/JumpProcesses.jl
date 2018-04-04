type DirectJumpAggregation{T,F1,F2,RNG} <: AbstractSSAJumpAggregator
  next_jump::Int
  next_jump_time::T
  end_time::T
  cur_rates::Vector{T}
  sum_rate::T
  rates::F1
  affects!::F2
  save_positions::Tuple{Bool,Bool}
  rng::RNG
end

########### The following routines should be templates for all SSAs ###########

# condition for jump to occur
@inline function (p::DirectJumpAggregation)(u, t, integrator) 
  p.next_jump_time == t
end

# executing jump at the next jump time
function (p::DirectJumpAggregation)(integrator) 
  execute_jumps!(p, integrator, integrator.u, integrator.p, integrator.t)
  generate_jumps!(p, integrator, integrator.u, integrator.p, integrator.t)
  register_next_jump_time!(integrator, p, integrator.t)
  nothing
end

# setting up a new simulation
function (p::DirectJumpAggregation)(dj, u, t, integrator) # initialize
  initialize!(p, integrator, u, integrator.p, t)
  register_next_jump_time!(integrator, p, t)
  nothing
end

############################# Required Functions #############################

# creating the JumpAggregation structure
function aggregate(aggregator::Direct, u, p, t, end_time, constant_jumps, save_positions, rng)
  rates = ((c.rate for c in constant_jumps)...)
  affects! = ((c.affect! for c in constant_jumps)...)
  cur_rates = Vector{Float64}(length(rates))
  sum_rate = zero(Float64)
  next_jump = 0
  next_jump_time = typemax(Float64)
  DirectJumpAggregation(next_jump, next_jump_time, end_time, cur_rates,
    sum_rate, rates, affects!, save_positions, rng)
end

# set up a new simulation and calculate the first jump / jump time
function initialize!(p::DirectJumpAggregation, integrator, u, params, t)
  generate_jumps!(p, integrator, u, params, t)
  nothing
end

# execute one jump, changing the system state
@inline function execute_jumps!(p::DirectJumpAggregation, integrator, u, params, t)
  idx = p.next_jump
  @inbounds p.affects![idx](integrator)
  nothing
end

# calculate the next jump / jump time
function generate_jumps!(p::DirectJumpAggregation, integrator, u, params, t)
  p.sum_rate, ttnj = time_to_next_jump(p, u, params, t)
  @fastmath p.next_jump_time = t + ttnj
  @inbounds p.next_jump = searchsortedfirst(p.cur_rates, rand(p.rng) * p.sum_rate)
  nothing
end

######################## SSA specific helper routines ########################

@fastmath function time_to_next_jump(p::DirectJumpAggregation, u, params, t)
  cur_rates = p.cur_rates
  rates = p.rates

  @inbounds fill_cur_rates(u, params, t, cur_rates, 1, rates...)
  @inbounds cur_rates[1] = cur_rates[1]
  @inbounds for i in 2:length(cur_rates) # normalize for choice, cumsum
    cur_rates[i] = cur_rates[i] + cur_rates[i-1]
  end
  @inbounds sum_rate = cur_rates[end]
  sum_rate, randexp(p.rng) / sum_rate
end

@inline function fill_cur_rates(u, p, t, cur_rates, idx, rate, rates...)
  @inbounds cur_rates[idx] = rate(u, p, t)
  idx += 1
  fill_cur_rates(u, p, t, cur_rates, idx, rates...)
end

@inline function fill_cur_rates(u, p, t, cur_rates, idx, rate)
  @inbounds cur_rates[idx] = rate(u, p, t)
  nothing
end
