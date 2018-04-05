mutable struct DirectJumpAggregation{T,S,F1,F2,RNG} <: AbstractSSAJumpAggregator
  next_jump::Int
  next_jump_time::T
  end_time::T
  cur_rates::Vector{T}
  sum_rate::T
  ma_jumps::S
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

# creating the JumpAggregation structure (tuple-based constant jumps)
function aggregate(aggregator::Direct, u, p, t, end_time, constant_jumps, 
                    ma_jumps, save_positions, rng)

  # handle constant jumps using function wrappers
  rates, affects! = get_jump_info_tuples(constant_jumps)

  build_jump_aggregation(u, p, t, end_time, ma_jumps, rates, affects!, 
                          save_positions, rng)
end

# creating the JumpAggregation structure (function wrapper-based constant jumps)
function aggregate(aggregator::DirectManyJumps, u, p, t, end_time, constant_jumps, 
                    ma_jumps, save_positions, rng)

  # handle constant jumps using function wrappers
  rates, affects! = get_jump_info_fwrappers(u, p, t, constant_jumps)

  build_jump_aggregation(u, p, t, end_time, ma_jumps, rates, affects!, 
                          save_positions, rng)
end

# set up a new simulation and calculate the first jump / jump time
function initialize!(p::DirectJumpAggregation, integrator, u, params, t)
  generate_jumps!(p, integrator, u, params, t)
  nothing
end

# execute one jump, changing the system state
@inline function execute_jumps!(p::DirectJumpAggregation, integrator, u, params, t)
  num_ma_rates = length(p.ma_jumps.scaled_rates)
  if p.next_jump <= num_ma_rates
      @inbounds executerx!(u, p.ma_jumps.net_stoch[p.next_jump])
  else
      idx = p.next_jump - num_ma_rates
      @inbounds p.affects![idx](integrator)
  end
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

function build_jump_aggregation(u, p, t, end_time, ma_jumps, rates, affects!, 
                                save_positions, rng)

  # mass action jumps
  majumps = ma_jumps
  if majumps == nothing
    majumps = MassActionJump(Vector{typeof(t)}(),
                             Vector{Vector{Pair{Int,eltype(u)}}}(),
                             Vector{Vector{Pair{Int,eltype(u)}}}() )
  end

  # current jump rates, allows mass action rates and constant jumps
  cur_rates = Vector{typeof(t)}(length(majumps.scaled_rates) + length(rates))

  sum_rate = zero(typeof(t))
  next_jump = 0
  next_jump_time = typemax(typeof(t))
  DirectJumpAggregation(next_jump, next_jump_time, end_time, cur_rates, sum_rate, 
                        majumps, rates, affects!, save_positions, rng)
end


# tuple-based constant jumps
@fastmath function time_to_next_jump(p::DirectJumpAggregation{T,S,F1,F2,RNG}, u, params, t) where {T,S,F1 <: Tuple, F2 <: Tuple, RNG}
  prev_rate = zero(t)
  new_rate  = zero(t)
  cur_rates = p.cur_rates

  # mass action rates
  majumps   = p.ma_jumps
  idx       = length(majumps.scaled_rates)
  @inbounds for i in 1:idx
    new_rate     = evalrxrate(u, majumps.scaled_rates[i], majumps.reactant_stoch[i])
    cur_rates[i] = new_rate + prev_rate
    prev_rate    = cur_rates[i]
  end
  
  # constant jump rates  
  idx  += 1
  rates = p.rates
  fill_cur_rates(u, params, t, cur_rates, idx, rates...)
  @inbounds for i in idx:length(cur_rates)
    cur_rates[i] = cur_rates[i] + prev_rate
    prev_rate    = cur_rates[i]
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


# function wrapper-based constant jumps
@fastmath function time_to_next_jump(p::DirectJumpAggregation{T,S,F1,F2,RNG}, u, params, t) where {T,S,F1 <: AbstractArray,F2 <: AbstractArray, RNG}
  prev_rate = zero(t)
  new_rate  = zero(t)
  cur_rates = p.cur_rates

  # mass action rates
  majumps   = p.ma_jumps
  idx       = length(majumps.scaled_rates)
  @inbounds for i in 1:idx
    new_rate     = evalrxrate(u, majumps.scaled_rates[i], majumps.reactant_stoch[i])
    cur_rates[i] = new_rate + prev_rate
    prev_rate    = cur_rates[i]
  end

  # constant jump rates
  idx  += 1
  rates = p.rates
  @inbounds for i in 1:length(p.rates)
    new_rate       = rates[i](u, params, t)
    cur_rates[idx] = new_rate + prev_rate
    prev_rate      = cur_rates[idx]
    idx           += 1
  end

  @inbounds sum_rate = cur_rates[end]
  sum_rate, randexp(p.rng) / sum_rate
end

