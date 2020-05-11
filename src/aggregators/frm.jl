mutable struct FRMJumpAggregation{T,S,F1,F2,RNG} <: AbstractSSAJumpAggregator
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
  FRMJumpAggregation{T,S,F1,F2,RNG}(nj::Int, njt::T, et::T, crs::Vector{T}, sr::T, maj::S, rs::F1, affs!::F2, sps::Tuple{Bool,Bool}, rng::RNG) where {T,S,F1,F2,RNG} =
    new{T,S,F1,F2,RNG}(nj, njt, et, crs, sr, maj, rs, affs!, sps, rng)
end
FRMJumpAggregation(nj::Int, njt::T, et::T, crs::Vector{T}, sr::T, maj::S, rs::F1, affs!::F2, sps::Tuple{Bool,Bool}, rng::RNG; kwargs...) where {T,S,F1,F2,RNG} =
    FRMJumpAggregation{T,S,F1,F2,RNG}(nj, njt, et, crs, sr, maj, rs, affs!, sps, rng)


########### The following routines should be templates for all SSAs ###########

# condition for jump to occur
@inline function (p::FRMJumpAggregation)(u, t, integrator)
  p.next_jump_time == t
end

# executing jump at the next jump time
function (p::FRMJumpAggregation)(integrator)
  execute_jumps!(p, integrator, integrator.u, integrator.p, integrator.t)
  generate_jumps!(p, integrator, integrator.u, integrator.p, integrator.t)
  register_next_jump_time!(integrator, p, integrator.t)
  nothing
end

# setting up a new simulation
function (p::FRMJumpAggregation)(dj, u, t, integrator) # initialize
  initialize!(p, integrator, u, integrator.p, t)
  register_next_jump_time!(integrator, p, t)
  nothing
end

############################# Required Functions #############################

# creating the JumpAggregation structure (tuple-based constant jumps)
function aggregate(aggregator::FRM, u, p, t, end_time, constant_jumps,
                    ma_jumps, save_positions, rng; kwargs...)

  # handle constant jumps using tuples
  rates, affects! = get_jump_info_tuples(constant_jumps)

  build_jump_aggregation(FRMJumpAggregation, u, p, t, end_time, ma_jumps, rates, affects!,
                          save_positions, rng; kwargs...)
end

# creating the JumpAggregation structure (function wrapper-based constant jumps)
function aggregate(aggregator::FRMFW, u, p, t, end_time, constant_jumps,
                    ma_jumps, save_positions, rng; kwargs...)

  # handle constant jumps using function wrappers
  rates, affects! = get_jump_info_fwrappers(u, p, t, constant_jumps)

  build_jump_aggregation(FRMJumpAggregation, u, p, t, end_time, ma_jumps, rates, affects!,
                          save_positions, rng; kwargs...)
end

# set up a new simulation and calculate the first jump / jump time
function initialize!(p::FRMJumpAggregation, integrator, u, params, t)
  generate_jumps!(p, integrator, u, params, t)
  nothing
end

# execute one jump, changing the system state
@inline function execute_jumps!(p::FRMJumpAggregation, integrator, u, params, t)
  num_ma_rates = get_num_majumps(p.ma_jumps)
  if p.next_jump <= num_ma_rates
      if u isa SVector
        integrator.u = executerx(u, p.next_jump, p.ma_jumps)
      else
        @inbounds executerx!(u, p.next_jump, p.ma_jumps)
      end 
  else
      idx = p.next_jump - num_ma_rates
      @inbounds p.affects![idx](integrator)
  end
  nothing
end

# calculate the next jump / jump time
function generate_jumps!(p::FRMJumpAggregation, integrator, u, params, t)
  nextmaj, ttnmaj = next_ma_jump(p, u, params, t)
  nextcrj, ttncrj = next_constant_rate_jump(p, u, params, t)

  # execute reaction with minimal time
  if ttnmaj < ttncrj
    p.next_jump      = nextmaj
    p.next_jump_time = t + ttnmaj
  else
    p.next_jump      = nextcrj
    p.next_jump_time = t + ttncrj
  end
  nothing
end


######################## SSA specific helper routines ########################

# mass action jumps
@fastmath function next_ma_jump(p::FRMJumpAggregation, u, params, t)
    ttnj      = typemax(typeof(t))
    nextrx    = zero(Int)
    majumps   = p.ma_jumps
    @inbounds for i in 1:get_num_majumps(majumps)
        p.cur_rates[i] = evalrxrate(u, i, majumps)
        dt = randexp(p.rng) / p.cur_rates[i]
        if dt < ttnj
            ttnj   = dt
            nextrx = i
        end
    end
    nextrx, ttnj
end

# tuple-based constant jumps
@fastmath function next_constant_rate_jump(p::FRMJumpAggregation{T,S,F1,F2,RNG}, u, params, t) where {T,S,F1 <: Tuple, F2 <: Tuple, RNG}
    ttnj   = typemax(typeof(t))
    nextrx = zero(Int)
    if !isempty(p.rates)
        idx = get_num_majumps(p.ma_jumps) + 1
        fill_cur_rates(u, params, t, p.cur_rates, idx, p.rates...)
        @inbounds for i in idx:length(p.cur_rates)
            dt = randexp(p.rng) / p.cur_rates[i]
            if dt < ttnj
                ttnj   = dt
                nextrx = i
            end
        end
    end
    nextrx, ttnj
end

# function wrapper-based constant jumps
@fastmath function next_constant_rate_jump(p::FRMJumpAggregation{T,S,F1,F2,RNG}, u, params, t) where {T,S,F1 <: AbstractArray,F2 <: AbstractArray, RNG}
    ttnj   = typemax(typeof(t))
    nextrx = zero(Int)
    if !isempty(p.rates)
        idx = get_num_majumps(p.ma_jumps) + 1
        @inbounds for i in 1:length(p.rates)
            p.cur_rates[idx] = p.rates[i](u, params, t)
            dt = randexp(p.rng) / p.cur_rates[idx]
            if dt < ttnj
                ttnj   = dt
                nextrx = idx
            end
            idx += 1
        end
    end
    nextrx, ttnj
end
