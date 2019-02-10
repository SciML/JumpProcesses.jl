mutable struct DirectCRJumpAggregation{T,S,F1,F2,RNG} <: AbstractSSAJumpAggregator
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
    DirectCRJumpAggregation{T,S,F1,F2,RNG}(nj::Int, njt::T, et::T, crs::Vector{T}, sr::T, maj::S, rs::F1, affs!::F2, sps::Tuple{Bool,Bool}, rng::RNG) where {T,S,F1,F2,RNG} =
      new{T,S,F1,F2,RNG}(nj, njt, et, crs, sr, maj, rs, affs!, sps, rng)
  end
  DirectCRJumpAggregation(nj::Int, njt::T, et::T, crs::Vector{T}, sr::T, maj::S, rs::F1, affs!::F2, sps::Tuple{Bool,Bool}, rng::RNG; kwargs...) where {T,S,F1,F2,RNG} =
    DirectCRJumpAggregation{T,S,F1,F2,RNG}(nj, njt, et, crs, sr, maj, rs, affs!, sps, rng)
  