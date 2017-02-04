type ExtendedJumpArray{T,T2} <: AbstractArray{Float64,1}
  u::T
  jump_u::T2
end
