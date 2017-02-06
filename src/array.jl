type ExtendedJumpArray{T,T2} <: AbstractArray{Float64,1}
  u::T
  jump_u::T2
end

Base.length(A::ExtendedJumpArray) = length(A.u) + length(A.jump_u)
Base.size(A::ExtendedJumpArray) = (length(A),)
function Base.getindex(A::ExtendedJumpArray,i::Int)
  i < length(A.u) ? A.u[i] : A.jump_u[i]
end
Base.getindex(A::ExtendedJumpArray,i...) = A.u[i...]
function Base.setindex!(A::ExtendedJumpArray,v,i::Int)
  i < length(A.u) ? (A.u[i] = v) : (A.jump_u[i] = v)
end
Base.setindex!(A::ExtendedJumpArray,v,I...) = (A.u[I...] = v)
linearindexing{T<:ExtendedJumpArray}(::Type{T}) = Base.LinearFast()
similar(A::ExtendedJumpArray) = deepcopy(A)
