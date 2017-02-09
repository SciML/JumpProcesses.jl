type ExtendedJumpArray{T,T2} <: AbstractArray{Float64,1}
  u::T
  jump_u::T2
end

Base.length(A::ExtendedJumpArray) = length(A.u) + length(A.jump_u)
Base.size(A::ExtendedJumpArray) = (length(A),)
function Base.getindex(A::ExtendedJumpArray,i::Int)
  i <= length(A.u) ? A.u[i] : A.jump_u[i-length(A.u)]
end
function Base.getindex(A::ExtendedJumpArray,I...)
  A[sub2ind(A.u,I...)]
end
function Base.getindex(A::ExtendedJumpArray,I::CartesianIndex{1})
  A[I[1]]
end
Base.setindex!(A::ExtendedJumpArray,v,I...) = (A[sub2ind(A.u,I...)] = v)
Base.setindex!(A::ExtendedJumpArray,v,I::CartesianIndex{1}) = (A[I[1]] = v)
function Base.setindex!(A::ExtendedJumpArray,v,i::Int)
  i <= length(A.u) ? (A.u[i] = v) : (A.jump_u[i-length(A.u)] = v)
end

linearindexing{T<:ExtendedJumpArray}(::Type{T}) = Base.LinearFast()
similar(A::ExtendedJumpArray) = deepcopy(A)

function recursivecopy!{T<:ExtendedJumpArray}(dest::T, src::T)
  recursivecopy!(dest.u,src.u)
  recursivecopy!(dest.jump_u,src.jump_u)
end
#indices(A::ExtendedJumpArray) = Base.OneTo(length(A.u) + length(A.jump_u))
display(A::ExtendedJumpArray) = display(A.u)
show(A::ExtendedJumpArray) = show(A.u)
plot_indices(A::ExtendedJumpArray) = eachindex(A.u)
