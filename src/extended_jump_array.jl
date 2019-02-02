mutable struct ExtendedJumpArray{T,T2} <: AbstractArray{Float64,1}
  u::T
  jump_u::T2
end

Base.length(A::ExtendedJumpArray) = length(A.u) + length(A.jump_u)
Base.size(A::ExtendedJumpArray) = (length(A),)
@inline function Base.getindex(A::ExtendedJumpArray,i::Int)
  i <= length(A.u) ? A.u[i] : A.jump_u[i-length(A.u)]
end
@inline function Base.getindex(A::ExtendedJumpArray,I...)
  A[CartesianIndices(A.u,I...)]
end
@inline function Base.getindex(A::ExtendedJumpArray,I::CartesianIndex{1})
  A[I[1]]
end
@inline Base.setindex!(A::ExtendedJumpArray,v,I...) = (A[CartesianIndices(A.u,I...)] = v)
@inline Base.setindex!(A::ExtendedJumpArray,v,I::CartesianIndex{1}) = (A[I[1]] = v)
@inline function Base.setindex!(A::ExtendedJumpArray,v,i::Int)
  i <= length(A.u) ? (A.u[i] = v) : (A.jump_u[i-length(A.u)] = v)
end

Base.IndexStyle(::Type{<:ExtendedJumpArray}) = IndexLinear()
Base.similar(A::ExtendedJumpArray) = ExtendedJumpArray(similar(A.u),similar(A.jump_u))
Base.similar(A::ExtendedJumpArray,::Type{S}) where {S} = ExtendedJumpArray(similar(A.u,S),similar(A.jump_u,S))
Base.zero(A::ExtendedJumpArray) = fill!(similar(A),0)

# Ignore axes
Base.similar(A::ExtendedJumpArray,::Type{S},axes::Tuple{Base.OneTo{Int}}) where {S} = ExtendedJumpArray(similar(A.u,S),similar(A.jump_u,S))

function recursivecopy!(dest::T, src::T) where T<:ExtendedJumpArray
  recursivecopy!(dest.u,src.u)
  recursivecopy!(dest.jump_u,src.jump_u)
end
Base.show(io::IO,A::ExtendedJumpArray) = show(io,A.u)
TreeViews.hastreeview(x::ExtendedJumpArray) = true
plot_indices(A::ExtendedJumpArray) = eachindex(A.u)

###### Broadcast overloading

const ExtendedJumpArrayStyle = Broadcast.ArrayStyle{ExtendedJumpArray}
Base.BroadcastStyle(::Type{<:ExtendedJumpArray}) = Broadcast.ArrayStyle{ExtendedJumpArray}()
Base.BroadcastStyle(::Broadcast.ArrayStyle{ExtendedJumpArray},::Broadcast.ArrayStyle) = Broadcast.ArrayStyle{ExtendedJumpArray}()
Base.BroadcastStyle(::Broadcast.ArrayStyle,::Broadcast.ArrayStyle{ExtendedJumpArray}) = Broadcast.ArrayStyle{ExtendedJumpArray}()
Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{ExtendedJumpArray}},::Type{ElType}) where ElType = similar(bc)

add_idxs1(A,x,expr) = expr
add_idxs1(A,::Type{T},expr) where {T<:ExtendedJumpArray} = :($(expr).u)
add_idxs1(A,::Type{T},expr) where {T<:AbstractArray} = :(@view($(expr)[1:length(A.u)]))

add_idxs2(A,x,expr) = expr
add_idxs2(A,::Type{T},expr) where {T<:ExtendedJumpArray} = :($(expr).jump_u)
add_idxs2(A,::Type{T},expr) where {T<:AbstractArray} = :(@view($(expr)[1:length(A.jump_u)]))

function Base.copy(bc::Broadcast.Broadcasted{ExtendedJumpArrayStyle})
    ret = Broadcast.flatten(bc)
    __broadcast(ret.f,ret.args...)
end

function Base.copyto!(dest::AbstractArray, bc::Broadcast.Broadcasted{ExtendedJumpArrayStyle})
    ret = Broadcast.flatten(bc)
    __broadcast!(ret.f,dest,ret.args...)
end

@generated function __broadcast(f,A::ExtendedJumpArray,B...)
  exs1 = ((add_idxs1(A,B[i],:(B[$i])) for i in eachindex(B))...,)
  exs2 = ((add_idxs2(A,B[i],:(B[$i])) for i in eachindex(B))...,)
  res = quote
      ExtendedJumpArray(broadcast(f,A.u,$(exs1...)),broadcast(f,A.jump_u,$(exs2...)))
  end
  res
end

@generated function __broadcast!(f,A::ExtendedJumpArray,B...)
  exs1 = ((add_idxs1(A,B[i],:(B[$i])) for i in eachindex(B))...,)
  exs2 = ((add_idxs2(A,B[i],:(B[$i])) for i in eachindex(B))...,)
  res = quote
      broadcast!(f,A.u,$(exs1...));broadcast!(f,A.jump_u,$(exs2...))
      A
  end
  res
end
