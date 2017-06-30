type CoupledArray{T,T2} <: AbstractArray{Float64,1}
  u::T
  u_control::T2
  order::Bool
end

Base.length(A::CoupledArray) = length(A.u) + length(A.u_control)
Base.size(A::CoupledArray) = (length(A),)
@inline function Base.getindex(A::CoupledArray,i::Int)
  if A.order == true
    i <= length(A.u) ? A.u[i] : A.u_control[i-length(A.u)]
  else
    i <= length(A.u) ? A.u_control[i] : A.u[i-length(A.u)]
  end
end

@inline function Base.getindex(A::CoupledArray,I...)
  A[sub2ind(A.u,I...)]
end

@inline function Base.getindex(A::CoupledArray,I::CartesianIndex{1})
  A[I[1]]
end

@inline Base.setindex!(A::CoupledArray,v,I...) = (A[sub2ind(A.u,I...)] = v)
@inline Base.setindex!(A::CoupledArray,v,I::CartesianIndex{1}) = (A[I[1]] = v)
@inline function Base.setindex!(A::CoupledArray,v,i::Int)
  if A.order == true
    i <= length(A.u) ? (A.u[i] = v) : (A.u_control[i-length(A.u)] = v)
  else
    i <= length(A.u) ? (A.u_control[i] = v) : (A.u[i-length(A.u)] = v)
  end
end

@compat Base.IndexStyle(::Type{<:CoupledArray}) = IndexLinear()
Base.similar(A::CoupledArray) = CoupledArray(similar(A.u),similar(A.u_control),A.order)
Base.similar{S}(A::CoupledArray,::Type{S}) = CoupledArray(similar(A.u,S),similar(A.u_control,S),A.order)


function recursivecopy!{T<:CoupledArray}(dest::T, src::T)
  recursivecopy!(dest.u,src.u)
  recursivecopy!(dest.u_control,src.u_control)
  dest.order = src.order
end

add_idxs1{T<:CoupledArray}(::Type{T},expr) = :($(expr).u)
add_idxs2{T<:CoupledArray}(::Type{T},expr) = :($(expr).u_control)
@generated function Base.broadcast!(f,A::CoupledArray,B...)
  exs1 = ((add_idxs1(B[i],:(B[$i])) for i in eachindex(B))...)
  exs2 = ((add_idxs2(B[i],:(B[$i])) for i in eachindex(B))...)
  res = quote
      broadcast!(f,A.u,$(exs1...));broadcast!(f,A.u_control,$(exs2...))
    end
  res
end


Base.display(A::CoupledArray) = display(A.u)
Base.show(A::CoupledArray) = show(A.u)
plot_indices(A::CoupledArray) = eachindex(A)
flip_u!(A::CoupledArray) = (A.order = !A.order)
