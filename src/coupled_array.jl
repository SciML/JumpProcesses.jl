type CoupledArray{T,T2} <: AbstractArray{Float64,1}
  u::T
  u_control::T2
  order::Bool
end

Base.length(A::CoupledArray) = length(A.u) + length(A.u_control)
Base.size(A::CoupledArray) = (length(A),)
function Base.getindex(A::CoupledArray,i::Int)
  if A.order == true
    i <= length(A.u) ? A.u[i] : A.u_control[i-length(A.u)]
  else
    i <= length(A.u) ? A.u_control[i] : A.u[i-length(A.u)]
  end
end

function Base.getindex(A::CoupledArray,I...)
  A[sub2ind(A.u,I...)]
end

function Base.getindex(A::CoupledArray,I::CartesianIndex{1})
  A[I[1]]
end

Base.setindex!(A::CoupledArray,v,I...) = (A[sub2ind(A.u,I...)] = v)
Base.setindex!(A::CoupledArray,v,I::CartesianIndex{1}) = (A[I[1]] = v)
function Base.setindex!(A::CoupledArray,v,i::Int)
  if A.order == true
    i <= length(A.u) ? (A.u[i] = v) : (A.u_control[i-length(A.u)] = v)
  else
    i <= length(A.u) ? (A.u_control[i] = v) : (A.u[i-length(A.u)] = v)
  end
end

linearindexing{T<:CoupledArray}(::Type{T}) = Base.LinearFast()
similar(A::CoupledArray) = deepcopy(A)


function recursivecopy!{T<:CoupledArray}(dest::T, src::T)
  recursivecopy!(dest.u,src.u)
  recursivecopy!(dest.u_control,src.u_control)
end

display(A::CoupledArray) = display(A.u)
show(A::CoupledArray) = show(A.u)
plot_indices(A::CoupledArray) = eachindex(A)
flip_u!(A::CoupledArray) = (A.order = !A.order)
