"""
$(TYPEDEF)

Extended state definition used within integrators when there are
`VariableRateJump`s in a system. For detailed examples and usage information see
the
- [Tutorial](https://diffeq.sciml.ai/stable/tutorials/discrete_stochastic_example/)

### Fields

$(FIELDS)

## Examples
```julia
using DiffEqJump, OrdinaryDiffEq
f(du,u,p,t) = du .= 0
rate(u,p,t) = (1+t)*u[1]*u[2]

# suppose we wish to decrease each of the two variables by one
# when a jump occurs
function affect!(integrator) 
   # Method 1, direct indexing works like normal
   integrator.u[1] -= 1
   integrator.u[2] -= 1

   # Method 2, if we want to broadcast or use array operations we need
   # to access integrator.u.u which is the actual state object. 
   # So equivalently to above we could have said:
   # integrator.u.u .-= 1
end

u0 = [10.0, 10.0]
vrj = VariableRateJump(rate, affect!)
oprob = ODEProblem(f, u0, (0.0,2.0))
jprob = JumpProblem(oprob, Direct(), vrj)
sol = solve(jprob,Tsit5())
```

## Notes
- If `ueja isa ExtendedJumpArray` with `ueja.u` of size `N` and `ueja.jump_u` of
  size `num_variableratejumps` then
  ```julia
  # for 1 <= i <= N
  ueja[i] == ueja.u[i]       

  # for N < i <= (N+num_variableratejumps)
  ueja[i] == ueja.jump_u[i]  
  ```
- In a system with `VariableRateJump`s all callback, `ConstantRateJump`, and
  `VariableRateJump` `affect!` functions will receive integrators with
  `integrator.u` an `ExtendedJumpArray`.
- As such, `affect!` functions that wish to modify the state via vector
  operations should use `ueja.u.u` to obtain the aliased state object.
"""
struct ExtendedJumpArray{T3<:Number, T1, T<:AbstractArray{T3,T1},T2} <: AbstractArray{T3,1}
  """The current state."""
  u::T
  """The current rate (i.e. hazard, intensity, or propensity) values for the `VariableRateJump`s."""
  jump_u::T2
end

Base.length(A::ExtendedJumpArray) = length(A.u) + length(A.jump_u)
Base.size(A::ExtendedJumpArray) = (length(A),)
@inline function Base.getindex(A::ExtendedJumpArray,i::Int)
  i <= length(A.u) ? A.u[i] : A.jump_u[i-length(A.u)]
end
@inline function Base.getindex(A::ExtendedJumpArray,I::Int...)
  prod(I) <= length(A.u) ? A.u[I...] : A.jump_u[prod(I)-length(A.u)]
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

# Required for non-diagonal noise
LinearAlgebra.mul!(c::ExtendedJumpArray,A::AbstractVecOrMat,u::AbstractVector) = mul!(c.u,A,u)

# Ignore axes
Base.similar(A::ExtendedJumpArray,::Type{S},axes::Tuple{Base.OneTo{Int}}) where {S} = ExtendedJumpArray(similar(A.u,S),similar(A.jump_u,S))

# Stiff ODE solver
function ArrayInterface.zeromatrix(A::ExtendedJumpArray)
  u = [vec(A.u);vec(A.jump_u)]
  u .* u' .* false
end
LinearAlgebra.ldiv!(A,b::ExtendedJumpArray) = LinearAlgebra.ldiv!(A,[vec(b.u);vec(b.jump_u)])

function recursivecopy!(dest::T, src::T) where T<:ExtendedJumpArray
  recursivecopy!(dest.u,src.u)
  recursivecopy!(dest.jump_u,src.jump_u)
end
Base.show(io::IO,A::ExtendedJumpArray) = show(io,A.u)
TreeViews.hastreeview(x::ExtendedJumpArray) = true
plot_indices(A::ExtendedJumpArray) = eachindex(A.u)

## broadcasting

struct ExtendedJumpArrayStyle{Style <: Broadcast.BroadcastStyle} <: Broadcast.AbstractArrayStyle{Any} end
ExtendedJumpArrayStyle(::S) where {S} = ExtendedJumpArrayStyle{S}()
ExtendedJumpArrayStyle(::S, ::Val{N}) where {S,N} = ExtendedJumpArrayStyle(S(Val(N)))
ExtendedJumpArrayStyle(::Val{N}) where N = ExtendedJumpArrayStyle{Broadcast.DefaultArrayStyle{N}}()

# promotion rules
@inline function Broadcast.BroadcastStyle(::ExtendedJumpArrayStyle{AStyle}, ::ExtendedJumpArrayStyle{BStyle}) where {AStyle, BStyle}
    ExtendedJumpArrayStyle(Broadcast.BroadcastStyle(AStyle(), BStyle()))
end
Broadcast.BroadcastStyle(::ExtendedJumpArrayStyle{Style}, ::Broadcast.DefaultArrayStyle{0}) where Style<:Broadcast.BroadcastStyle = ExtendedJumpArrayStyle{Style}()
Broadcast.BroadcastStyle(::ExtendedJumpArrayStyle, ::Broadcast.DefaultArrayStyle{N}) where N = Broadcast.DefaultArrayStyle{N}()

combine_styles(args::Tuple{})         = Broadcast.DefaultArrayStyle{0}()
@inline combine_styles(args::Tuple{Any})      = Broadcast.result_style(Broadcast.BroadcastStyle(args[1]))
@inline combine_styles(args::Tuple{Any, Any}) = Broadcast.result_style(Broadcast.BroadcastStyle(args[1]), Broadcast.BroadcastStyle(args[2]))
@inline combine_styles(args::Tuple)   = Broadcast.result_style(Broadcast.BroadcastStyle(args[1]), combine_styles(Base.tail(args)))

function Broadcast.BroadcastStyle(::Type{ExtendedJumpArray{T,S}}) where {T, S}
    ExtendedJumpArrayStyle(Broadcast.result_style(Broadcast.BroadcastStyle(T)))
end

@inline function Base.copy(bc::Broadcast.Broadcasted{ExtendedJumpArrayStyle{Style}}) where Style
    ExtendedJumpArray(copy(unpack(bc, Val(:u))),copy(unpack(bc, Val(:jump_u))))
end

@inline function Base.copyto!(dest::ExtendedJumpArray, bc::Broadcast.Broadcasted{ExtendedJumpArrayStyle{Style}}) where Style
    copyto!(dest.u,unpack(bc, Val(:u)))
    copyto!(dest.jump_u,unpack(bc, Val(:jump_u)))
    dest
end

# drop axes because it is easier to recompute
@inline unpack(bc::Broadcast.Broadcasted{Style}, i) where Style = Broadcast.Broadcasted{Style}(bc.f, unpack_args(i, bc.args))
@inline unpack(bc::Broadcast.Broadcasted{ExtendedJumpArrayStyle{Style}}, i) where Style = Broadcast.Broadcasted{Style}(bc.f, unpack_args(i, bc.args))
unpack(x,::Any) = x
unpack(x::ExtendedJumpArray, ::Val{:u}) = x.u
unpack(x::ExtendedJumpArray, ::Val{:jump_u}) = x.jump_u

@inline unpack_args(i, args::Tuple) = (unpack(args[1], i), unpack_args(i, Base.tail(args))...)
unpack_args(i, args::Tuple{Any}) = (unpack(args[1], i),)
unpack_args(::Any, args::Tuple{}) = ()
