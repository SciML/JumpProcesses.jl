"""
$(TYPEDEF)

Extended state definition used within integrators when there are
`VariableRateJump`s in a system. For detailed examples and usage information, see
the
- [Tutorial](https://docs.sciml.ai/JumpProcesses/stable/tutorials/discrete_stochastic_example/)

### Fields

$(FIELDS)

## Examples
```julia
using JumpProcesses, OrdinaryDiffEq
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
struct ExtendedJumpArray{T3 <: Number, T1, T <: AbstractArray{T3, T1}, T2} <:
       AbstractArray{T3, 1}
    """The current state."""
    u::T
    """The current rate (i.e. hazard, intensity, or propensity) values for the `VariableRateJump`s."""
    jump_u::T2
end

Base.length(A::ExtendedJumpArray) = length(A.u) + length(A.jump_u)
Base.size(A::ExtendedJumpArray) = (length(A),)
@inline function Base.getindex(A::ExtendedJumpArray, i::Int)
    i <= length(A.u) ? A.u[i] : A.jump_u[i - length(A.u)]
end
@inline function Base.getindex(A::ExtendedJumpArray, I::Int...)
    prod(I) <= length(A.u) ? A.u[I...] : A.jump_u[prod(I) - length(A.u)]
end
@inline function Base.getindex(A::ExtendedJumpArray, I::CartesianIndex{1})
    A[I[1]]
end
@inline Base.setindex!(A::ExtendedJumpArray, v, I...) = (A[CartesianIndices(A.u, I...)] = v)
@inline Base.setindex!(A::ExtendedJumpArray, v, I::CartesianIndex{1}) = (A[I[1]] = v)
@inline function Base.setindex!(A::ExtendedJumpArray, v, i::Int)
    i <= length(A.u) ? (A.u[i] = v) : (A.jump_u[i - length(A.u)] = v)
end

Base.IndexStyle(::Type{<:ExtendedJumpArray}) = IndexLinear()
Base.similar(A::ExtendedJumpArray) = ExtendedJumpArray(similar(A.u), similar(A.jump_u))
function Base.similar(A::ExtendedJumpArray, ::Type{S}) where {S}
    ExtendedJumpArray(similar(A.u, S), similar(A.jump_u, S))
end
Base.zero(A::ExtendedJumpArray) = fill!(similar(A), 0)

# Required for non-diagonal noise
function LinearAlgebra.mul!(c::ExtendedJumpArray, A::AbstractVecOrMat, u::AbstractVector)
    mul!(c.u, A, u)
end

# Ignore axes
function Base.similar(A::ExtendedJumpArray, ::Type{S},
    axes::Tuple{Base.OneTo{Int}}) where {S}
    ExtendedJumpArray(similar(A.u, S), similar(A.jump_u, S))
end

# ODE norm to prevent type-unstable fallback
@inline function DiffEqBase.ODE_DEFAULT_NORM(u::ExtendedJumpArray, t)
    Base.FastMath.sqrt_fast(real(sum(abs2, u)) / max(length(u), 1))
end

# Stiff ODE solver
function ArrayInterface.zeromatrix(A::ExtendedJumpArray)
    u = [vec(A.u); vec(A.jump_u)]
    u .* u' .* false
end
function LinearAlgebra.ldiv!(A::LinearAlgebra.LU, b::ExtendedJumpArray)
    LinearAlgebra.ldiv!(A, [vec(b.u); vec(b.jump_u)])
end

function recursivecopy!(dest::T, src::T) where {T <: ExtendedJumpArray}
    recursivecopy!(dest.u, src.u)
    recursivecopy!(dest.jump_u, src.jump_u)
end
Base.show(io::IO, A::ExtendedJumpArray) = show(io, A.u)
TreeViews.hastreeview(x::ExtendedJumpArray) = true
plot_indices(A::ExtendedJumpArray) = eachindex(A.u)

## broadcasting

# The jump array styles stores two sub-styles in the type,
# one for the `u` array and one for the `jump_u` array
struct ExtendedJumpArrayStyle{UStyle <: Broadcast.BroadcastStyle,
    JumpUStyle <: Broadcast.BroadcastStyle} <:
       Broadcast.BroadcastStyle end
# Init style based on type of u/jump_u
function ExtendedJumpArrayStyle(::US, ::JumpUS) where {US, JumpUS}
    ExtendedJumpArrayStyle{US, JumpUS}()
end
function Base.BroadcastStyle(::Type{
    ExtendedJumpArray{T3, T1, UType, JumpUType},
}) where {T3,
    T1,
    UType,
    JumpUType,
}
    ExtendedJumpArrayStyle(Base.BroadcastStyle(UType), Base.BroadcastStyle(JumpUType))
end

# Combine with other styles by combining individually with u/jump_u styles
function Base.BroadcastStyle(::ExtendedJumpArrayStyle{UStyle, JumpUStyle},
    ::Style) where {UStyle, JumpUStyle,
    Style <: Base.Broadcast.BroadcastStyle}
    ExtendedJumpArrayStyle(Broadcast.result_style(UStyle(), Style()),
        Broadcast.result_style(JumpUStyle(), Style()))
end

# Decay back to the DefaultArrayStyle for higher-order default styles, to support adding to raw vectors as needed
function Base.BroadcastStyle(::ExtendedJumpArrayStyle{UStyle, JumpUStyle},
    ::Broadcast.DefaultArrayStyle{0}) where {UStyle, JumpUStyle}
    ExtendedJumpArrayStyle(UStyle(), JumpUStyle())
end

function Base.BroadcastStyle(::ExtendedJumpArrayStyle{UStyle, JumpUStyle},
    ::Broadcast.DefaultArrayStyle{N}) where {N, UStyle, JumpUStyle}
    Broadcast.DefaultArrayStyle{N}()
end

# Lookup the first ExtendedJumpArray to pick output container size
"`A = find_eja(args)` returns the first ExtendedJumpArray among the arguments."
find_eja(bc::Base.Broadcast.Broadcasted) = find_eja(bc.args)
find_eja(args::Tuple) = find_eja(find_eja(args[1]), Base.tail(args))
find_eja(x) = x
find_eja(::Tuple{}) = nothing
find_eja(a::ExtendedJumpArray, rest) = a
find_eja(::Any, rest) = find_eja(rest)

function Base.similar(bc::Broadcast.Broadcasted{ExtendedJumpArrayStyle{US, JumpUS}},
    ::Type{ElType}) where {US, JumpUS, ElType}
    A = find_eja(bc)
    ExtendedJumpArray(similar(A.u, ElType), similar(A.jump_u, ElType))
end

# Helper functions that repack broadcasted functions
@inline function repack(bc::Broadcast.Broadcasted{Style}, i) where {Style}
    Broadcast.Broadcasted{Style}(bc.f, repack_args(i, bc.args))
end
@inline function repack(bc::Broadcast.Broadcasted{ExtendedJumpArrayStyle{US, JumpUS}},
    i::Val{:u}) where {US, JumpUS}
    Broadcast.Broadcasted{US}(bc.f, repack_args(i, bc.args))
end
@inline function repack(bc::Broadcast.Broadcasted{ExtendedJumpArrayStyle{US, JumpUS}},
    i::Val{:jump_u}) where {US, JumpUS}
    Broadcast.Broadcasted{JumpUS}(bc.f, repack_args(i, bc.args))
end

# Helper functions that repack arguments
@inline repack(x, ::Any) = x
@inline repack(x::ExtendedJumpArray, ::Val{:u}) = x.u
@inline repack(x::ExtendedJumpArray, ::Val{:jump_u}) = x.jump_u

# Repack args with generated function to do this in a type-stable way without recursion
@generated function repack_args(extract_symbol, args::NTuple{N, Any}) where {N}
    # Extract over the arg tuple
    extracted_args = [:(repack(args[$i], extract_symbol)) for i in 1:N]
    # Splat extracted args to another args tuple
    return quote
        ($(extracted_args...),)
    end
end

@inline function Base.copyto!(dest::ExtendedJumpArray,
    bc::Broadcast.Broadcasted{ExtendedJumpArrayStyle{US, JumpUS}}) where {
    US,
    JumpUS,
}
    copyto!(dest.u, repack(bc, Val(:u)))
    copyto!(dest.jump_u, repack(bc, Val(:jump_u)))
    dest
end

Base.:*(x::ExtendedJumpArray, y::Number) = ExtendedJumpArray(y .* x.u, y .* x.jump_u)
Base.:*(y::Number, x::ExtendedJumpArray) = ExtendedJumpArray(y .* x.u, y .* x.jump_u)
Base.:/(x::ExtendedJumpArray, y::Number) = ExtendedJumpArray(x.u ./ y, x.jump_u ./ y)
function Base.:+(x::ExtendedJumpArray, y::ExtendedJumpArray)
    ExtendedJumpArray(x.u .+ y.u, x.jump_u .+ y.jump_u)
end
