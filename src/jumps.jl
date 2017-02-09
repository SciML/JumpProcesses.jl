immutable ConstantRateJump{F1,F2} <: AbstractJump
  rate::F1
  affect!::F2
  save_positions::Tuple{Bool,Bool}
end
ConstantRateJump(rate,affect!;save_positions=(true,true)) = ConstantRateJump(rate,affect!,save_positions)

immutable VariableRateJump{R,F,I,T,T2} <: AbstractJump
  rate::R
  affect!::F
  idxs::I
  rootfind::Bool
  interp_points::Int
  save_positions::Tuple{Bool,Bool}
  abstol::T
  reltol::T2
end

VariableRateJump(rate,affect!;
                   idxs = nothing,
                   rootfind=true,
                   save_positions=(true,true),
                   interp_points=10,
                   abstol=1e-12,reltol=0) = VariableRateJump(
                              rate,affect!,idxs,
                              rootfind,interp_points,
                              save_positions,abstol,reltol)

immutable JumpSet{T1,T2} <: AbstractJump
  variable_jumps::T1
  constant_jumps::T2
end

JumpSet(jump::ConstantRateJump) = JumpSet((),(jump,))
JumpSet(jump::VariableRateJump) = JumpSet((jump,),())
JumpSet() = JumpSet((),())
JumpSet(jb::Void) = JumpSet()

# For Varargs, use recursion to make it type-stable

JumpSet(jumps::AbstractJump...) = JumpSet(split_jumps((), (), jumps...)...)

@inline split_jumps(vj, cj) = vj, cj
@inline split_jumps(vj, cj, v::VariableRateJump, args...) = split_jumps((vj..., v), cj, args...)
@inline split_jumps(vj, cj, c::ConstantRateJump, args...) = split_jumps(vj, (cj..., c), args...)
@inline split_jumps(vj, cj, j::JumpSet, args...) = split_jumps((vj...,j.variable_jumps...), (cj..., j.constant_jumps...), args...)
