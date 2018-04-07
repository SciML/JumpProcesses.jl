struct ConstantRateJump{F1,F2} <: AbstractJump
  rate::F1
  affect!::F2
end

struct VariableRateJump{R,F,I,T,T2} <: AbstractJump
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

struct RegularJump{R,C,DC,MD}
    rate::R
    c::C
    dc::DC
    mark_dist::MD
    constant_c::Bool
end

RegularJump(rate,c,dc;mark_dist = nothing,constant_c = false) =
            RegularJump(rate,c,dc,mark_dist,constant_c)


struct MassActionJump{T,S} <: AbstractJump
  scaled_rates::T
  reactant_stoch::S
  net_stoch::S

  function MassActionJump{T,S}(rates::T, rs::S, ns::S, scale_rates::Bool) where {T,S}
    sr = copy(rates)
    if scale_rates && !isempty(sr) 
      scalerates!(sr, rs)
    end
    new(sr, rs, ns)
  end
end
MassActionJump(usr::T, rs::S, ns::S; scale_rates = true ) where {T,S} = MassActionJump{T,S}(usr, rs, ns, scale_rates)


struct JumpSet{T1,T2,T3,T4} <: AbstractJump
  variable_jumps::T1
  constant_jumps::T2
  regular_jump::T3
  massaction_jump::T4
end

JumpSet(jump::ConstantRateJump) = JumpSet((),(jump,),nothing,nothing)
JumpSet(jump::VariableRateJump) = JumpSet((jump,),(),nothing,nothing)
JumpSet(jump::RegularJump)      = JumpSet((),(),jump,nothing)
JumpSet(jump::MassActionJump)   = JumpSet((),(),nothing,jump)
JumpSet() = JumpSet((),(),nothing,nothing)
JumpSet(jb::Void) = JumpSet()

# For Varargs, use recursion to make it type-stable

JumpSet(jumps::AbstractJump...) = JumpSet(split_jumps((), (), nothing, nothing, jumps...)...)

@inline split_jumps(vj, cj, rj, maj) = vj, cj, rj, maj
@inline split_jumps(vj, cj, rj, maj, v::VariableRateJump, args...) = split_jumps((vj..., v), cj, rj, maj, args...)
@inline split_jumps(vj, cj, rj, maj, c::ConstantRateJump, args...) = split_jumps(vj, (cj..., c), rj, maj, args...)
@inline split_jumps(vj, cj, rj, maj, c::RegularJump, args...) = split_jumps(vj, cj, regular_jump_combine(rj,c), maj, args...)
@inline split_jumps(vj, cj, rj, maj, c::MassActionJump, args...) = split_jumps(vj, cj, rj, massaction_jump_combine(maj,c), args...)
@inline split_jumps(vj, cj, rj, maj, j::JumpSet, args...) = split_jumps((vj...,j.variable_jumps...), 
                                                                        (cj..., j.constant_jumps...), 
                                                                        regular_jump_combine(rj,j.regular_jump), 
                                                                        massaction_jump_combine(maj,j.massaction_jump), args...)

regular_jump_combine(rj1::RegularJump,rj2::Void) = rj1
regular_jump_combine(rj1::Void,rj2::RegularJump) = rj2
regular_jump_combine(rj1::Void,rj2::Void) = rj1
regular_jump_combine(rj1::RegularJump,rj2::RegularJump) = error("Only one regular jump is allowed in a JumpSet")

massaction_jump_combine(maj1::MassActionJump, maj2::Void) = maj1
massaction_jump_combine(maj1::Void, maj2::MassActionJump) = maj2
massaction_jump_combine(maj1::Void, maj2::Void) = maj1
massaction_jump_combine(maj1::MassActionJump, maj2::MassActionJump) = error("Only one mass action jump type is allowed in a JumpSet")


##### helper methods for unpacking rates and affects! from constant jumps #####
function get_jump_info_tuples(constant_jumps)
  if (constant_jumps != nothing) && !isempty(constant_jumps)
    rates    = ((c.rate for c in constant_jumps)...)
    affects! = ((c.affect! for c in constant_jumps)...)
  else
    rates    = ()
    affects! = ()
  end
  
  rates, affects!
end
  
function get_jump_info_fwrappers(u, p, t, constant_jumps)
  RateWrapper   = FunctionWrappers.FunctionWrapper{typeof(t),Tuple{typeof(u), typeof(p), typeof(t)}}
  AffectWrapper = FunctionWrappers.FunctionWrapper{Void,Tuple{Any}}

  if (constant_jumps != nothing) && !isempty(constant_jumps)  
    rates    = [RateWrapper(c.rate) for c in constant_jumps]
    affects! = [AffectWrapper(x->(c.affect!(x);nothing)) for c in constant_jumps]
  else
    rates    = Vector{RateWrapper}()
    affects! = Vector{AffectWrapper}()
  end  

  rates, affects!
end
