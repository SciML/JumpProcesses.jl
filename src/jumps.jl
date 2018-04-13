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

  function MassActionJump{T,S}(rates::T, rs::S, ns::S, scale_rates::Bool) where {T <: AbstractVector,S}
    sr = copy(rates)
    if scale_rates && !isempty(sr) 
      scalerates!(sr, rs)
    end
    new(sr, rs, ns)
  end
  function MassActionJump{T,S}(rate::T, rs::S, ns::S, scale_rates::Bool) where {T <: Number,S}
    sr = scale_rates ? scalerate(rate, rs) : rate
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

# handle vector of mass action jumps 
function JumpSet(vjs, cjs, rj, majv::Vector{T}) where {T <: MassActionJump}
  if isempty(majv)
    error("JumpSets do not accept empty mass action jump collections; use \"nothing\" instead.")
  end

  maj = setup_majump_to_merge(majv[1].scaled_rates, majv[1].reactant_stoch, majv[1].net_stoch)
  for i = 2:length(majv)
    massaction_jump_combine(maj, majv[i])
  end

  JumpSet(vjs, cjs, rj, maj)
end

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


# functionality to merge two mass action jumps together

# if given containers of rates and stoichiometry directly create a jump
function setup_majump_to_merge(sr::T, rs::AbstractVector{S}, ns::AbstractVector{S}) where {T <: AbstractVector, S <: AbstractArray}
  MassActionJump(sr, rs, ns)
end

# if just given the data for one jump (and not in a container) wrap in a vector
function setup_majump_to_merge(sr::T, rs::S, ns::S) where {T <: Number, S <: AbstractArray}
  MassActionJump([sr], [rs], [ns])
end

# when given a collection of reactions to add to maj
function majump_merge!(maj::MassActionJump{U,V}, sr::U, rs::V, ns::V) where {U <: AbstractVector, V <: AbstractVector}
  append!(maj.scaled_rates, sr)
  append!(maj.reactant_stoch, rs)
  append!(maj.net_stoch, ns)
  maj
end

# when given a single jump's worth of data to add to maj
function majump_merge!(maj::MassActionJump{U,V}, sr::T, rs::S, ns::S) where {U <: AbstractVector, V <: AbstractVector, T <: Number, S <: AbstractArray}
  push!(maj.scaled_rates, sr)
  push!(maj.reactant_stoch, rs)
  push!(maj.net_stoch, ns)
  maj
end

# when maj only stores a single jump's worth of data (and not in a collection)
# create a new jump with the merged data stored in vectors
function majump_merge!(maj::MassActionJump{T,S}, sr::T, rs::S, ns::S) where {T <: Number, S <: AbstractArray}
  MassActionJump([maj.scaled_rates, sr], [maj.reactant_stoch, rs], [maj.net_stoch, ns])
end

massaction_jump_combine(maj1::MassActionJump, maj2::Void) = maj1
massaction_jump_combine(maj1::Void, maj2::MassActionJump) = maj2
massaction_jump_combine(maj1::Void, maj2::Void) = maj1
massaction_jump_combine(maj1::MassActionJump, maj2::MassActionJump) = majump_merge!(maj1, maj2.scaled_rates, maj2.reactant_stoch, maj2.net_stoch)


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
