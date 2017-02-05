type ConstantRateJumpCallback{R,T,A} <: DECallback
    next_jump::T
    end_time::T
    rate::R
    affect!::A
    save_positions::Tuple{Bool,Bool}
end

@inline function (p::ConstantRateJumpCallback)(t,u,integrator)
  p.next_jump==t #condition
end

@inline function (p::ConstantRateJumpCallback)(integrator) # affect!
  p.affect!(integrator)
  p.next_jump = integrator.t + time_to_next_jump(integrator.t,integrator.u,p.rate)
  if p.next_jump < p.end_time
    add_tstop!(integrator,p.next_jump)
  end
end

@inline function time_to_next_jump(t,u,rate)
  randexp()/rate(t,u)
  #rand(Exponential(1/rate(t,u)))
end

ConstantRateJumpCallback(next_jump,c::ConstantRateJump,end_time) = ConstantRateJumpCallback(next_jump,end_time,c.rate,c.affect!,c.save_positions)

DiscreteCallback(c::ConstantRateJumpCallback) = DiscreteCallback(c,c,c.save_positions)

function DiscreteCallback(t,u,c::ConstantRateJump,end_time)
  next_jump = time_to_next_jump(t,u,c.rate)
  DiscreteCallback(ConstantRateJumpCallback(next_jump,c,end_time))
end
