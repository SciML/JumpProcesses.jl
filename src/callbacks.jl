type ConstantRateJumpCallback{R,T,A} <: DECallback
    next_jump::T
    end_time::T
    rate::R
    affect!::A
    save_positions::Tuple{Bool,Bool}
end

@inline function (p::ConstantRateJumpCallback)(u,t,integrator)
  p.next_jump==t #condition
end

@inline function (p::ConstantRateJumpCallback)(integrator) # affect!
  p.affect!(integrator)
  p.next_jump = integrator.t + time_to_next_jump(integrator.t,integrator.u,p.rate)
  if p.next_jump < p.end_time
    add_tstop!(integrator,p.next_jump)
  end
end

@inline function time_to_next_jump(u,p,t,rate)
  randexp()/rate(u,p,t)
  #rand(Exponential(1/rate(u,p,t)))
end

ConstantRateJumpCallback(next_jump,c::ConstantRateJump,end_time) = ConstantRateJumpCallback(next_jump,end_time,c.rate,c.affect!,c.save_positions)

DiscreteCallback(c::ConstantRateJumpCallback) = DiscreteCallback(c,c,c.save_positions)

function DiscreteCallback(u,p,t,c::ConstantRateJump,end_time)
  next_jump = time_to_next_jump(u,p,t,c.rate)
  DiscreteCallback(ConstantRateJumpCallback(next_jump,c,end_time))
end
