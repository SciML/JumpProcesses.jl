
# for function wrapped rates
@inline function sumrates(u, p, t, cursum, jumps::AbstractArray)
    for v in jumps
        cursum += v(u, p, t)
    end
    cursum
end

# for tuples
@inline function _sumrates(u, p, t, cursum, jump, jumps...)
    cursum += jump(u, p, t)
    _sumrates(u, p, t, cursum, jumps...)
end

@inline function _sumrates(u, p, t, cursum, jump)
    cursum + jump(u, p, t)
end

@inline sumrates(u, p, t, cursum, jumps::Tuple) = _sumrates(u, p, t, cursum, jumps...)


function extend_problem(prob::DiffEqBase.AbstractODEProblem, agg::CHV, jumps;
                        rng = DEFAULT_RNG)

    _f = SciMLBase.unwrapped_f(prob.f)

    if isinplace(prob)
        jump_f = let _f = _f
            function (du::ExtendedJumpArray, u::ExtendedJumpArray, p, s)
                t = u.jump_u[1]
                _f(du.u, u.u, p, t)
                ratesum = sumrates(u.u, p, t, jumps)
                du.jump_u = 1 / (ratesum + agg.nullrate)
            end
        end
    else
        jump_f = let _f = _f
            function (u::ExtendedJumpArray, p, t)
                du = ExtendedJumpArray(_f(u.u, p, t), u.jump_u)
                ratesum = sumrates(u.u, p, t, jumps)
                du.jump_u = 1 / (ratesum + agg.nullrate)
                return du
            end
        end
    end

    ttype = eltype(prob.tspan)
    u0 = ExtendedJumpArray(prob.u0, zero(ttype))
    remake(prob, f = ODEFunction{isinplace(prob)}(jump_f), u0 = u0)
end

function extend_problem(prob, agg::CHV, jumps; rng = DEFAULT_RNG)
    error("CHV currently does not support problems of type $(typeof(prob)), please use another aggregator.")
end