"""
David F. Anderson, Masanori Koyama; An asymptotic relationship between coupling
methods for stochastically modeled population processes. IMA J Numer Anal 2015;
35 (4): 1757-1778. doi: 10.1093/imanum/dru044
"""
function SplitCoupledJumpProblem(prob::DiffEqBase.AbstractJumpProblem,
        prob_control::DiffEqBase.AbstractJumpProblem,
        aggregator::AbstractAggregatorAlgorithm,
        coupling_map::Vector{Tuple{Int, Int}}; kwargs...)
    JumpProblem(cat_problems(prob.prob, prob_control.prob), aggregator,
        build_split_jumps(prob, prob_control, coupling_map)...; kwargs...)
end

# make new problem by joining initial_data
function cat_problems(prob::DiscreteProblem, prob_control::DiscreteProblem)
    u0_coupled = CoupledArray(prob.u0, prob_control.u0, true)
    DiscreteProblem(u0_coupled, prob.tspan, prob.p)
end

function cat_problems(prob::DiffEqBase.AbstractODEProblem,
        prob_control::DiffEqBase.AbstractODEProblem)
    l = length(prob.u0) # add l_c = length(prob_control.u0)

    _f = SciMLBase.unwrapped_f(prob.f)
    _f_control = SciMLBase.unwrapped_f(prob_control.f)

    new_f = function (du, u, p, t)
        _f(@view(du[1:l]), u.u, p, t)
        _f_control(@view(du[(l + 1):(2 * l)]), u.u_control, p, t)
    end
    u0_coupled = CoupledArray(prob.u0, prob_control.u0, true)
    ODEProblem(new_f, u0_coupled, prob.tspan, prob.p)
end

function cat_problems(prob::DiscreteProblem, prob_control::DiffEqBase.AbstractODEProblem)
    l = length(prob.u0) # add l_c = length(prob_control.u0)
    if !(prob.f isa typeof(DiffEqBase.DISCRETE_INPLACE_DEFAULT))
        @warn("Coupling to DiscreteProblem with nontrivial f. Note that, unless scale_by_time=true, the meaning of f will change when using an ODE/SDE/DDE/DAE solver.")
    end

    _f = SciMLBase.unwrapped_f(prob.f)
    _f_control = SciMLBase.unwrapped_f(prob_control.f)

    new_f = function (du, u, p, t)
        _f(@view(du[1:l]), u.u, p, t)
        _f_control(@view(du[(l + 1):(2 * l)]), u.u_control, p, t)
    end
    u0_coupled = CoupledArray(prob.u0, prob_control.u0, true)
    ODEProblem(new_f, u0_coupled, prob.tspan, prob.p)
end

function cat_problems(prob::DiffEqBase.AbstractSDEProblem,
        prob_control::DiffEqBase.AbstractSDEProblem)
    l = length(prob.u0)
    new_f = function (du, u, p, t)
        prob.f(@view(du[1:l]), u.u, p, t)
        prob_control.f(@view(du[(l + 1):(2 * l)]), u.u_control, p, t)
    end
    new_g = function (du, u, p, t)
        prob.g(@view(du[1:l]), u.u, p, t)
        prob_control.g(@view(du[(l + 1):(2 * l)]), u.u_control, p, t)
    end
    u0_coupled = CoupledArray(prob.u0, prob_control.u0, true)
    SDEProblem(new_f, new_g, u0_coupled, prob.tspan, prob.p)
end

function cat_problems(prob::DiffEqBase.AbstractSDEProblem,
        prob_control::DiffEqBase.AbstractODEProblem)
    l = length(prob.u0)

    _f = SciMLBase.unwrapped_f(prob.f)
    _f_control = SciMLBase.unwrapped_f(prob_control.f)

    new_f = function (du, u, p, t)
        _f(@view(du[1:l]), u.u, p, t)
        _f_control(@view(du[(l + 1):(2 * l)]), u.u_control, p, t)
    end
    new_g = function (du, u, p, t)
        prob.g(@view(du[1:l]), u.u, p, t)
        for i in (l + 1):(2 * l)
            du[i] = 0.0
        end
    end
    u0_coupled = CoupledArray(prob.u0, prob_control.u0, true)
    SDEProblem(new_f, new_g, u0_coupled, prob.tspan, prob.p)
end

function cat_problems(prob::DiffEqBase.AbstractSDEProblem, prob_control::DiscreteProblem)
    l = length(prob.u0)
    if !(prob_control.f isa typeof(DiffEqBase.DISCRETE_INPLACE_DEFAULT))
        @warn("Coupling to DiscreteProblem with nontrivial f. Note that, unless scale_by_time=true, the meaning of f will change when using an ODE/SDE/DDE/DAE solver.")
    end
    new_f = function (du, u, p, t)
        prob.f(@view(du[1:l]), u.u, p, t)
        prob_control.f(@view(du[(l + 1):(2 * l)]), u.u_control, p, t)
    end
    new_g = function (du, u, p, t)
        prob.g(@view(du[1:l]), u.u, p, t)
        for i in (l + 1):(2 * l)
            du[i] = 0.0
        end
    end
    u0_coupled = CoupledArray(prob.u0, prob_control.u0, true)
    SDEProblem(new_f, new_g, u0_coupled, prob.tspan)
end

function cat_problems(prob_control::DiffEqBase.AbstractODEProblem, prob::DiscreteProblem)
    cat_problems(prob, prob_control)
end
function cat_problems(prob_control::DiscreteProblem, prob::DiffEqBase.AbstractSDEProblem)
    cat_problems(prob, prob_control)
end
function cat_problems(prob_control::DiffEqBase.AbstractODEProblem,
        prob::DiffEqBase.AbstractSDEProblem)
    cat_problems(prob, prob_control)
end

# this only depends on the jumps in prob, not prob.prob
function build_split_jumps(prob::DiffEqBase.AbstractJumpProblem,
        prob_control::DiffEqBase.AbstractJumpProblem,
        coupling_map::Vector{Tuple{Int, Int}})
    num_jumps = length(prob.discrete_jump_aggregation.rates)
    num_jumps_control = length(prob_control.discrete_jump_aggregation.rates)
    jumps = []
    # overallocates, will fix later
    uncoupled = deleteat!(Vector(1:num_jumps), [c[1] for c in coupling_map])
    uncoupled_control = deleteat!(Vector(1:num_jumps_control), [c[2] for c in coupling_map])
    for c in uncoupled   # make uncoupled jumps in prob
        new_rate = prob.discrete_jump_aggregation.rates[c]
        new_affect! = prob.discrete_jump_aggregation.affects![c]
        push!(jumps, ConstantRateJump(new_rate, new_affect!))
    end
    for c in uncoupled_control  # make uncoupled jumps in prob_control
        rate = prob_control.discrete_jump_aggregation.rates[c]
        new_rate = (u, p, t) -> rate(u.u_control, p, t)
        affect! = prob_control.discrete_jump_aggregation.affects![c]
        new_affect! = function (integrator)
            flip_u!(integrator.u)
            affect!(integrator)
            flip_u!(integrator.u)
        end
        push!(jumps, ConstantRateJump(new_rate, new_affect!))
    end

    for c in coupling_map # make coupled jumps. 3 new jumps for each old one
        rate = prob.discrete_jump_aggregation.rates[c[1]]
        rate_control = prob_control.discrete_jump_aggregation.rates[c[2]]
        affect! = prob.discrete_jump_aggregation.affects![c[1]]
        affect_control! = prob_control.discrete_jump_aggregation.affects![c[2]]
        # shared jump
        new_affect! = function (integrator)
            affect!(integrator)
            flip_u!(integrator.u)
            affect_control!(integrator)
            flip_u!(integrator.u)
        end
        new_rate = (u, p, t) -> min(rate(u.u, p, t), rate_control(u.u_control, p, t))
        push!(jumps, ConstantRateJump(new_rate, new_affect!))
        # only prob
        new_affect! = affect!
        new_rate = (u, p, t) -> rate(u.u, p, t) -
                  min(rate(u.u, p, t), rate_control(u.u_control, p, t))
        push!(jumps, ConstantRateJump(new_rate, new_affect!))
        # only prob_control
        new_affect! = function (integrator)
            flip_u!(integrator.u)
            affect!(integrator)
            flip_u!(integrator.u)
        end
        new_rate = (u, p, t) -> rate_control(u.u_control, p, t) -
                  min(rate(u.u, p, t), rate_control(u.u_control, p, t))
        push!(jumps, ConstantRateJump(new_rate, new_affect!))
    end
    jumps
end
