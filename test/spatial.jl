using Plots
using DataFrames
using ModelingToolkit
using DiffEqBase
using DiffEqJump
using LightGraphs
#using SpecialGraphs

const nedgess = 300
const nvertss = 50

const dgs = DiGraph(nvertss, nedgess);
# Degree of each node: 2nd argument  5
const dgr = random_regular_digraph(nvertss, 5)

function setupSystem(graph)
    nverts = length(vertices(graph))
    nedges = length(edges(graph))
    # Allow for different beta on each edgse
    @parameters t β[1:nedges]  γ[1:nverts] ;
    @variables S[1:nverts](t);
    @variables I[1:nverts](t);
    @variables R[1:nverts](t);

    rxsS = [Reaction(β[i],[I[src(e)], S[dst(e)]], [I[dst(e)]], [1,1], [2])
        for (i,e) ∈ enumerate(edges(graph))]

    rxsI = [Reaction(γ[v],[I[v]], [R[v]])  # note: src to src, yet there is no edges
        for v ∈ vertices(graph)]

    rxs = vcat(rxsS, rxsI);
    vars = vcat(S,I,R);
    params = vcat(β,γ);

    rs = ReactionSystem(rxs, t, vars, params);
    js = convert(JumpSystem, rs);
    println("Completed: convert(JumpSystem)")
    S0 = ones(nverts)
    I0 = zeros(nverts)
    R0 = zeros(nverts)

    S0[1] = 0.; # One person is infected
    I0[1] = 1.;
    R0[1] = 1. - S0[1] - I0[1]
    vars0 = vcat(S0, I0, R0);

    # Two column vectors
    γ = fill(0.25, nverts);
    β = fill(0.50, nedges);
    params = vcat(β,γ)

    initial_state = [convert(Variable,state) => vars0[i] for (i,state) in enumerate(states(js))];
    initial_params = [convert(Variable,par) => params[i] for (i,par) in enumerate(parameters(js))];

    tspan = (0.0,20.0)
    @time dprob = DiscreteProblem(js, initial_state, tspan, initial_params)
    println("Completed: DiscreteProblem")
    @time jprob = JumpProblem(js, dprob, NRM())
    println("Completed: JumpProblem")
    @time sol = solve(jprob, SSAStepper())
    println("Completed: solve")

    return sol
end


function processData(sol)
    nverts = nvertss
    nedges = nedgess
    println("nverts=$nverts, nedges= $nedges")

    dfs = convert(Matrix, DataFrame(sol))
    Sf = dfs[1:nverts,:]
    If = dfs[nverts+1:2*nverts,:]
    Rf = dfs[2*nverts+1:3*nverts,:]
    Savg = (sum(Sf; dims=1)') / nverts
    Iavg = (sum(If; dims=1)') / nverts
    Ravg = (sum(Rf; dims=1)') / nverts
    print(Savg)
    return Savg, Iavg, Ravg
end

# Times: sol.t
# Solution at nodes: sol.u
# sol.u[1] |> length == 120 (3 * nverts)

sol = setupSystem(dgr)
Savg, Iavg, Ravg = processData(sol)

plot(sol.t, Savg)
plot(sol.t, Iavg)
plot(sol.t, Ravg)
