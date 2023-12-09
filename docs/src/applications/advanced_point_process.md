# [Temporal Point Processes (TPP) with JumpProcesses] (@id tpp_tutorial)

JumpProcesses was initially developed to simulate the trajectory of jump
processes. Therefore, those with a background in point process might find the
nomenclature in the library documentation confusing. In reality, jump and point
processes share many things in common, but diverge in scope. This tutorial will
cover JumpProcesses from the perspective of point process theory.

Historically, jump processes have been developed in the context of dynamical
systems to describe dynamics with sudden changes — the jumps — in a system's
value at random times. In contrast, the development of point processes has been
more focused on describing the occurrence of random events — the points — over
a support. The fact that any temporal point process (TPP) that satisfies some
basic assumptions can be described in terms of a stochastic differential
equation (SDE) with discontinuous jumps — more commonly known as a jump process
— means TPPs can be simulated with JumpProcesses.

## [TPP Theory](@id tpp_theory)

TPPs describe a set of discrete points over continuous time. Conventionally, we
assume that time starts at ``0``. We can represent a TPP as a random integer
measure ``N( \cdot )``, this random function counts the number of points in
a set of intervals over the real line. For instance, ``N([5, 10])`` denotes the
number of points (or events) in between time ``5`` and ``10`` inclusive. The
number of points in this interval is a random variable. If ``N`` is a Poisson
process with conditional intensity (or rate) equal to ``1``, then ``N[5, 10]``
is distributed according to a Poisson distribution with parameter ``\lambda = 5``.

For convenience, we denote ``N(t) \equiv N[0, t)`` as the number of points since
the start of time until ``t``, exclusive of ``t``. A TPP is _simple_, if only a
single event can occur in any unit of time ``t``, that is, ``\Pr(N[t, t +
\epsilon) > 2) = 0``. We can then define a differential of ``N``, ``dN``, which
describes the change in ``N(t)`` over an infinitesimal amount of time.

```math
dN(t) = \begin{cases}
  1 \text{ , if } N[t, t + \epsilon] = 1 \\
  0 \text{ , if } N[t, t + \epsilon] = 0.
\end{cases}
```

Therefore, we can use any TPP to define a stochastic differential equations
(SDE) with discontinuous jumps.

```math
du = f(u,p,t)dt + g(u,p,t) dW(t) + h(u,p,t) dN(t)
```

In the jump literature, ``N`` is usually a Poisson process which allows for
a number of convenient properties for jump modelling. See Björk [^2] and Hanson [^3]
for more theoretical discussions of SDE with discontinuous jumps.

We can start implementing our TPP interface in Julia by working with the
following SDE with discontinuous jumps.

```math
du = dN(t)
```

In this case, ``u(t)`` is a monotonic function which counts the total number of
points since the start of time. Therefore, there is a one-to-one map between
``N`` and ``u``.

A TPP is marked if, in addition to the temporal support ``\mathbb{R}``, there is
a mark space ``\mathcal{K}`` such that ``N`` is a random integer measure over
``\mathbb{R} \times \mathcal{K}``. Intuitively, for every point in the process
there is a mark associated with it. If the mark space is discrete, we have
a multivariate TPP. There are different ways to interpret TPPs, and we will
move between these interpretations throughout the tutorial.

To make the connection between the JumpProcesses library and point process
theory, we will compare against the
[PointProcess.jl](https://github.com/gdalle/PointProcesses.jl) library, which
offers a common interface for marked TPPs. Since TPP sampling is more efficient
if we split any marked TPP into a sparsely connected multivariate TPP, we define
`SciMLPointProcess` as a multivariate TPP such that each sub-component is itself
a marked TPP on a continuous space. Therefore, we have that our structure
includes a vector of sub-TPP `jumps`, a vector of mark distributions `mark_dist`
and the sparsely connected graph `g`.

```@example tpp
using JumpProcesses
using PointProcesses

struct SciMLPointProcess{M, J <: JumpProcesses.AbstractJump, G, D, T <: Real} <:
       AbstractPointProcess{M}
    jumps::Vector{J}
    mark_dist::Vector{D}
    g::G
    p::Any
    tmin::T
    tmax::T
end

function Base.show(io::IO, pp::SciMLPointProcess)
    println(io,
        "SciMLPointProcess with $(length(pp.jumps)) processes on the interval [$(pp.tmin), $(pp.tmax)).")
end
```

In alignment with `PointProcesses` API we define methods for extracting the
boundaries of the time interval we will be working with.

```@example tpp
PointProcesses.min_time(pp::SciMLPointProcess) = pp.tmin
PointProcesses.max_time(pp::SciMLPointProcess) = pp.tmax
```

As we want to keep `SciMLPointProcess` as general as possible we define two
methods for initializing and resetting the parameters of the TPP. The usefulness
of these methods will become apparent further below.

```@example tpp
params(pp::SciMLPointProcess) = pp.p(pp)
params!(pp::SciMLPointProcess, p) = pp.p(pp, p)
nothing # hide
```

The likelihood of any _simple_ TPP is fully characterized by its conditional
intensity ``\lambda^\ast (t, k) \equiv \lambda^\ast(t) \times f^\ast (k \mid t)``, where ``f^\ast(k \mid t)`` is the mark distribution and
``\lambda^\ast(t)`` is the conditional intensity of the ground process which can
be further factorized as:

```math
\lambda^\ast (t) \equiv \lambda(t \mid H_{t^-} ) =
  \frac{p^\ast(t)}{1 - \int_{t^-}^{t_n} p^\ast(u) \, du},
```

The internal history of the process up to but not including ``t`` is ``H_{t^-} \equiv \{ (t_n, k_n) \mid 0 \leq t_n \leq t \}``. Following the convention in
the TPP literature, the superscript ``\ast`` denotes the conditioning of any
function on ``H_{t^-}``. So, ``p^\ast(t) \equiv p(t \mid H_{t^-})`` is the
density function corresponding to the probability of an event taking place at
time ``t`` given ``H_{t^-}``. The mark distribution denotes the density function
corresponding to the probability of observing mark ``k`` given the occurrence of
an event at time ``t`` and internal history ``H_{t^-}``. In summary, the
conditional intensity is the likelihood of observing a point in the next
infinitesimal unit of time, given that no point has occurred since the last
observed point in ``H_{t^-}``. Alternatively, we can interpret it as the number
of points that we expect to see in the next marginal interval ``E[dNt(dt \times d\mu(k))] = \lambda^\ast (t, k) dt d\mu(k)``. For more details, see Chapter 7,
Daley and Vere-Jones[^1].

PointProcesses provides a convenient interface for keeping track of the history
of a marked TPP called `History`.

```@example julia
mutable struct History{M, T <: Real}
    times::Vector{T}
    marks::Vector{M}
    tmin::T
    tmax::T
end
```

We define a method for resetting the history, which can be useful during simulation.

```@example tpp
function reset!(h::History)
    empty!(event_times(h))
    empty!(event_marks(h))
end
nothing # hide
```

## [Marked Hawkes Process](@id marked_hawkes)

Until now, everything has been fairly theoretical. But we have enough to start
walking through a concrete example. Throughout this tutorial we will implement
a case of the Hawkes process. Hawkes processes are classic TPP models that show
self-exciting behavior whereby the occurrence of an event increases the
likelihood of other events nearby. They are useful models to describe
earthquakes, gang violence, bank defaults, etc. For a complete treatment of
Hawkes processes see Laub, Lee and Taimre [^4].

In this tutorial we propose a spatial Hawkes process as a simple model for call
detail records (CDRs). CDR records the time and location when a user starts
a mobile data session. Behind the scenes, anytime a user sees a message from
someone in their network, they are more likely to use their phone. In addition
to that, users tend to gravitate around their home, so the locations they visit
are spread around their home according to a Gaussian distribution.

Formally, consider a graph ``G`` with ``V`` nodes. Our Hawkes process is
characterized by ``V`` TPPs such that the conditional intensity rate of node
``i`` connected to a set of nodes ``E_i`` in the graph ``G`` is given by:

```math
\lambda_i^\ast (t) = \lambda
  + \sum_{j \in E_i} \sum_{t_{n_j} < t} \alpha \exp \left[ -\beta (t - t_{n_j}) \right]
```

This conditional intensity is known as a self-exciting, because the occurrence
of an event ``j`` at ``t_{n_j}`` will increase the conditional intensity of all
processes connected to it by ``alpha``. This influence will then decrease at
rate ``\beta``.

The conditional intensity of this process has a recursive formulation which we
can use to our advantage to significantly speed simulation. Let ``t_{N_i} = \max \{t_{n_j} < t \mid j \in E_i\}`` and ``\phi_i^\ast(t)`` below.

```math
\begin{split}
  \phi_i^\ast (t)
    &= \sum_{j \in E_i} \sum_{t_{n_j} < t} \alpha \exp \left[-\beta (t - t_{N_i} + t_{N_i} - t_{n_j}) \right] \\
    &= \exp \left[ -\beta (t - t_{N_i}) \right]
      \sum_{j \in E_i} \sum_{t_{n_j} \leq t_{N_i}} \alpha \exp \left[-\beta (t_{N_i} - t_{n_j}) \right] \\
    &= \exp \left[ -\beta (t - t_{N_i}) \right] \left( \alpha + \phi_i^\ast (t_{N_i}) \right)
\end{split}
```

Then the conditional intensity can be re-written in terms of ``\phi_i^\ast (t_{N_i})``.

```math
\lambda_i^\ast (t) = \lambda + \phi_i^\ast (t) =
    \lambda + \exp \left[ -\beta (t - t_{N_i}) \right] \left( \alpha + \phi_i^\ast (t_{N_i}) \right)
```

We translate these expressions to Julia by employing a closure which will allow
us to obtain the rate for each node ``i`` in ``G``.

```@example tpp
function hawkes_rate(i::Int, g)
    function rate(u, p, t)
        (; λ, α, β, ϕ, T) = p
        return λ + exp(-β * (t - T[i])) * ϕ[i]
    end
    return rate
end
nothing # hide
```

We assume that each sup-process `i` is a marked TPP whose mark is drawn from
a multivariate Gaussian distribution with mean ``\mu_i`` and standard deviation
equal to ``I``. Therefore, the mark distribution can be represented as a mixture
distribution.

```math
f^\ast(k \mid t) = \frac{\lambda_i^\ast (t)}{\sum_i \lambda_i^\ast (t)}
  \frac{1}{\sqrt{2\pi}} \exp \left[ -1/2 (m - \mu_i)^\top(m - \mu_i) \right]
```

In Julia, we define a method for constructing Hawkes jumps. JumpProcesses define
different types of jumps that vary according to the behavior of the conditional
intensity and the intended simulation algorithm. Since the conditional intensity
of our Hawkes process is not fixed, we will use `VariableRateJump` to construct
the Hawkes jumps. The structure requires a `rate` which we defined above and an
`affect!` which tells the program what happens when a jump occurs and is when we
draw the marks from our distribution. In addition to that, since we intend to
use the `Coevolve` algorithm for simulation — see below —, we need to define the
rate upper-bound, the interval for which the upper-bound is valid, and,
optionally, a lower-bound for improved simulation efficiency.

```@example tpp
function hawkes_jump(i::Int, g, mark_dist)
    rate = hawkes_rate(i, g)
    urate = rate
    lrate(u, p, t) = p.λ
    rateinterval = (u, p, t) -> begin
        _lrate = lrate(u, p, t)
        _urate = urate(u, p, t)
        return _urate == _lrate ? typemax(t) : 1 / (2 * _urate)
    end
    function affect!(integrator)
        (; λ, α, β, ϕ, T, h) = integrator.p
        for j in g[i]
            ϕ[j] = α + exp(-β * (integrator.t - T[j])) * ϕ[j]
            T[j] = integrator.t
        end
        m = rand(mark_dist[i])
        push!(h, integrator.t, (i, m); check = false)
    end
    return VariableRateJump(rate, affect!; lrate, urate, rateinterval)
end
nothing # hide
```

To initialize `SciMLPointProcess` we also need to define a function for
returning and resetting the parameters of our model.

```@example tpp
function hawkes_p(pp::SciMLPointProcess{M, J, G, D, T}) where {M, J, G, D, T}
    g = pp.g
    tmin = pp.tmin
    tmax = pp.tmax
    h = History(; times = T[], marks = Tuple{Int, M}[], tmin = tmin, tmax = tmax)
    return (λ = 0.5,
        α = 0.1,
        β = 2.0,
        ϕ = zeros(T, length(g)),
        T = zeros(T, length(g)),
        h = h)
end

function hawkes_p(pp::SciMLPointProcess{M, J, G, D, T}, p) where {M, J, G, D, T}
    reset!(p.h)
    p.ϕ .= zero(p.ϕ)
    p.T .= zero(p.T)
end

nothing # hide
```

Now, we are ready to initialze our `SciMLPointProcess` as a Hawkes process.

```@example tpp
using Graphs
using Distributions
V = 10
G = erdos_renyi(V, 0.2)
g = [neighbors(G, i) for i in 1:nv(G)]
mark_dist = [MvNormal(rand(2), [0.1, 0.1]) for i in 1:nv(G)]
jumps = [hawkes_jump(i, g, mark_dist) for i in 1:nv(G)]
tspan = (0.0, 50.0)
hawkes = SciMLPointProcess{
    Vector{Real},
    eltype(jumps),
    typeof(g),
    eltype(mark_dist),
    eltype(tspan),
}(jumps,
    mark_dist,
    g,
    hawkes_p,
    tspan[1],
    tspan[2])
```

## [Sampling](@id tpp_sampling)

JumpProcesses shines in the simulation of SDEs with discontinuous jumps. The
mapping we introduced in the [previous Section](@ref tpp_theory) whereby ``du = dN(t)`` implies that JumpProcesses also excels in simulating TPPs.

JumpProcesses offers a plethora of simulation algorithms for TPPs. The library
call them _aggregators_ because these algorithms are methods for aggregating
a set of jumps to determine the next jump time. In [Jump Aggregators for Exact
Simulation](@ref), we discuss the trade-off between different simulation
algorithms.

To simulate a `SciMLPointProcess`, we start by overloading `Base.rand`. In our
implementation, we initialize a `JumpProblem` with the jumps and parameters
passed to the `SciMLPointProcess` as well as with the desired simulation
algorithm. In this tutorial we use the `Coevolve` _aggregator_ which is an
algorithm inspired by Ogata's algorithm for bounded TPPs with modifications to
improve efficiency and to allow for the concurrent evolution of the TPP and
simulation time.

Finally, we sample a path from our `JumpProblem` using the `SSAStepper`.
A stepper tells the solver how to step through time. When simulating TPPs, we do
not need to evolve time incrementally by small deltas. The `SSAStepper` allow us
to step through time one candidate at a time.

```@example tpp
using OrdinaryDiffEq
using Random

function Base.rand(rng::AbstractRNG, pp::SciMLPointProcess)
    return rand(rng, pp, min_time(pp), max_time(pp), 1)[1]
end

function Base.rand(rng::AbstractRNG, pp::SciMLPointProcess, n::Int)
    return rand(rng, pp, min_time(pp), max_time(pp), n)
end

function Base.rand(pp::SciMLPointProcess, n::Int)
    return rand(Random.default_rng(), pp, min_time(pp), max_time(pp), n)
end

function Base.rand(rng::AbstractRNG,
        pp::SciMLPointProcess{M, J, G, D, T},
        tmin::T,
        tmax::T,
        n::Int) where {M, J, G, D, T <: Real}
    tspan = (tmin, tmax)
    save_positions = (false, false)
    out = Array{History, 1}(undef, n)
    p = params(pp)
    dprob = DiscreteProblem([0], tspan, p)
    jprob = JumpProblem(dprob, Coevolve(), jumps...; dep_graph = pp.g, save_positions, rng)
    for i in 1:n
        params!(pp, p)
        solve(jprob, SSAStepper())
        out[i] = deepcopy(p.h)
    end
    return out
end
```

We can easily sample the Hawkes process introduced in the previous section.

```@example tpp
h = rand(hawkes)
```

It would be useful to visualize our sampled points to get a good feeling of our
simulation. Since PointProcesses does not offer recipes for visualizing TPP
samples, we propose a few options. First, we visualize our sample as a sequence
of points though time, each line represents the realization of one of the
sub-TPPs of `SciMLPointProcess`.

```@example tpp
using Plots
@userplot BarcodePlot
@recipe function f(x::BarcodePlot)
    h, ix = x.args
    times = event_times(h)
    marks = event_marks(h)
    histories = [times[filter(n -> marks[n][1] == i, 1:length(times))] for i in ix]
    seriestype := :scatter
    yticks --> (1:length(ix), ix)
    alpha --> 0.5
    markerstrokewidth --> 0
    markerstrokealpha --> 0
    markershape --> :circle
    ylims --> (0.0, length(ix) + 1.0)
    for i in 1:length(ix)
        @series begin
            label := ix[i]
            histories[i], fill(i, length(histories[i]))
        end
    end
end
```

Lets visualize our data as a barcode.

```@example tpp
barcodeplot(h, 1:10; legend = false)
```

In reality, we do not observe which components generated which points.
Therefore, a fairer representation of the data would be as a scatter plot. We
also plot the latent home locations which show that points are indeed clustered
around them.

```@example tpp
scatter(map(m -> tuple(m[2]...), event_marks(h)), label = "events")
scatter!(map(d -> tuple(mean(d)...), mark_dist), label = "cluster epicentre")
```

## [The Conditional Intensity](@id tpp_conditional_intensity)

The first Section of this tutorial introduced the conditional intensity. Here we
will delve deeper into this concept which plays a prominent role in TPP theory
due to its ability to fully characterize any TPP.

The conditional intensity is a stochastic process since it depends on the
realization of ``N``. Given a history of events ``H_{t^-}``, we can recover the
conditional intensity of any jump by evaluating its rate function. We can take
advantage of OrdinaryDiffEq to build a `DiscreteProblem` which we solve with the
`FunctionMap` stepper, which steps through time stopping on each event and on
specific locations specified by the user. Our method computes the conditional
intensity for each sub-TPP of the `SciMLPointProcess`, it returns the solution
of the `DiscreteProblem` and the final parameters.

```@example tpp
function intensity(pp::SciMLPointProcess, t, h; saveat = [], save_positions = (true, true))
    p = params(pp)
    times = event_times(h)
    marks = event_marks(h)
    tmin, tmax = min_time(pp), t
    tstops = typeof(saveat) <: Number ? collect(tmin:saveat:tmax) : copy(saveat)
    append!(tstops, times)
    sort!(tstops)
    unique!(tstops)
    rates(u, p, t) = [jump.rate(nothing, p, t) for jump in pp.jumps]
    condition(u, t, integrator) = t ∈ times
    function affect!(integrator)
        n = searchsortedfirst(times, integrator.t)
        mark = marks[n]
        ix = mark[1]
        pp.jumps[ix].affect!(integrator)
        # overwrite history with true mark
        integrator.p.h.marks[n] = mark
        integrator.u = rates(integrator.u, integrator.p, integrator.t)
    end
    callback = DiscreteCallback(condition, affect!; save_positions)
    dprob = DiscreteProblem(rates, rates(nothing, p, tmin), (tmin, tmax), p; callback,
        tstops, saveat)
    sol = solve(dprob, FunctionMap())
    return sol
end
nothing # hide
```

As an illustration, we plot the conditional intensity of the first sub-TPP of
our sampled Hawkes process.

```@example tpp
λ = intensity(hawkes, max_time(hawkes), h; saveat = 0.1)
plot(λ, idxs = 1)
```

Now, we can specialize a number of functions required by the interface defined
in PointProcesses.

```@example tpp
function ground_intensity(pp::SciMLPointProcess, t, h)
    λ = intensity(pp, t, h; saveat = [t], save_positions = (false, false))
    ground_intensity(pp, t, λ)
end

function ground_intensity(pp::SciMLPointProcess, t, λ::ODESolution)
    return vec(sum(λ(t), dims = 1))
end

function ground_intensity(pp::SciMLPointProcess, t::T, λ::ODESolution) where {T <: Number}
    return sum(λ(t))
end

function mark_distribution(pp::SciMLPointProcess, t, h)
    λ = intensity(pp, t, h)
    mark_distribution(pp, t, λ)
end

function mark_distribution(pp::SciMLPointProcess, t, λ)
    λt = λ(t)
    d = MixtureModel(pp.mark_dist, λt ./ sum(λt))
    return d
end

function intensity(pp::SciMLPointProcess, m, t, h)
    λ = intensity(pp, t, h)
    intensity(pp, m, t, λ)
end

function intensity(pp::SciMLPointProcess, m, t, λ::ODESolution)
    λt = λ(t)
    return sum([densityof(pp.mark_dist[i], m) * λt[i] for i in 1:length(pp.jumps)])
end

function log_intensity(pp::SciMLPointProcess, m, t, h)
    return log(intensity(pp, m, t, h))
end

nothing # hide
```

We can then visualize the ground intensity.

```@example tpp
plot(ground_intensity(hawkes, λ.t, λ))
```

## [The Compensator](@id tpp_compensator)

The compensator is defined as the integral of the conditional intensity.

```math
\Lambda(t, k) \equiv \int_0^t \lambda^\ast (u, k) du
```

With some abuse of notation, we obtain the compensator of the ground process by
integrating over the marks.

```math
\Lambda(t) \equiv \int_0^t \sum_{k \in \mathcal{K}} \lambda^\ast (u, k) du
```

In Julia we can derive the compensator by simply integrating over the
conditional intensity using an ODEProblem.

```@example tpp
function integrated_intensity(pp::SciMLPointProcess,
        t,
        h;
        alg = nothing,
        saveat = [],
        save_positions = (true, true))
    p = params(pp)
    times = event_times(h)
    marks = event_marks(h)
    tspan = (min_time(pp), t)
    rates(u, p, t) = [jump.rate(nothing, p, t) for jump in pp.jumps]
    condition(u, t, integrator) = t ∈ times
    function affect!(integrator)
        n = searchsortedfirst(times, integrator.t)
        mark = marks[n]
        ix = mark[1]
        pp.jumps[ix].affect!(integrator)
        integrator.p.h.marks[n] = mark
    end
    callback = DiscreteCallback(condition, affect!; save_positions)
    prob = ODEProblem(rates,
        zeros(eltype(times), length(pp.jumps)),
        tspan,
        p;
        tstops = times,
        callback,
        saveat)
    sol = solve(prob, alg)
    return sol
end

nothing # hide
```

The PointProcess interface expects that we define a method for computing the
compensator of the ground process for a given interval. This will be useful in
the next section when we compute the log-likelihood.

```@example tpp
function integrated_ground_intensity(pp::SciMLPointProcess, h, a, b)
    Λ = integrated_intensity(pp,
        b,
        h;
        alg = Rodas4P(),
        saveat = [a, b],
        save_positions = (false, false))
    return sum(Λ(b)) - sum(Λ(a))
end

nothing # hide
```

In practice, we must pay close attention to the shape of the conditional
intensity. In the Hawkes case, the conditional intensity is highly stiff as
events will cause the intensity to spike. Therefore, we must select an ODE
solver that can deal with stiff problems like `Rodas4P()`.

```@example tpp
Λ = integrated_intensity(hawkes,
    max_time(hawkes),
    h;
    saveat = event_times(h),
    alg = Rodas4P())
plot(Λ)
```

The compensator of the Hawkes process lends itself to an analytical solution
which can be implemented in Julia as a `DiscreteProblem` much like the
conditional intensity above.

```@example tpp
function hawkes_integrated_intensity(pp::SciMLPointProcess, t, h; saveat = [])
    (; λ, α, β) = params(pp)
    p = (λ = λ, α = α, β = β, h = h)
    saveat = typeof(saveat) <: Number ? collect(min_time(h):saveat:t) : copy(saveat)
    tstops = copy(event_times(h))
    function compensator(u, p, t)
        (; λ, α, β, h) = p
        u = zeros(typeof(t), length(pp.jumps))
        for (i, Ei) in enumerate(pp.g)
            u[i] += λ * t
            for (_t, (_j, _)) in zip(event_times(h), event_marks(h))
                _t >= t && break
                _j ∉ Ei && continue
                u[i] += (α / β) * (1 - exp(-β * (t - _t)))
            end
        end
        return u
    end
    dprob = DiscreteProblem(compensator, zeros(eltype(event_times(h)), length(pp.jumps)),
        (min_time(h), t), p; tstops, saveat)
    sol = solve(dprob, FunctionMap())
    return sol
end
```

```@example tpp
Λ_exact = hawkes_integrated_intensity(hawkes, max_time(hawkes), h)
plot!(Λ_exact)
```

Time-change theorems for TPPs show that the compensator transforms a complex TPP
into a Poisson process with unit rate. We can interpret the compensator as
making the TPP as random as possible. We can check that this is indeed the case
for our model and simulation algorithm.

First, we define a method for filtering the history.

```@example tpp
function Base.filter(f, h::History)
    times = event_times(h)
    marks = event_marks(h)
    filtered_times = eltype(times)[]
    filtered_marks = eltype(marks)[]
    for (t, mark) in zip(times, marks)
        if f(t, mark)
            push!(filtered_times, t)
            push!(filtered_marks, mark)
        end
    end
    return History(;
        times = filtered_times,
        marks = filtered_marks,
        tmin = min_time(h),
        tmax = max_time(h))
end
```

And a recipe for QQ-plots.

```@example tpp
@userplot QQPlot
@recipe function f(x::QQPlot)
    empirical_quant, expected_quant = x.args
    max_empirical_quant = maximum(maximum, empirical_quant)
    max_expected_quant = maximum(expected_quant)
    upperlim = ceil(maximum([max_empirical_quant, max_expected_quant]))
    @series begin
        seriestype := :line
        linecolor := :lightgray
        label --> ""
        (x) -> x
    end
    @series begin
        seriestype := :scatter
        aspect_ratio := :equal
        xlims := (0.0, upperlim)
        ylims := (0.0, upperlim)
        xaxis --> "Expected quantile"
        yaxis --> "Empirical quantile"
        markerstrokewidth --> 0
        markerstrokealpha --> 0
        markersize --> 1.5
        size --> (400, 500)
        label --> permutedims(["quantiles $i" for i in 1:length(empirical_quant)])
        expected_quant, empirical_quant
    end
end
```

To produce the QQ-plots, we ran the simulation a reasonable number of times to
ensure an accurate estimation of the empirical quantiles of the interarrival
times. The `time_change` method comes from PointProcess which applies the
compensator to the input history. We take advantage of the fact that the
compensator is an `ODESolution` which overloads an interpolation method in
itself to produce the QQ-plot for the ground process.

```@example tpp
Δt̃ = []
for _ in 1:250
    _h = rand(hawkes)
    _Λ = integrated_intensity(hawkes,
        max_time(hawkes),
        _h;
        saveat = event_times(_h),
        alg = Rodas4P())
    _h̃ = time_change(_h, (t) -> sum(_Λ(t)))
    append!(Δt̃, diff(event_times(_h̃)))
end
empirical_quant = quantile(Δt̃, 0.01:0.01:0.99)
expected_quant = quantile(Exponential(1.0), 0.01:0.01:0.99)
qqplot(empirical_quant, expected_quant; legend = false)
```

Likewise, we can produce the QQ-plot for each sub-TPP.

```@example tpp
Δt̃ = [[] for _ in 1:nv(G)]
for _ in 1:250
    _h = rand(hawkes)
    _Λ = integrated_intensity(hawkes,
        max_time(hawkes),
        _h;
        saveat = event_times(_h),
        alg = Rodas4P())
    for i in 1:nv(G)
        _h̃ = time_change(filter((t, mark) -> mark[1] == i, _h), (t) -> Λ(t; idxs = i))
        append!(Δt̃[i], diff(event_times(_h̃)))
    end
end
empirical_quant = map(t -> quantile(t, 0.01:0.01:0.99), Δt̃)
expected_quant = quantile(Exponential(1.0), 0.01:0.01:0.99)
qqplot(empirical_quant, expected_quant)
```

## [The Log-Likelihood](@id tpp_loglikelihood)

Once we know how to compute the conditional intensity and the compensator, it is
very easy to obtain the log-likelihood of a TPP because it is simply the sum of
the log-conditional intensity at event times minus the compensator up to the end
of the observed time. For a full derivation, see Chapter 7, Daley and
Vere-Jones[^1].

```math
\ell (H_{T^-}) = \sum_{n=1}^N \left( \log \lambda^\ast(t_n, k_n) \right)
  - \int_0^T \sum_{k \in \mathcal{K}} \lambda^\ast (u, k) du
```

In Julia, we follow the PointProcess API to define a method for computing the log-likelihood.

```@example tpp
using DensityInterface

function DensityInterface.logdensityof(pp::SciMLPointProcess, h)
    T = max_time(pp)
    times = event_times(h)
    marks = event_marks(h)
    Λ = integrated_ground_intensity(pp, h, min_time(h), T)
    λ = intensity(pp, T, h; saveat = times, save_positions = (false, false))
    logλ = 0
    for (t, (i, m)) in zip(times, marks)
        logλ += log_intensity(pp, m, t, λ)
        @info log_intensity(pp, m, t, λ), intensity(pp, m, t, λ)
    end
    return logλ - Λ
end
```

This allow us to compute the log-likelihood of our sample.

```@example tpp
logdensityof(hawkes, h)
```

This tutorial demonstrated the versatility of the SciML library to model TPPs.
We made extensive use of JumpProcesses and OrdinaryDiffEq to fulfill the
interface specified by the PointProcess library. However, we have left out
a discussion about fitting the parameters of a TPP to the data. In many cases,
we cannot derive an analytical estimator via maximum likelihood and we must tap
into machine learning tools like gradient descent and expectation-maximization
algorithms to minimize the log-likelihood function. Julia is a great tool for
these kind of tasks with libraries such as Optimization and Flux which are
outside of the scope this tutorial.

## References

[^1]: D. J. Daley and D. Vere-Jones, An Introduction to the Theory of Point
    Processes: Volume I: Elementary Theory and Methods, Springer-Verlag
    (2003). doi:10.1007/b97277.
[^2]: T., Björk, Point Processes and Jump Diffusions: An Introduction with
    Finance Applications, Cambridge University Press
    (2021). doi:10.1017/9781009002127.
[^3]: F. B. Hanson, Applied Stochastic Processes and Control for
    Jump-Diffusions: Modeling, Analysis and Computation, Society for
    Industrial and Applied Mathematics (2007). doi:10.1137/1.9780898718638.
[^4]: P. J. Laub, Y. Lee and T. Taimre, The Elements of Hawkes Processes,
    Springer International Publishing (2021). doi:10.1007/978-3-030-84639-8.
