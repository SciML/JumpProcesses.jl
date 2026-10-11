# [Ensembles, `remake`, and problem reuse](@id ensembles_problem_reuse)

Stochastic simulations are usually run many times, so it is worth knowing how to reuse
a `JumpProblem` efficiently and how to run simulations in parallel safely. This tutorial
explains the rules that JumpProcesses follows and shows patterns that balance
performance and safety.

The key fact: a [`JumpProblem`](@ref) stores **mutable jump state**, namely the jump
aggregator and its callbacks. Solvers owned by JumpProcesses ([`SSAStepper`](@ref)
and the ODE/DAE pathways) reuse this state directly and re-initialize it at the start
of every solve. They never copy it. This makes repeated solves fast, but it means that
**one `JumpProblem` supports one active solve or integrator at a time**:

  - Repeated serial solves of one problem are safe and fast.
  - Concurrent solves, from threads or tasks, need independent copies of the problem.
  - Problems created with `remake` share the original's jump state.
  - `EnsembleProblem`s create the copies they need for you.

We use a simple birth-death process, ``\emptyset \to X`` at rate ``p_1`` and
``X \to \emptyset`` at rate ``p_2 X``, throughout:

```@example reuse
using JumpProcesses, SciMLBase

maj = MassActionJump([[0 => 1], [1 => 1]], [[1 => 1], [1 => -1]]; param_idxs = [1, 2])
dprob = DiscreteProblem([10], (0.0, 100.0), [1.0, 0.1])
# only save the initial and final states
jprob = JumpProblem(dprob, Direct(), maj; save_positions = (false, false))
nothing # hide
```

## Repeated serial solves

Solving the same problem repeatedly in a loop is the fastest way to generate many
samples: each solve re-initializes the problem's jump state from the problem's
current initial condition and parameters, so nothing needs to be copied or rebuilt.
No seed is needed either. Each solve draws from Julia's default random number
generator, whose state advances from one solve to the next:

```@example reuse
using Random # hide
Random.seed!(1) # hide
final_counts = [solve(jprob, SSAStepper()).u[end][1] for _ in 1:1000]
sum(final_counts) / length(final_counts)
```

To make the samples reproducible, create one generator and pass it to every solve.
Each solve continues the generator's stream, so the samples still differ:

```@example reuse
using Random
rng = Xoshiro(1234)
final_counts = [solve(jprob, SSAStepper(); rng).u[end][1] for _ in 1:1000]
sum(final_counts) / length(final_counts)
```

Do not create a new generator with the same seed for each solve, for example
`solve(jprob, SSAStepper(); rng = Xoshiro(1234))` inside the loop: every solve would
then follow the same path.

## Parameter sweeps with `remake`

`remake` creates a problem with a new initial condition, parameters, or time span.
The new problem **shares** the original's jump state rather than copying it, which
makes `remake` cheap:

```@example reuse
remade = remake(jprob; p = [2.0, 0.1])
remade.discrete_jump_aggregation === jprob.discrete_jump_aggregation
```

Since every solve re-initializes the shared state from the problem being solved, remade
problems can be solved one after another:

```@example reuse
Random.seed!(2) # hide
mean_counts = map([0.5, 1.0, 2.0]) do birth_rate
    prob = remake(jprob; p = [birth_rate, 0.1])
    sum(solve(prob, SSAStepper()).u[end][1] for _ in 1:500) / 500
end
```

However, the original and remade problems must not be solved concurrently, and must
not be used for integrators that are alive at the same time. Use `deepcopy` when you
need an independent problem.

## Ensembles

An [`EnsembleProblem`](https://docs.sciml.ai/DiffEqDocs/stable/features/ensemble/)
handles the copying needed to run trajectories safely:

  - `EnsembleSerial()` with `safetycopy = false` (the default when no `prob_func` is
    given) reuses the problem sequentially and makes no copies.
  - `EnsembleThreads()` with `safetycopy = false` copies the problem once per spawned
    task, and each task reuses its copy for its trajectories. With one thread, or a
    batch of one trajectory, it falls back to the serial path and uses the original
    problem.
  - With `safetycopy = true` (the default when a `prob_func` is given), every
    trajectory solves its own copy, made **before** `prob_func` is called.

```@example reuse
ensemble = EnsembleProblem(jprob)
esol = solve(ensemble, SSAStepper(), EnsembleThreads(); trajectories = 1000)
length(esol.u)  # the trajectories
```

### Custom `prob_func`s

A `prob_func` lets each trajectory use a different problem. A `prob_func` that only
modifies the problem it is given, for example with `remake`, is safe without safety
copies, because with `EnsembleThreads()` the problem it is given is already the copy
owned by the current task. Turning off the safety copies then avoids one copy per
trajectory:

```@example reuse
prob_func(prob, ctx) = remake(prob; p = [0.5 + 0.001 * ctx.sim_id, 0.1])
with_copies = EnsembleProblem(jprob; prob_func)  # safetycopy = true by default
without_copies = EnsembleProblem(jprob; prob_func, safetycopy = false)

function run_ensemble(ensemble)
    solve(ensemble, SSAStepper(), EnsembleThreads();
        trajectories = 2000)
end
run_ensemble(with_copies), run_ensemble(without_copies)  # compile before timing
(; with_copies = @elapsed(run_ensemble(with_copies)),
    without_copies = @elapsed(run_ensemble(without_copies)))
```

A `prob_func` must only derive its result from the problem it receives. Returning
some other, shared problem defeats the ensemble's copies:

```julia
# Unsafe with EnsembleThreads: uses the global `jprob`, not the task's copy `prob`
bad_prob_func(prob, ctx) = remake(jprob; p = [0.5 + 0.001 * ctx.sim_id, 0.1])
```

### Stateful callbacks

Callbacks passed as keyword arguments to `solve` are shared by every trajectory of an
ensemble, since ensembles copy only the problem. A callback that stores its own mutable
state should instead be passed to the `JumpProblem`, so that the ensemble's problem
copies include it, and should reset its state in its `initialize` function, so that
reusing it for the next trajectory starts fresh. For example, a callback that stops a
simulation after a given number of jumps:

```@example reuse
function stop_after_jumps(n)
    njumps = Ref(0)  # callback state
    reset!(cb, u, t, integrator) = (njumps[] = 0)
    function affect!(integrator)
        njumps[] += 1
        njumps[] >= n && terminate!(integrator)
    end
    DiscreteCallback((u, t, integrator) -> true, affect!; initialize = reset!)
end

birth = MassActionJump([[0 => 1]], [[1 => 1]]; param_idxs = [1])
birth_prob = DiscreteProblem([0], (0.0, 100.0), [1.0])
counting_jprob = JumpProblem(birth_prob, Direct(), birth; callback = stop_after_jumps(5))
esol = solve(EnsembleProblem(counting_jprob), SSAStepper(), EnsembleThreads();
    trajectories = 100)
all(sol.u[end] == [5] for sol in esol.u)
```

Passing `callback = stop_after_jumps(5)` to `solve` instead would share one counter
between all threads.

## Manual threading

When launching tasks yourself, give each task its own copy of the problem, made once
per task and reused for all of that task's solves:

```@example reuse
nsims = 1000
chunks = Iterators.partition(1:nsims, cld(nsims, Threads.nthreads()))
tasks = map(chunks) do chunk
    Threads.@spawn begin
        local_prob = deepcopy(jprob)  # one independent copy per task
        [solve(local_prob, SSAStepper()).u[end][1] for _ in chunk]
    end
end
final_counts = reduce(vcat, fetch.(tasks))
length(final_counts)
```

Never solve one problem, or problems created from it with `remake`, from several tasks
at the same time. For the same reason, integrators that are alive at the same time need
independent problems:

```@example reuse
Random.seed!(3) # hide
integrator1 = init(deepcopy(jprob), SSAStepper())
integrator2 = init(deepcopy(jprob), SSAStepper())
step!(integrator1)
step!(integrator2)
integrator1.t, integrator2.t
```

## Distributed ensembles

`EnsembleDistributed()` sends the problem to each worker. With `safetycopy = false` and
`pmap_batch_size > 1`, which is the default for 200 or more trajectories, a worker may
run several trajectories as concurrent tasks that share its copy of the problem. To be
safe, use `safetycopy = true` or `pmap_batch_size = 1`:

```julia
using Distributed
esol = solve(EnsembleProblem(jprob; safetycopy = true), SSAStepper(),
    EnsembleDistributed(); trajectories = 10_000)
```

## Aliasing `SSAStepper` inputs

Independently of the jump state, `SSAStepper` supports the common SciML `alias` keyword
argument for its `u0`, `p`, and `tstops` inputs. By default it copies `u0`, reuses `p`,
and never modifies a `tstops` array you pass. For example, a callback that modifies
parameters changes the problem's own parameter vector by default. Passing
`alias_p = false` gives the solve its own copy:

```@example reuse
stop_births = DiscreteCallback((u, t, integrator) -> t == 50.0,
    function (integrator)
        integrator.p[1] = 0.0
        reset_aggregated_jumps!(integrator)
    end)
sol = solve(jprob, SSAStepper(); callback = stop_births, tstops = [50.0],
    alias = SciMLBase.DiscreteAliasSpecifier(alias_p = false))
jprob.prob.p
```

Permitting `u0` to be aliased avoids a copy, but the solve then mutates the problem's
`u0`, so the next solve starts from the final state of the previous one:

```@example reuse
Random.seed!(4) # hide
prob_u0 = deepcopy(jprob)
solve(prob_u0, SSAStepper(); alias = SciMLBase.DiscreteAliasSpecifier(alias_u0 = true))
prob_u0.prob.u0
```

Use an `ODEAliasSpecifier` to also control `tstops`. `alias` never controls the jump
state; even `alias = false` reuses the problem's jump aggregator.

## SDE and RODE problems

StochasticDiffEq's SDE and RODE solvers control jump-state copying themselves, through
the `alias_jumps` field of their alias specifier. When it is unspecified they reuse the
jump state on thread 1 and copy it on other threads. To request a copy explicitly:

```julia
using StochasticDiffEqHighOrder
sol = solve(sde_jprob, SRIW1(); alias = SciMLBase.SDEAliasSpecifier(alias_jumps = false))
```

The guidance above still applies: do not rely on these copies to make shared problems
safe for concurrent use.

## Reproducibility with `SortingDirect`

[`SortingDirect`](@ref) learns a search order for the jumps during a solve and keeps it
for later solves of the same problem (including remade problems), which speeds up many
consecutive simulations. Sampling remains exact, but a given seed reproduces a
trajectory only from the same learned order. To replay a simulation exactly, start from
a copy of a problem that has never been solved:

```@example reuse
sd_jprob = JumpProblem(dprob, SortingDirect(), maj; save_positions = (false, false))
fresh = deepcopy(sd_jprob)  # never solved
replay(seed) = solve(deepcopy(fresh), SSAStepper(); seed).u
replay(42) == replay(42)
```

Seeded ensembles using `SortingDirect` are also affected: exact replay requires the same
starting learned order, seeds, `batch_size`, thread count, and ensemble algorithm, and
serial and threaded runs generally give different trajectories.

## Summary

| Context                                      | Recommended pattern                              | Copies of the jump state          | Safe?                   |
|:-------------------------------------------- |:------------------------------------------------ |:--------------------------------- |:----------------------- |
| Many serial solves of one problem            | Reuse the problem                                | None                              | Yes                     |
| Serial parameter sweep                       | `remake`                                         | None (shared)                     | Yes                     |
| `EnsembleSerial`, `safetycopy = false`       | Default `prob_func`                              | None                              | Yes                     |
| `EnsembleThreads`, `safetycopy = false`      | Default, or a `prob_func` that remakes its input | One per spawned task              | Yes                     |
| Any ensemble, `safetycopy = true`            | Any `prob_func` deriving from its input          | One per trajectory                | Yes                     |
| `prob_func` returning a shared problem       | Avoid                                            | n/a                               | No                      |
| Stateful callbacks in ensembles              | Pass to `JumpProblem`, reset in `initialize`     | Copied with the problem           | Yes                     |
| Stateful callbacks passed to `solve`         | Avoid with threads                               | Shared                            | No                      |
| Manual threads or tasks                      | `deepcopy` once per task                         | One per task                      | Yes                     |
| Simultaneously alive integrators             | `deepcopy` each problem                          | One per integrator                | Yes                     |
| `EnsembleDistributed`, `pmap_batch_size > 1` | `safetycopy = true` or `pmap_batch_size = 1`     | One per trajectory, or per worker | Yes with these settings |
