# [GPU ensembles](@id gpu_ensembles)

JumpProcesses provides [`EnsembleGPUKernel`](@ref) to run independent pure-jump
trajectories in parallel, with one trajectory per device thread. It uses
KernelAbstractions and Adapt; load both packages as well as your GPU backend.
The examples below use CUDA and require a working CUDA installation. They are
not executed during the documentation build.

## Supported solvers and inputs

| Solver | Jump representation | Required solve options |
|:-------|:--------------------|:-----------------------|
| `SSAStepper()` | `MassActionJump` | `trajectories`, `saveat` |
| `SimpleExplicitTauLeaping()` | `MassActionJump` | `trajectories`, `saveat` |
| `SimpleTauLeaping()` | `MassActionJump` or `RegularJump` | `trajectories`, `dt` |

All three paths require a `DiscreteProblem`. `SSAStepper` simulates individual
reaction events exactly; the other two introduce tau-leaping error. There are no
`EnsembleGPUKernel` implementations for `SimpleImplicitTauLeaping`,
`SimpleTrapezoidalLeaping`, `SimpleAdaptiveTauLeaping`, or StochasticDiffEq's
tau-leaping algorithms.

## Mass-action ensembles

Construct the problem using ordinary host arrays. The ensemble implementation
transfers the reaction data to the device and returns solutions that can be
inspected on the CPU.

```julia
using JumpProcesses, KernelAbstractions, Adapt, CUDA, Statistics

maj = MassActionJump(
    [0.1 / 1000.0, 0.01],
    [[1 => 1, 2 => 1], [2 => 1]],
    [[1 => -1, 2 => 1], [2 => -1, 3 => 1]]
)
prob = DiscreteProblem([999.0, 10.0, 0.0], (0.0, 250.0))
jprob = JumpProblem(prob, PureLeaping(), maj)

sol = solve(
    EnsembleProblem(jprob), SimpleExplicitTauLeaping(; epsilon = 0.01),
    EnsembleGPUKernel(CUDABackend()); trajectories = 10_000, saveat = 10.0
)
mean(solution.u[end][3] for solution in sol.u)
```

`epsilon` controls adaptive leap selection. `saveat` specifies the output grid,
not a fixed leap size; the kernel shortens leaps to land on the next save time.
The SSA and adaptive explicit kernels require `saveat` (an interval or collection
of times), and support `save_start` and `save_end`.

For an exact SSA ensemble of the same model, construct a problem with an SSA
aggregator and change the time-stepper:

```julia
ssa_prob = JumpProblem(prob, Direct(), maj)
ssa_sol = solve(
    EnsembleProblem(ssa_prob), SSAStepper(), EnsembleGPUKernel(CUDABackend());
    trajectories = 10_000, saveat = 10.0
)
```

The `SSAStepper` and `SimpleExplicitTauLeaping` GPU kernels do not accept
`RegularJump`, `ConstantRateJump`, `VariableRateJump`, user callbacks, or an
ensemble `prob_func`. They share the
same initial condition and reaction data across trajectories. Use a CPU ensemble
such as `EnsembleThreads()` if these restrictions do not fit the model.

## Fixed-step ensembles

Use the same mass-action `jprob` from the first example with fixed steps:

```julia
fixed_sol = solve(
    EnsembleProblem(jprob), SimpleTauLeaping(), EnsembleGPUKernel(CUDABackend());
    trajectories = 10_000, dt = 0.1
)
```

### More general rates with RegularJump

The `SimpleTauLeaping` GPU kernel additionally supports the same `RegularJump`
interface as its CPU counterpart. Its rate and update functions must compile for the GPU:
use device-compatible operations and parameters, and write updates in place.

```julia
using JumpProcesses, KernelAbstractions, Adapt, CUDA

function rate!(out, u, p, t)
    out[1] = p * u[1] / (1 + u[1])
    return nothing
end

function change!(du, u, p, t, counts, mark)
    du[1] = -counts[1]
    du[2] = counts[1]
    return nothing
end

rj = RegularJump(rate!, change!, 1)
prob = DiscreteProblem([1000.0, 0.0], (0.0, 1.0), 100.0)
jprob = JumpProblem(prob, PureLeaping(), rj)
sol = solve(
    EnsembleProblem(jprob), SimpleTauLeaping(), EnsembleGPUKernel(CUDABackend());
    trajectories = 10_000, dt = 0.01
)
```

This fixed-step kernel saves the step grid; do not rely on the SSA and adaptive
explicit kernels' `saveat` controls here. Use a `dt` that divides the simulation interval.
It does not reject negative populations, and marked jumps, callbacks, and
per-trajectory problem customization are not supported by this path.

`EnsembleGPUKernel()` without an explicit backend currently selects
KernelAbstractions' CPU backend. Pass `CUDABackend()` to request CUDA execution.
Other KernelAbstractions backends also need compatible device random-number
support; backend availability alone does not establish solver support.
