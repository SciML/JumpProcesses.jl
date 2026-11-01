# [VR_Direct and VR_DirectFW](@id vr_direct_algorithm)

[`VR_Direct`](@ref) and [`VR_DirectFW`](@ref) simulate general
`VariableRateJump`s in ODE and SDE problems with one exponential threshold for
the total rate, as in a direct method. This page explains how they integrate the
total rate over each solver step: with a piecewise Chebyshev interpolant, fit
once per step, whose stored antiderivative answers every evaluation of the event
condition. They differ only in how they store the rate and affect functions, so
everything here applies to both, and both are called `VR_Direct` below. The code
is in `src/variable_rate.jl`. The machinery shared with the other aggregators is
described in [How jumps are simulated](@ref algorithms_overview).

## At a glance

`VR_Direct` treats all ``N`` variable-rate jumps as one stream of events: the
next jump, of any type, happens when the integral of the total rate ``\Lambda``
reaches an exponential threshold. It finds that time with a `ContinuousCallback`,
so it runs alongside the ODE or SDE solver. The solver takes a step, and
DiffEqBase evaluates the callback's condition over that step. If the condition
changes sign, DiffEqBase finds the root, cuts the step there, and runs the
callback's affect, which fires one jump. The first evaluation of the condition
in a step fits ``\Lambda`` over the whole step, and every evaluation uses that
fit, without evaluating the rates again.

```text
initialize                                                            §5
    E_rem ← draw Exp(1)          # what remains of the threshold
    t₀    ← start time           # the time E_rem applies at
    fit   ← empty

for each step [t_prev, t] the solver takes:
    DiffEqBase evaluates the condition g at t_prev and at t
    on the first evaluation for this step:                            §5
        E_rem ← E_rem − (old fit's integral from t₀ to t_prev)
        t₀    ← t_prev
        fit Λ on [t₀, t], panel by panel from the left:               §1, §2
            sample Λ at 8 Chebyshev points
            halve a panel while its error estimate exceeds the tolerance
            stop after the panel where the integral passes E_rem
    each evaluation returns g(s) = E_rem − (fit's integral from t₀ to s)  §3, §4

    if g changes sign in the step:
        find τ with g(τ) = 0 on the fit (no rate evaluations)
        the solver cuts the step at τ, unless another callback's event is earlier
        affect at τ:                                                  §6
            choose jump i with probability λᵢ(τ)/Λ(τ) and apply its affect
            E_rem ← draw Exp(1);   t₀ ← τ;   fit ← empty
```

## What is computed

With ``N`` variable-rate jumps, the total rate is

```math
\Lambda(t) = \sum_{i=1}^{N} \lambda_i\big(u(t),\, p,\, t\big).
```

`VR_Direct` draws ``E \sim \mathrm{Exp}(1)`` and fires a jump at the first time
``\tau`` with ``\int_{t_\ell}^{\tau} \Lambda(s)\,ds = E``, where ``t_\ell`` is
the time of its last jump, or the start time. Its callback's condition is the
remaining threshold,

```math
g(t) = E_{\mathrm{rem}} - \int_{t_0}^{t} \Lambda(s)\,ds,
```

where ``t_0`` is the start of the current solver step and ``E_{\mathrm{rem}}``
is what remains of ``E`` at ``t_0``. The condition starts positive and decreases,
and the jump happens where it reaches zero. The rest of this page is about how
the integral is computed, and how it is kept consistent across steps.

## The pieces

### 1. Fitting one panel

`fit_total_rate!` fits the total rate on a panel ``[a, b]``, which is the whole
step or part of it. The points are the ``n = 8`` (`NUM_CHEB_NODES`) Chebyshev
points of the first kind, mapped onto the panel:

```math
x_j = \cos\Big(\frac{\pi\,(j - \tfrac12)}{n}\Big), \quad j = 1, \dots, n, \qquad
\tau_j = m + h\,x_j, \quad m = \tfrac{a+b}{2}, \quad h = \tfrac{b-a}{2}.
```

At each point, the state comes from the solver's interpolant, and every rate is
evaluated (`total_variable_rate`):

```math
f_j = \Lambda\big(u(\tau_j),\, p,\, \tau_j\big).
```

A matrix (`transform`) maps the samples to Chebyshev coefficients,

```math
c_k = \frac{2}{n} \sum_{j=1}^{n} f_j \cos\Big(\frac{\pi k\,(j - \tfrac12)}{n}\Big),
\qquad k = 0, \dots, n-1,
```

which define the interpolant on the panel, in the variable ``x \in [-1, 1]``:

```math
q(x) = \frac{c_0}{2} + \sum_{k=1}^{n-1} c_k\, T_k(x), \qquad q(x_j) = f_j.
```

Its integral over the panel uses ``\int_{-1}^{1} T_k\,dx = 2/(1-k^2)`` for even
``k`` and ``0`` for odd ``k`` (the weights `integral_weights`):

```math
I = h \Big( c_0 + \sum_{\substack{k = 2 \\ k \text{ even}}}^{n-1} \frac{2\, c_k}{1 - k^2} \Big).
```

The estimate of its error is the size of the last two coefficients,

```math
\varepsilon = (b - a)\big(|c_{n-1}| + |c_{n-2}|\big),
```

for the reasons in [A.1](@ref vr_direct_a1). The points, the matrix and the
weights are computed once, with `cospi` in the problem's time type,
`eltype(prob.tspan)`, so a higher precision type such as `BigFloat` is not
limited to `Float64` accuracy.

### 2. Refining, and stopping at the jump

`fit_step!` fits the step ``[t_0, t_1]`` with `fit_panels!`, recursively from
left to right. Let ``S(a)`` be the integral accumulated from ``t_0`` to a
panel's start, and ``R(a) = E_{\mathrm{rem}} - S(a)`` the remaining threshold
there. A panel's integral must be accurate to

```math
\mathrm{tol} = \mathrm{abstol} + \mathrm{reltol}\,\big|R(a)\big|,
```

with the solver's `abstol` and `reltol`, or their smallest entries if they are
arrays. This is the test the solver would apply if ``R`` minus the integral were
a component of the state, as each jump's integrated rate is with `VR_FRM`; see
[A.2](@ref vr_direct_a2).

The panel contains the jump if the total passes the threshold,
``S(a) + I > E_{\mathrm{rem}}``. That total is the number stored for the panel's
end and used by the condition there (§3, §4), so the fit and the condition agree
under rounding. The comparison is strict, so that the condition is negative
after the panel, rather than zero over the rest of the step. On that panel the
interpolant must also be accurate relative to ``R(a)``, which can be much smaller
than ``I``, so the rounding error of its antiderivative is added to the
estimate (`antiderivative_rounding`; see [A.3](@ref vr_direct_a3)):

```math
\varepsilon \leftarrow \max\Big(\varepsilon,\; (b - a)(n + 1)\,\epsilon_{\mathrm{mach}} \sum_{k} |c_k| \Big).
```

Then:

```text
if ε > tol and depth < D and a < m < b:   # D: MAX_CHEB_DEPTH = 12, or
    fit [a, m]                            #    MAX_CHEB_CROSSING_DEPTH = 40
    if it passed the threshold: stop      #    for the panel with the jump
    fit [m, b]
else:
    store the panel, with S(a) and S(b) = S(a) + I
```

Fitting stops after the panel where the threshold is passed, since the solver
will cut the step at the jump. The test ``a < m < b`` stops halving once a panel
can't be split in floating point.

### 3. What each panel stores

Instead of ``q``, `push_panel!` stores the coefficients of its antiderivative in
``x``, chosen to be zero at ``x = -1``:

```math
\int_{-1}^{x} q(s)\,ds = \sum_{j=0}^{n} C_j\, T_j(x), \qquad
C_j = \frac{c_{j-1} - c_{j+1}}{2j} \;\; (j \ge 1,\; c_k = 0 \text{ for } k \ge n), \qquad
C_0 = -\sum_{j=1}^{n} (-1)^j\, C_j.
```

| Quantity                                        | Kept             | Size                  |
|:----------------------------------------------- |:---------------- |:--------------------- |
| panel start ``a`` and end ``b``                 | per panel        | 2                     |
| integrals ``S(a)`` and ``S(b)``                 | per panel        | 2                     |
| antiderivative coefficients ``C_0, \dots, C_n`` | per panel        | ``n + 1 = 9``         |
| samples ``f_j`` and coefficients ``c_k``        | buffers, reused  | ``2n``                |
| points, transform matrix, integration weights   | once per problem | ``n``, ``n^2``, ``n`` |

These are the fields of `TotalRateFit`, the `rate_fit` field of
`VR_DirectEventCache`. Its vectors are emptied every step but keep their
capacity, so nothing is allocated after the first steps.

### 4. Evaluating the condition

For each evaluation, at the step's ends and during root finding,
`fitted_rate_integral` finds the panel containing ``t`` by binary search, or the
last panel if ``t`` is past it, and the condition returns

```math
g(t) = \begin{cases}
E_{\mathrm{rem}} - \Big[\, S(a) + h \displaystyle\sum_{j=0}^{n} C_j\, T_j(x) \Big], &
x = \dfrac{2t - a - b}{b - a}, \quad t < b, \\[1.2em]
E_{\mathrm{rem}} - S(b), & t \ge b,
\end{cases}
```

with the sum computed by Clenshaw's recurrence (`chebyshev_sum`):

```math
\beta_{n+1} = \beta_{n+2} = 0, \qquad
\beta_j = C_j + 2x\,\beta_{j+1} - \beta_{j+2} \;\; (j = n, \dots, 1), \qquad
\sum_{j=0}^{n} C_j T_j(x) = C_0 + x\,\beta_1 - \beta_2.
```

That is a few dozen floating-point operations, with no rate evaluations. At or
past a panel's end, ``g`` uses the stored total ``S(b)``: the number the fit
decided with, and the one the next panel starts from. Past the last panel it
keeps that value instead of extrapolating the polynomial, and since the fit stops
only once ``S(b) > E_{\mathrm{rem}}``, the value there is negative.

### 5. Between steps

DiffEqBase evaluates the condition only while checking a step
``[t_{\mathrm{prev}}, t]``: first at ``t_{\mathrm{prev}}``, then at ``t`` and
during root finding. So the fit is current exactly when the step's two ends are
those of the fitted step. The first evaluation for a new step, seen as a change
in `integrator.tprev` or `integrator.t`, commits the previous fit
``F_{\mathrm{old}}`` up to the new step's start, and fits the new step:

```math
E_{\mathrm{rem}} \leftarrow E_{\mathrm{rem}} - F_{\mathrm{old}}(t_{\mathrm{prev}}), \qquad
t_0 \leftarrow t_{\mathrm{prev}}, \qquad \text{then fit } [t_0,\, t].
```

  - After a normal step, ``t_{\mathrm{prev}}`` is the old step's end, so this
    subtracts the whole previous integral.
  - If another callback's event cut the step short, the next step starts at
    the event, before the old fit's end. Only the integral up to the event is
    used, and the rest of the interval is fitted again with the state after the
    event. The next step can end where the cut step would have, for example at
    the final time, so the start is compared as well as the end.
  - While the integrator is still at ``t_0``, as right after one of
    `VR_Direct`'s own jumps, there is nothing to fit.

At initialization, ``E_{\mathrm{rem}}`` is drawn, ``t_0`` is the start time, and
the fit is empty.

### 6. Firing a jump

At the root ``\tau``, the affect evaluates every rate at ``u(\tau)``, keeping
their cumulative sums (`cum_rate_sum`). It draws ``U\,\Lambda(\tau)`` with ``U``
uniform on ``[0, 1)``, finds the jump ``i`` whose interval of the cumulative sums
contains it, and runs that jump's affect. It then draws a new threshold, sets
``t_0 = \tau``, and empties the fit.

## Interaction with the integrator

  - **When the condition is evaluated.** The callback has `interp_points = 0`, so
    DiffEqBase checks only the step's ends before root finding. Interior checks
    are not needed, since the fit has already examined the whole step: the
    condition changes sign over the step exactly when the fit found the
    threshold reached in it. Root finding then evaluates only the fit.
  - **Other events.** The jumps of an SSA aggregator, user `tstops`, and
    discrete callbacks all act at the end of a step, so the next step's fit
    starts from the state after them, with no reset needed. A continuous
    callback's event can end a step early, which §5 handles.
  - **Callback options.** All the jumps share one `ContinuousCallback`. Its
    `save_positions` is the elementwise OR of the jumps' `save_positions`, and its
    `abstol` and `reltol` are the smallest of the jumps'. The jumps'
    `rootfind` and `interp_points` are not used.
  - **SDEs.** The rates are evaluated along the solver's interpolant between
    steps. In an SDE that doesn't resolve the noise within a step, so a rate that
    depends on a component driven by noise needs steps that are short compared
    with the rate's variation, for example by setting `dtmax`. `VR_FRM`
    integrates each rate as a component of the state instead, so the SDE solver's
    error control covers it.
  - **Number types.** The thresholds, the fit and its tables use the time type of
    the problem, `eltype(prob.tspan)`.

## Accuracy and error control

  - **The solver's `abstol` and `reltol`** control the state, and so the rates at
    the fit's points, through the solver's error control and interpolant. They
    also set the tolerance of each panel's integral (§2). An error ``\delta`` in
    the integral moves a jump by about ``\delta / \Lambda(\tau)``.
  - **The jumps' `abstol` and `reltol`** are passed to the callback. In current
    DiffEqBase versions, `abstol` only guards against detecting the same event
    twice, and root finding brackets the root of ``g`` to adjacent floating-point
    times. They don't affect the accuracy of the integral.
  - **The depth limits** are fixed. A panel that reaches its limit is accepted
    even if its error estimate is above the tolerance.

The errors come from the solver's interpolant of the state; from interpolating
``\Lambda`` on each panel, which the tail coefficients estimate (A.1); from
rounding, which is bounded on the panel with the jump (A.3); and from root
finding, which is accurate to the floating-point resolution of the time.

These are checked in `test/variable_rate.jl`. Test 7 compares first jump times
with those from the analytically integrated rate, to a relative error of
``10^{-5}`` in `Float32` and ``10^{-6}`` in `Float64`, with solver steps that
each integrate to many times the threshold. Test 11 does so in `BigFloat`, to
``10^{-25}`` with tolerances of ``10^{-30}``. Test 12 checks thresholds reached
exactly at panel ends. Tests 6 and 9 and the "VariableRateJump mean counts vs
exact" tests check mean numbers of jumps, Test 8 a rate driven by SDE noise, and
Tests 9 and 10 steps cut short by another callback.

## Cost

For ``N`` jumps and ``n = 8`` points per panel:

| Work                             | Cost                                                                                     |
|:-------------------------------- |:---------------------------------------------------------------------------------------- |
| fitting a step                   | ``n`` state interpolations and ``nN`` rate evaluations per panel fitted                  |
| each evaluation of the condition | a binary search over the step's panels and a sum of ``n + 1`` terms; no rate evaluations |
| firing a jump                    | ``N`` rate evaluations and a binary search                                               |
| memory                           | ``n + 5 = 13`` numbers per panel of the current step                                     |

A smooth rate usually needs one panel per step. A halved panel's samples are
not reused, since first-kind points don't nest, so a step that ends with ``P``
panels costs at most ``2P - 1`` panel fits. Root finding and the other condition
evaluations add almost nothing, so the cost per step doesn't depend on how many
there are.

## Limitations, and when to use something else

  - **All the rates at every point.** Each point of the fit evaluates all ``N``
    rates, whatever they depend on, so the cost per step grows with ``N``. When
    bounds on the rates are available, `Coevolve` evaluates each jump's rate only
    at that jump's own candidate times.
  - **Rates driven by SDE noise** need short steps, or `VR_FRM`; see above.
  - **Features between the points.** The error estimate sees only the samples.
    A feature that falls between a panel's points, such as a narrow pulse in a
    rate over a long step, or an oscillation at a frequency that aliases into
    small final coefficients, can be missed with no sign in the estimate. If a
    rate has such features at known times, pass those times as `tstops`, or
    limit the step size with `dtmax`.
  - **Discontinuous rates.** Near a discontinuity or a kink, the coefficients
    decay slowly, and panels are halved up to the depth limit. Passing the time
    of a known discontinuity as a `tstop` makes it a step boundary instead.

## References

Sampling the next jump from one exponential threshold on the integrated total
rate, and choosing the jump with probability ``\lambda_i / \Lambda``, is
Gillespie's direct method [^1] applied to time-dependent rates. Locating the jump
time with the solver's event detection follows the hybrid method of Salis and
Kaznessis [^2]. Chebyshev interpolation at first-kind points, its coefficients'
decay, the aliasing bound and the integration of Chebyshev series are standard
[^3], as is Clenshaw's recurrence [^4]. Specific to JumpProcesses are fitting the
total rate once per solver step and answering every evaluation of the condition
from the stored antiderivative, the tolerance relative to the remaining
threshold, stopping the fit at the jump, and the rounding term.

[^1]: D. T. Gillespie, A general method for numerically simulating the stochastic
    time evolution of coupled chemical reactions, Journal of Computational
    Physics, 22 (4), 403–434 (1976). doi:10.1016/0021-9991(76)90041-3.

[^2]: H. Salis and Y. Kaznessis, Accurate hybrid stochastic simulation of a
    system of coupled chemical or biochemical reactions, Journal of Chemical
    Physics, 122 (5), 054103 (2005). doi:10.1063/1.1835951.

[^3]: L. N. Trefethen, Approximation Theory and Approximation Practice, Extended
    Edition, SIAM (2019). doi:10.1137/1.9781611975949.

[^4]: C. W. Clenshaw, A note on the summation of Chebyshev series, Mathematics of
    Computation, 9 (51), 118–120 (1955). doi:10.1090/S0025-5718-1955-0071856-0.

## Appendix: the error estimates

Both estimates are heuristics rather than rigorous bounds. The one in §1
estimates how far a panel's interpolant is from the true rate. The tolerance in
§2 decides how small that error must be, with a rounding term added on the one
panel where rounding can matter.

### [A.1 The interpolation estimate](@id vr_direct_a1)

Expand the true rate on the panel in Chebyshev polynomials:

```math
\Lambda(x) = \sum_{k \ge 0} a_k\, T_k(x).
```

For a smooth rate the coefficients decay quickly: geometrically,
``|a_k| \lesssim C\rho^{-k}`` with ``\rho > 1``, when ``\Lambda`` is analytic in
a neighborhood of the panel, and algebraically when it is only finitely
differentiable [^3]. The interpolant of degree ``n-1`` captures the terms up to
``k = n-1``, so its error comes from the omitted tail, at most doubled by
aliasing:

```math
\max_{x \in [-1,1]} \big|\Lambda(x) - q(x)\big| \le 2 \sum_{k \ge n} |a_k|.
```

The tail is not known, but with geometric decay it is comparable to the last
coefficients that are: for ``\rho \ge 2``,
``\sum_{k \ge n} |a_k| \lesssim |a_{n-1}|``. The computed ``c_k`` are close to
``a_k`` when the coefficients decay, so ``|c_{n-1}|`` and ``|c_{n-2}|`` stand in
for the tail. Two coefficients are used rather than one because a rate that is
even or odd about the panel's midpoint has every other coefficient zero, so
``c_{n-1}`` alone could vanish while the error does not.

The factor ``b - a`` turns an error in the rate into an error in the panel's
integral:

```math
\Big| \int_a^b (\Lambda - q)\, dt \Big| = h\, \Big| \int_{-1}^{1} (\Lambda - q)\, dx \Big|
\le 2h \max_x |\Lambda - q| = (b - a) \max_x |\Lambda - q|.
```

For the integral this is usually pessimistic, since integration damps high-order
error: ``\int_{-1}^{1} T_k\,dx`` is ``0`` for odd ``k`` and ``2/(1-k^2)`` for even
``k``. It is not a guaranteed bound, though: a rate oscillating at just the wrong
frequency for a panel's length can alias into small final coefficients.

### [A.2 The tolerance](@id vr_direct_a2)

```math
\mathrm{tol} = \mathrm{abstol} + \mathrm{reltol}\,\big|R(a)\big|, \qquad R(a) = E_{\mathrm{rem}} - S(a).
```

This is the mixed test an ODE solver applies to a component ``y`` of the state,
``\mathrm{abstol} + \mathrm{reltol}\,|y|``, with ``y`` the remaining threshold.
There are two reasons for it:

  - With `VR_FRM`, each jump's integrated rate is a component of the state, so
    the solver's own error control governs it. Using the same form gives
    `VR_Direct` comparable accuracy for the same `abstol` and `reltol`.
  - It is the scale that sets the jump time. The jump happens where
    ``R(a) - \int_a^{\tau} \Lambda = 0``, so an error ``\delta`` in that integral
    moves it by about ``\Delta\tau \approx \delta / \Lambda(\tau)``. Measuring
    ``\delta`` against ``|R|``, which is of order one since it comes from an
    ``\mathrm{Exp}(1)`` draw, controls the jump time on a consistent scale. A
    tolerance relative to the panel's integral ``I`` instead would allow errors
    larger than ``R`` itself when ``I`` is much larger than ``R``.

### [A.3 The rounding term](@id vr_direct_a3)

The condition evaluates ``S(a) + h \sum_j C_j T_j(x)`` by Clenshaw's recurrence,
whose absolute rounding error is about ``\epsilon_{\mathrm{mach}}`` times the
number of terms times their size. Since
``C_j = (c_{j-1} - c_{j+1})/(2j)`` and ``|C_0| \le \sum_{j \ge 1} |C_j|``,

```math
\sum_{j=0}^{n} |C_j| \le 2 \sum_{k} |c_k|,
```

so the rounding error is about

```math
h\,(n+1)\,\epsilon_{\mathrm{mach}} \cdot 2\sum_k |c_k| = (b-a)(n+1)\,\epsilon_{\mathrm{mach}} \sum_k |c_k|.
```

These terms are about the size of the whole panel's integral ``I``, but near the
jump the value needed is about ``R``, which can be far smaller. The sum then
cancels heavily, and its absolute error is about ``\epsilon_{\mathrm{mach}} I``.
In the `Float32` case of Test 7, the rate ``10^8 t^4`` integrates to
``I = 2 \times 10^7`` over the step ``[0, 1]``, and
``\epsilon_{\mathrm{mach}} \approx 1.2 \times 10^{-7}``, so the rounding error is
about 2, against a threshold of order one. Taking the larger of this term and
``\varepsilon`` halves the panel until ``\epsilon_{\mathrm{mach}} I`` is below
the tolerance.

The term is used only on the panel containing the jump. Elsewhere the condition
only needs the right sign, and away from the root its value is far from zero, so
rounding of this size can't flip it. The panels before the jump contribute only
their total ``I``, a short weighted sum without cancellation. For the same
reason, only that panel gets the deeper halving limit, 40 instead of 12.

!!! info "Developer notes"

    These notes are for working on `VR_Direct` itself.

      - `VR_DirectEventCache` is both the condition and the affect of the
        callback. `prev_time` is ``t_0``, the start of the current fit;
        `prev_threshold` is ``E_{\mathrm{rem}}`` there; `current_time` is the end
        of the fitted step; and `current_threshold` is the last value of ``g``
        returned.
      - The condition fits a new step when
        `integrator.t != prev_time && (integrator.t != current_time || integrator.tprev != prev_time)`.
        This relies on DiffEqBase evaluating continuous conditions only while
        checking a step, so that `(integrator.tprev, integrator.t)` identifies
        it. A step that starts before `prev_time` is an error.
      - `fitted_rate_integral` returns a panel's stored end total at or past its
        end, and never extrapolates. `fit_panels!` decides whether the threshold
        was passed with the same number, so the two can't disagree under rounding
        (Test 12).
      - The state at the fit's points is interpolated into the first of the
        integrator's temporary caches (`get_tmp_cache`), and the panel vectors
        keep their capacity, so fitting and evaluating the condition don't
        allocate once the vectors have grown.
      - The constants are `NUM_CHEB_NODES`, `MAX_CHEB_DEPTH` and
        `MAX_CHEB_CROSSING_DEPTH`. The tables, the error estimate and the stored
        panel size all follow `NUM_CHEB_NODES`.
      - The tests are Tests 6 to 12 in `test/variable_rate.jl`, and the test that
        the callback uses `interp_points = 0`.
