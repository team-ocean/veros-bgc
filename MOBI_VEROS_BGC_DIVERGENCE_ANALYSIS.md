# MOBI–Veros-BGC numerical-divergence analysis

Date: 2026-08-10

## Executive summary

The late-time numerical divergence is caused by the Veros-BGC carbonate solver accepting nonphysical hydrogen-ion concentrations. MOBI solves the carbonate system in a fixed, strictly positive interval. Veros-BGC instead constructs an additive interval around the previous `hSWS` value. That interval crosses zero for ordinary oceanic concentrations below approximately `1e-8`, and the root routine neither proves that the interval brackets a root nor validates its returned value. Once a negative root is accepted, the carbonate rational functions can produce negative or enormous `pCO2`; air–sea exchange then changes from stabilizing outgassing to carbon uptake and drives DIC to overflow.

The saved 300-year experiment establishes the order of failure: invalid `hSWS` appears years before the runaway, then `pCO2` and `cflux` become nonphysical, then DIC and alkalinity explode, and finally the state becomes NaN. Re-evaluating a saved pre-runaway cell with MOBI's positive bracket recovers a physical root and reverses the flux from inward to outward.

The comparison also found independent translation discrepancies in the carbonate coefficient, organic alkalinity bookkeeping, zooplankton excretion, calcite production, biological substep count, solar-declination phase, calcite dissolution length scale, temperature/salinity safeguards, and the optional Bohrium root kernel. These discrepancies predate commit `3860c6d285670ba166d5286c369a2ccc5091a474`; that commit exposed and fixed a separate dimensionality problem but did not introduce the carbonate-solver defect.

## Compared sources and provenance

- MOBI oracle: `/home/nutrik/Documents/bugfix/UVic2.9`, especially `updates/07/source/common/co2calc.F`, `updates/07/source/common/gasbc.F`, `updates/07/source/mom/npzd_src.F`, `updates/07/source/mom/tracer.F`, and related headers.
- The historical MOBI file used during the original translation is represented by Git commit `f5530a619...`; the relevant current `updates/07` biology source was verified byte-identical to that historical blob.
- Veros core compatibility target: `/home/nutrik/Documents/bugfix/veros` at tag `v0.2.3` (`bcbead...`).
- Veros-BGC comparison target: merged `master` at `1d7146c`, including the calcite/bottom-mask corrections from pull request #8.
- Saved long-run evidence: `/home/nutrik/Documents/bugfix/backup/4deg.snapshot.nc`.

Line numbers below describe the inspected revisions and may move after the fixes.

## Proven failure chain

### 1. MOBI guarantees a positive carbonate root bracket

MOBI calls its safeguarded root solver with hydrogen-ion bounds `1e-6` and `1e-10` in `updates/07/source/common/co2calc.F`. Both endpoints are positive and span the expected oceanographic solution. The comment that discusses using a previous solution describes a pH-local search; in hydrogen-ion space that would require multiplicative positive bounds, not an additive interval.

Veros-BGC `veros_bgc/core/atmospherefluxes.py` instead derives a boundary from the previous `hSWS` and solves between `hSWS - boundary` and `hSWS + boundary`, with a minimum boundary of `1e-8`. Therefore:

- a valid previous value below roughly `1e-8` produces a negative lower endpoint;
- a root that moves above approximately twice the previous concentration can fall outside the interval;
- once a negative value is stored, the next interval may remain wholly negative and cannot recover.

This is not an edge case: 856 of 2,315 wet surface cells had `hSWS < 1.01e-8` in the first saved year.

### 2. The Python root routine accepts invalid results

The Veros-BGC `drtsafe_masked` implementation does not reject any of the following:

- endpoint residuals with the same sign;
- nonfinite endpoints or iterates;
- nonpositive or out-of-bracket hydrogen concentrations;
- failure to meet the requested residual tolerance;
- exhaustion of the 100-iteration limit.

It returns the last iterate, which is immediately stored as `hSWS` and used by the carbonate formulas. Negative hydrogen-ion concentrations enter rational expressions whose denominators have nonphysical negative-axis poles and sign changes.

### 3. Saved-state timing proves that chemistry fails before DIC

The annual snapshots show:

| Model year | Evidence |
| --- | --- |
| 1 | Minimum wet `hSWS` is `3.887e-9`; DIC and alkalinity are finite. |
| 5 | First negative wet value appears: `-1.2148e-9`. |
| 5–54 | At least one negative value persists without immediate global failure. |
| 53 | Surface `pCO2` is already negative in at least one cell (`-93.43`). |
| 55 | Two cells have negative `hSWS`; the first catastrophic flux reversal is visible. |
| 56 | 32 cells have negative `hSWS`; maximum `pCO2` is `3.341e27` and maximum `cflux` is `7.287e24`. |
| 57 | 2,310 cells have negative `hSWS`; maximum `pCO2` reaches approximately `8.692e285`. |
| 58 | The carbon state contains NaNs. |

At 110°E, 16°N in year 55, the saved physical inputs are still ordinary:

- temperature: `24.04129088 °C`;
- salinity: `33.872667`;
- DIC: `2114.4811777 mmol m-3`;
- alkalinity: `2181.1384228 mmol m-3`.

The saved solver result is nevertheless `hSWS = -8.1778e-8`, `pCO2 = -3625`, and `cflux = +0.0023535`, an inward carbon flux. Recomputing the same state with the MOBI positive bracket gives `hSWS = 3.03635e-8`, physical positive `pCO2`, and an outward flux. Using the existing, mistyped Python equilibrium coefficient gives `pCO2 ≈ 1201`; correcting that independent coefficient typo gives `pCO2 ≈ 1366.57`.

One year after the invalid chemistry result, DIC in that cell reaches approximately `4.85e22 mmol m-3`; the global maximum is `2.67578e27`. In year 57 the global maximum is approximately `2.45e288`, followed by NaNs. The chronology identifies the solver result as the trigger and air–sea exchange as the positive-feedback channel.

A convergence-basin probe gives the same conclusion. For the saved pre-failure state, prior guesses of `3e-9` and `5e-9` return negative values, whereas `3e-8` and `5e-7` converge to the same physical root near `3.036e-8`. A valid carbonate solution must not depend on which prior guess happens to be stored.

## Additional translation discrepancies

### Carbonic-acid equilibrium coefficient `k1`

Veros-BGC uses `-62.088` in the `k1` expression. MOBI uses `-62.008` in every inspected `co2calc.F` variant. The typo makes `k1` about 20.2% too large under representative conditions and materially weakens calculated outgassing. Isolated examples changed computed `pCO2` by roughly 14–15% when only this constant was corrected. It is a significant carbon-cycle bias but does not, by itself, explain the explosive failure.

Required change: replace `62.088` with `62.008`.

### Organic DIC–alkalinity coupling

MOBI applies organic alkalinity as

```text
delta_alkalinity_organic = -delta_DIC_organic / C:N
```

and handles calcite separately. Air–sea CO2 flux is not part of the biological source array used in this relationship.

Veros-BGC's broad `dic_alk_scale` rule instead uses the positive sign and compares the current temporary DIC field with original DIC after atmospheric exchange and calcite production have already modified it. It therefore:

- applies the wrong sign to organic transformations;
- converts atmospheric carbon flux into an alkalinity tendency;
- mixes the independently stoichiometric calcite tendency into the organic rule.

The runaway snapshots contain a direct fingerprint: at the maximum-DIC cell in years 56 and 57, `ALK / DIC` is exactly `1 / 7.1`, the configured C:N ratio. This identifies `dic_alk_scale` as an amplification and contamination path.

Required change: remove the broad state-difference rule and attach explicit alkalinity tendencies only to the organic transformations, with the negative DIC/C:N relationship. Gas exchange must not alter alkalinity; calcite must retain its independent two-equivalent stoichiometry.

### Zooplankton excretion is not closed in the active rule set

MOBI subtracts excretion from zooplankton and returns the corresponding phosphorus and carbon to dissolved pools. The active Veros-BGC groups add the carbon excretion source to DIC, but omit the common zooplankton sink and phosphate source. A separately registered generic excretion group is not selected by the bundled setup; selecting it unchanged would also be wrong because it scales every sink by P:N and would duplicate or mis-scale the dedicated DIC rule.

A one-cell process budget with grazing `1`, sloppy feeding `0.5`, and excretion `0.2` creates `7.1 × 0.2 = 1.42` carbon units in the active Python rules, while MOBI conserves carbon.

Required change: make the zooplankton excretion sink and PO4 source explicit in the active basic group, retain exactly one correctly C:N-scaled DIC source, and remove or replace the misleading unused generic group.

### Calcite production omits zooplankton pathways

MOBI's implicit calcite-production expression includes:

- phytoplankton mortality;
- zooplankton mortality;
- the sloppy fraction of phytoplankton grazing;
- the sloppy fraction of zooplankton self-grazing.

Veros-BGC includes only the phytoplankton mortality and phytoplankton grazing terms. In a normalized one-cell example with phytoplankton mortality `1`, zooplankton mortality `2`, phytoplankton grazing `3`, zooplankton self-grazing `4`, and assimilation `0.5`, Python produces `2.5` while MOBI produces `6.5` before the common scale factors.

Required change: pass and include the missing zooplankton terms. Do not include detritus grazing, which is absent from the MOBI expression.

The signs for calcite production, instantaneous dissolution, and the bottom redistribution corrected in pull request #8 otherwise match MOBI.

### Biological integration uses four substeps instead of eight

MOBI defines `dtnpzd = dtts / 4`. Its default tracer half-step is `dtts = 43,200 s`, while the leapfrog scalar span is `c2dtts = 2 × dtts = 86,400 s`. The driver consequently computes eight biological substeps of 10,800 seconds per one-day tracer update.

The bundled Veros-BGC setup sets `dt_tracer = 86,400 s` and `dt_bio = dt_tracer / 4`, so only four 21,600-second explicit Euler steps are taken. This changes nonlinear grazing and mortality trajectories and reduces stability.

Required change: configure `dt_bio = dt_tracer / 8` for oracle parity.

### Solar-declination phase is reversed

MOBI uses a seasonal phase offset of `0.22`; Veros-BGC uses `0.72`. The difference is exactly half a year and reverses the declination sign for the same model time. UVic restart initialization sets the relative year to zero, and the Veros setup applies no half-year time offset, so the mismatch is not explained by a stored calendar shift.

Required change: use `0.22` unless an explicit, documented forcing-calendar transformation is introduced. The bundled setup has none.

### Calcite dissolution length scale differs

MOBI's default `dcaco3` is 6,500 m. The bundled Veros-BGC four-degree setup uses 2,500 m. No provenance for an intentional retuning was found.

Required change: restore 6,500 m in the oracle-parity setup.

### Missing carbonate input safeguards

MOBI clamps surface temperature to `[-2, 35] °C` and salinity to `[0, 45]` before carbonate chemistry. Veros-BGC passes the raw surface fields. The saved pre-failure range was already within the MOBI limits, so this omission did not trigger the observed failure, but it is a robustness and translation discrepancy.

Required change: apply the same clamps at the carbonate-calculation boundary.

### Optional Bohrium kernel uses integer-style `abs`

The generated Bohrium/C root code calls `abs` on a floating-point expression in its Newton safeguard, although another expression correctly uses `fabs`. Depending on compiler declarations, `abs` can coerce to integer and alter the safeguarded-step decision. The observed run used the NumPy path and therefore was not affected.

Required change: use `fabs` for the floating-point comparison and keep the generated and NumPy solvers behaviorally aligned.

## Inspected differences that should be retained

### Existing wind-stress reconstruction

MOBI receives an explicit wind-speed field; the Veros setup reconstructs wind speed from surface stress. The user confirmed that this adaptation is intentional. Saved reconstructed winds are finite and physically ordinary (`0.58–10.56 m s-1`). It affects calibration but is not the instability trigger and will remain unchanged.

### Air–sea flux direction, units, and timestep

The apparent factor-of-100 difference in piston velocity is a unit conversion: MOBI carries centimetres per second while Veros-BGC uses metres per second. Both define positive flux as atmosphere-to-ocean, and the surface concentration update converts consistently to `mmol m-3`. The PRE rule executes and applies the returned tendency once. No sign, scaling, or double-timestep defect was found.

### `dpCO2` diagnostic expression

The Python `dpCO2` diagnostic differs from a MOBI expression, but it does not feed `cflux`; the Python form is dimensionally preferable. It will remain unchanged.

### Pull request #8 calcite redistribution

The Boolean bottom mask, corrected vertical redistribution coefficients, multiplication-based mask reset, current-timestep diagnostics, and halo exclusion merged through pull request #8 agree with the inspected MOBI behavior and existing regressions. They will remain intact.

### `hSWS` restart storage

The current variable is not restart-written. This is a reproducibility concern only while the solution depends on its previous value. Restoring a deterministic global positive bracket removes that dependency; `hSWS` can remain a derived diagnostic state rather than expanding restart compatibility in this change.

## Planned corrections

The focused correction will preserve the current architecture and:

1. restore a deterministic, strictly positive carbonate bracket `[1e-10, 1e-6]`;
2. validate endpoint signs, finiteness, convergence, bracket containment, positivity, and final residual before accepting a root;
3. clamp carbonate temperature and salinity as MOBI does;
4. correct the `k1` constant from `62.088` to `62.008`;
5. replace floating-point `abs` with `fabs` in the optional generated solver;
6. replace broad DIC-state alkalinity scaling with explicit organic-process tendencies;
7. close zooplankton excretion in phosphorus, nitrogen-biomass, and carbon budgets without double counting;
8. include the missing zooplankton calcite-production terms;
9. configure eight biology substeps per tracer day;
10. restore the MOBI `0.22` seasonal phase and 6,500 m calcite dissolution scale;
11. keep the existing wind reconstruction, verified flux conventions, `dpCO2` diagnostic, and pull request #8 fixes.

## Verification strategy

Implementation will be test-driven. New tests will use literal MOBI-derived expectations rather than duplicating the Python formulas under test. Coverage will include:

- carbonate reference tuples and the saved pre-failure state;
- independence from prior positive, tiny, zero, and negative `hSWS` guesses;
- explicit invalid-bracket, nonfinite, iteration, positivity, and residual rejection;
- temperature and salinity clamps;
- isolated organic DIC/alkalinity stoichiometry for each process;
- a closed grazing/excretion carbon budget;
- all four MOBI calcite-production contributors;
- eight-substep, seasonal-phase, and `dcaco3` configuration parity;
- all existing bottom-mask, redistribution, and monitor regressions.

The final acceptance run will use Veros v0.2.3 and the fixed Veros-BGC plugin in an isolated environment for 60 model years of 360 days. It must produce 60 annual snapshots and satisfy all of the following over wet cells:

- DIC, alkalinity, `pCO2`, `hSWS`, and `cflux` are finite;
- `1e-10 <= hSWS <= 1e-6`;
- DIC and alkalinity are nonnegative;
- year-60 maximum DIC and alkalinity are each less than ten times their year-1 maxima;
- all new and existing tests pass.

Large NetCDF output will remain local. The reproducible environment, command, configuration, annual summary metrics, and acceptance result will be committed as Markdown evidence before a pull request is opened.

