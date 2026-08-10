# MOBI parity and carbon-divergence correction design

Date: 2026-08-10  
Status: Approved in conversation

## Objective

Correct the demonstrated carbonate failure chain and the additional verified MOBI-to-Veros-BGC translation discrepancies, then prove numerical stability with a 60-year integration using Veros v0.2.3.

The scientific evidence and source comparison are recorded in `MOBI_VEROS_BGC_DIVERGENCE_ANALYSIS.md`. This document defines the implementation boundary and acceptance contract.

## Scope

The change will:

- make carbonate root solving deterministic, positive, validated, and independent of prior `hSWS`;
- correct the carbonic-acid coefficient typo;
- restore MOBI organic carbon/alkalinity stoichiometry;
- close zooplankton excretion budgets;
- restore all MOBI calcite-production contributors;
- restore the MOBI biological substep count, seasonal phase, calcite scale, and carbonate input limits;
- correct floating-point absolute-value use in the optional Bohrium solver;
- add focused regression and parity tests;
- run and document a 60-year four-degree integration.

The change will not:

- replace the carbonate implementation with a third-party package;
- refactor the entire NPZD rule engine into a new carbon-ledger architecture;
- alter the intentional stress-derived wind reconstruction;
- alter verified air–sea flux direction, units, or timestep application;
- change the dimensionally preferable Python `dpCO2` diagnostic;
- change the calcite redistribution and monitor fixes merged in pull request #8.

## Carbonate calculation and root solver

`carbon_flux` will clamp its chemistry inputs to MOBI's surface limits: temperature `[-2, 35] °C` and salinity `[0, 45]`.

Every wet carbonate cell will solve in the fixed hydrogen-ion interval `[1e-10, 1e-6]`. The prior `hSWS` field will no longer define or narrow the interval. Land cells will remain excluded through the existing wet mask.

The masked root routine will preserve the original active mask and validate the solution contract. It will reject:

- nonfinite endpoint residuals;
- endpoints that do not have opposite residual signs, except an endpoint already satisfying tolerance;
- nonfinite iterates or derivatives;
- exhausted iteration limits;
- returned values outside `[1e-10, 1e-6]` or at/below zero;
- final residuals inconsistent with the requested accuracy.

The residual check will use a scale-aware threshold based on the requested root accuracy and local derivative, approximately `2 * accuracy * max(abs(df), 1)`. Invalid active-cell solutions will raise a descriptive error with the failure class and affected-cell count; unmasked land values will not cause false failures. No unvalidated value will reach the carbonate rational functions.

The `k1` constant will change from `62.088` to the MOBI value `62.008`. The optional generated C/Bohrium safeguard will use `fabs` for floating-point input and retain the same validation semantics as the NumPy implementation.

Because the global bracket makes the result independent of previous state, `hSWS` will remain a derived, non-restart diagnostic field.

## Organic carbon and alkalinity

The broad `dic_alk_scale` rule will be removed from active processing. Alkalinity tendencies will instead be tied explicitly to the organic transformations that change DIC, with:

```text
delta_alkalinity_organic = -delta_DIC_organic / C:N
```

Primary production therefore produces the opposite alkalinity tendency to organic remineralization, recycling, excretion, and bottom remineralization. Atmospheric CO2 exchange will not affect alkalinity. Calcite will remain independent with its existing two-equivalent DIC/alkalinity stoichiometry.

The implementation will use process-local rules rather than derive alkalinity from the difference between temporary and original total DIC. This prevents ordering effects and contamination by gas or calcite updates.

## Zooplankton excretion and calcite production

The active basic rule group will explicitly:

- subtract excretion from zooplankton;
- return the corresponding P:N amount to PO4;
- leave the carbon-group DIC source as the single C:N-scaled carbon return.

The unused generic excretion group will be removed or replaced so it cannot be accidentally enabled to double-count or incorrectly P:N-scale the DIC member.

Implicit calcite production will include exactly the MOBI pathways:

- phytoplankton mortality;
- zooplankton mortality;
- sloppy phytoplankton grazing;
- sloppy zooplankton self-grazing.

Detritus grazing will not be added. Existing calcite signs, dissolution, bottom redistribution, and mask handling will remain unchanged.

## Bundled setup parity

The four-degree Veros-BGC setup will use:

- `dt_bio = dt_tracer / 8`, giving eight 10,800-second biology steps per 86,400-second tracer day;
- solar-declination phase `0.22` instead of `0.72`;
- `dcaco3 = 6500 m` instead of `2500 m`.

The existing wind reconstruction from stress will remain unchanged, as explicitly requested by the user.

## Test-driven implementation

Production changes will follow failing regression tests. New tests will be divided by behavior:

### Carbonate tests

- literal MOBI oracle values over representative temperature, salinity, DIC, and alkalinity tuples;
- the saved year-55 pre-runaway tuple, with the corrected coefficient expectation;
- identical roots for prior `hSWS` values spanning negative, zero, tiny positive, ordinary, and overly large guesses;
- physical positive roots and finite `pCO2`/`cflux` for alkaline states below `1e-8`;
- explicit rejection of invalid brackets, nonfinite values, nonconvergence, and excessive residuals;
- MOBI temperature and salinity clamping behavior;
- coefficient regression for `62.008`;
- aligned NumPy and optional generated-kernel safeguards where the backend is available.

Expected values will be literal fixtures derived independently from the MOBI equations, not reconstructed from the implementation under test.

### Biology and alkalinity tests

- process-by-process organic `delta ALK = -delta DIC / 7.1` relationships;
- no alkalinity response to atmospheric carbon flux;
- independent calcite `delta ALK = 2 * delta DIC` magnitude/sign behavior;
- a closed one-cell grazing and excretion carbon budget;
- exactly one DIC excretion return;
- calcite production containing both mortality and both sloppy-grazing pathways;
- no detritus-grazing contribution.

### Setup and existing regressions

- eight biological substeps per tracer day;
- the `0.22` declination phase;
- `dcaco3 = 6500 m`;
- existing bottom-mask, per-column calcite conservation, current-timestep monitor, and halo-exclusion tests.

Each regression will be observed failing for the intended reason before the corresponding minimal production change is applied. The completed suite will run under the repository's supported Veros v0.2.3 environment, followed by compilation and whitespace checks.

## Sixty-year integration proof

The final validation will run from an isolated feature worktree and isolated Python environment containing:

- Veros exactly at tag `v0.2.3`;
- the feature-branch Veros-BGC installed from the isolated worktree;
- the dependency versions needed by those historical packages.

A copied four-degree setup will be configured for `86400 * 360 * 60` seconds. Snapshot output will be annual, giving exactly 60 completed annual records. The run log and NetCDF results will live outside the Git worktree so large generated files cannot enter the pull request.

An acceptance script will inspect all wet cells and require:

1. successful process completion;
2. exactly 60 annual snapshots;
3. finite DIC, alkalinity, `pCO2`, `hSWS`, and `cflux` at every annual record;
4. `1e-10 <= hSWS <= 1e-6`;
5. nonnegative DIC and alkalinity;
6. year-60 maximum DIC below ten times the year-1 maximum;
7. year-60 maximum alkalinity below ten times the year-1 maximum;
8. all new and existing automated tests passing.

The acceptance contract demonstrates removal of the observed runaway and retention of bounded carbon state. It does not claim complete climatological equivalence between the two circulation models.

The repository will contain a compact Markdown validation record with exact commands, commit IDs, package versions, configuration, annual extrema, and pass/fail checks. Large NetCDF output and verbose logs will remain local but their paths and relevant summary metadata will be recorded.

## Failure handling

If a focused test or the long integration fails, the failure will be diagnosed from first principles before further changes. No acceptance item will be weakened to obtain a passing result, and no pull request will claim success until the complete 60-year contract passes. Any newly discovered model choice that would materially expand scope will be brought back to the user.

## Git and pull-request delivery

Work will remain on `agent/fix-mobi-parity-carbon-divergence` and will be executed in an isolated worktree after the written plan is approved. Commits will separate documentation, focused tests/fixes, and validation evidence where practical.

Before publication, the branch will receive a whole-change review and fresh verification of:

- the full test suite;
- Python compilation;
- clean whitespace/diff checks;
- the 60-year acceptance report;
- the absence of generated datasets or environment files from Git.

The branch will then be pushed and a GitHub pull request will be created with the failure evidence, correction summary, test results, and 60-year validation linked from its description.

