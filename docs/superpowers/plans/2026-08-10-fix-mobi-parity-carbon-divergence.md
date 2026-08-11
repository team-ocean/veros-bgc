# MOBI Parity and Carbon-Divergence Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the proven carbonate runaway and every approved MOBI translation discrepancy, then demonstrate a bounded 60-year four-degree integration with Veros v0.2.3.

**Architecture:** Preserve the existing rule-based NPZD and air–sea-flux architecture. Harden the existing safeguarded root solver, restore literal MOBI coefficients and process terms through focused internal rules, and validate both unit behavior and an annual NetCDF trajectory. Keep the stress-derived wind reconstruction, verified flux conventions, current `dpCO2` diagnostic, and pull request #8 calcite redistribution unchanged.

**Tech Stack:** Python 3.11, NumPy 1.23.5, pytest, pytest-forked, h5netcdf, Veros v0.2.3 (`bcbead95d56341f3ee9f30311c05a546674095e7`), Veros-BGC, uv, Git, and GitHub.

## Global Constraints

- Work only on `agent/fix-mobi-parity-carbon-divergence` in an isolated worktree created with `superpowers:using-git-worktrees` before Task 1.
- Use Veros exactly at tag `v0.2.3`; inspect newer Veros only for tooling, never as the compatibility reference.
- Follow strict RED/GREEN test-driven development with `superpowers:test-driven-development` for every production change.
- Use `superpowers:systematic-debugging` before changing code in response to any unexpected failure.
- Preserve the existing surface-stress wind-speed reconstruction exactly.
- Preserve air–sea CO2 direction, units, timestep application, and the current Python `dpCO2` diagnostic.
- Preserve pull request #8's Boolean bottom mask, `rcak`/`rcab` conservation, current-timestep monitor indexing, and halo exclusion.
- Do not add a third-party carbonate package or redesign the NPZD rule engine.
- Keep generated environments, copied run setups, logs, and NetCDF files outside Git.
- Apply local source edits with `apply_patch`; use formatting or generation commands only for mechanical outputs.
- Commit after every independently passing task.
- Do not publish a pull request unless every focused test, the full suite, compilation, diff checks, review, and the 60-year acceptance contract pass.
- Approved pre-flight ruling: NPZD tests must assert selected-rule outcomes and conservation, not private rule names or the absence of removed registrations.
- Approved pre-flight ruling: the generated Bohrium C-kernel change from floating-point `abs` to `fabs` is a narrow TDD exception because the backend is unavailable and a source-text assertion would be a change-detector test; verify it through compilation and review.

## File Structure

- Create `test/test_atmospherefluxes.py`: synthetic root-solver contract, MOBI carbonate tuple, prior-state independence, and chemistry-clamp regressions.
- Modify `veros_bgc/core/atmospherefluxes.py`: fixed positive bracket, solver validation, MOBI input bounds, corrected `k1`, and Bohrium `fabs`.
- Create `test/test_npzd_mobi_parity.py`: excretion, organic alkalinity, calcite-production, seasonal-phase, and setup-parity regressions.
- Modify `veros_bgc/core/npzd_rules.py`: explicit organic alkalinity rules and complete calcite production.
- Modify `veros_bgc/core/npzd.py`: activate excretion correctly, register explicit alkalinity rules, use complete calcite rules, and restore the seasonal phase.
- Modify `veros_bgc/setup/bgc_global_4deg/bgc_global_four_degree.py`: eight biology substeps and the 6,500 m calcite scale.
- Create `scripts/validate_60_year_run.py`: deterministic NetCDF acceptance checker and Markdown metrics renderer.
- Create `test/test_validate_60_year_run.py`: synthetic passing and failing 60-record snapshots.
- Create `docs/validation/2026-08-10-mobi-parity-60-year.md` only after the real run: exact environment, commands, hashes, annual metrics, and acceptance output.
- Keep `test/test_npzd_regressions.py` unchanged except when a genuinely incompatible assumption is proven; all three existing regressions must continue to pass.

---

### Task 1: Enforce the masked root-solver contract

**Files:**
- Create: `test/test_atmospherefluxes.py`
- Modify: `veros_bgc/core/atmospherefluxes.py:537-707`

**Interfaces:**
- Consumes: existing `drtsafe_step(vs, mask, drtsafe_val, x_high, df, f, x_low, dx, dx_old, accuracy)` and `drtsafe_boundary_update(...)` in `veros_bgc.core.atmospherefluxes`.
- Produces: unchanged `drtsafe_masked(vs, mask, function, guess_low, guess_high, args=None, accuracy=1e-10, max_iterations=100) -> numpy.ndarray`, now guaranteed finite, positive, bracket-contained, converged, residual-validated, and non-mutating with respect to the caller's mask.

- [ ] **Step 1: Create the isolated compatibility environment and establish the baseline**

Run from the isolated Veros-BGC worktree:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python numpy==1.23.5 scipy==1.10.1 h5py==3.8.0 h5netcdf==1.1.0 pillow==9.5.0 ruamel.yaml==0.17.40 loguru==0.7.3 tqdm==4.67.1 click==8.1.8 entrypoints==0.4 requests==2.32.4 pytest==8.4.1 pytest-forked==1.6.0 graphviz==0.21
uv pip install --python .venv/bin/python --no-deps -e /home/nutrik/Documents/bugfix/.worktrees/veros-v0.2.3
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python -m pytest -q
```

Expected: the existing three NPZD regression tests pass. If uv cannot create the environment, use `python3.11 -m venv .venv` and the same package list through `.venv/bin/python -m pip install`; record the tool actually used in the validation report.

- [ ] **Step 2: Write failing root-solver contract tests**

Create `test/test_atmospherefluxes.py` with these imports, helpers, and tests:

```python
import numpy as np
import pytest

from veros.state import VerosState

from veros_bgc.core.atmospherefluxes import drtsafe_masked


SHAPE = (2, 2)


def full(value):
    return np.full(SHAPE, value, dtype=np.float64)


def linear_residual(vs, mask, x, root):
    return x - root, np.ones_like(x)


def test_drtsafe_masked_returns_bracketed_root_without_mutating_mask():
    vs = VerosState()
    active = np.array([[True, False], [True, True]])
    original = active.copy()

    result = drtsafe_masked(
        vs,
        active,
        linear_residual,
        full(1e-10),
        full(1e-6),
        args=(full(3e-8),),
        accuracy=1e-12,
    )

    np.testing.assert_array_equal(active, original)
    np.testing.assert_allclose(result[active], 3e-8, rtol=0.0, atol=1e-12)


def test_drtsafe_masked_rejects_unbracketed_active_cells():
    vs = VerosState()
    active = np.ones(SHAPE, dtype=bool)

    with pytest.raises(RuntimeError, match="not bracketed"):
        drtsafe_masked(
            vs,
            active,
            linear_residual,
            full(1e-10),
            full(1e-6),
            args=(full(2e-6),),
        )


def nonfinite_endpoint_residual(vs, mask, x):
    residual = x - 3e-8
    residual = np.where(x == 1e-10, np.nan, residual)
    return residual, np.ones_like(x)


def test_drtsafe_masked_rejects_nonfinite_endpoint_values():
    vs = VerosState()
    active = np.ones(SHAPE, dtype=bool)

    with pytest.raises(RuntimeError, match="non-finite endpoint"):
        drtsafe_masked(
            vs,
            active,
            nonfinite_endpoint_residual,
            full(1e-10),
            full(1e-6),
        )


def discontinuous_residual(vs, mask, x):
    residual = np.where(x <= 1e-10, -1.0, np.where(x >= 1e-6, 1.0, 1e-3))
    return residual, np.ones_like(x)


def test_drtsafe_masked_rejects_excessive_final_residual():
    vs = VerosState()
    active = np.ones(SHAPE, dtype=bool)

    with pytest.raises(RuntimeError, match="residual"):
        drtsafe_masked(
            vs,
            active,
            discontinuous_residual,
            full(1e-10),
            full(1e-6),
            accuracy=1e-10,
        )


def test_drtsafe_masked_reports_iteration_exhaustion():
    vs = VerosState()
    active = np.ones(SHAPE, dtype=bool)

    with pytest.raises(RuntimeError, match="did not converge"):
        drtsafe_masked(
            vs,
            active,
            linear_residual,
            full(1e-10),
            full(1e-6),
            args=(full(3e-8),),
            max_iterations=0,
        )
```

- [ ] **Step 3: Run the root-solver tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest test/test_atmospherefluxes.py -q
```

Expected: the root value test may pass, but mask preservation and all validation tests fail because the current routine mutates `mask` and silently accepts invalid states.

- [ ] **Step 4: Implement explicit validation in `drtsafe_masked`**

Replace the body beginning at the current initial-guess section with logic equivalent to this complete contract:

```python
    args = () if args is None else args
    active_mask = mask.copy()
    work_mask = active_mask.copy()

    drtsafe_val = 0.5 * (guess_low + guess_high)
    dx = np.abs(guess_high - guess_low)
    dx_old = dx.copy()

    f_low, df_low = function(vs, active_mask, guess_low, *args)
    f_high, df_high = function(vs, active_mask, guess_high, *args)
    f, df = function(vs, active_mask, drtsafe_val, *args)

    bad_endpoint = active_mask & ~(
        np.isfinite(f_low) & np.isfinite(df_low)
        & np.isfinite(f_high) & np.isfinite(df_high)
    )
    if bad_endpoint.any():
        raise RuntimeError(
            "drtsafe_masked received non-finite endpoint values in {} active cells".format(
                np.count_nonzero(bad_endpoint)
            )
        )

    low_tolerance = 2 * accuracy * np.maximum(np.abs(df_low), 1.0)
    high_tolerance = 2 * accuracy * np.maximum(np.abs(df_high), 1.0)
    low_done = active_mask & (np.abs(f_low) <= low_tolerance)
    high_done = active_mask & ~low_done & (np.abs(f_high) <= high_tolerance)
    drtsafe_val[low_done] = guess_low[low_done]
    drtsafe_val[high_done] = guess_high[high_done]
    work_mask[low_done | high_done] = False

    same_sign = work_mask & (np.signbit(f_low) == np.signbit(f_high))
    if same_sign.any():
        raise RuntimeError(
            "drtsafe_masked root is not bracketed in {} active cells".format(
                np.count_nonzero(same_sign)
            )
        )

    bad_initial = work_mask & ~(np.isfinite(f) & np.isfinite(df))
    if bad_initial.any():
        raise RuntimeError(
            "drtsafe_masked received non-finite iterates in {} active cells".format(
                np.count_nonzero(bad_initial)
            )
        )

    x_low, x_high, f_low, f_high = np.where(
        f_low < 0,
        (guess_low, guess_high, f_low, f_high),
        (guess_high, guess_low, f_high, f_low),
    )

    for _ in range(max_iterations):
        if not work_mask.any():
            break

        drtsafe_step(
            vs, work_mask, drtsafe_val, x_high, df, f,
            x_low, dx, dx_old, accuracy,
        )

        if not work_mask.any():
            break

        f, df = function(vs, work_mask, drtsafe_val, *args)
        bad_iterate = work_mask & ~(np.isfinite(f) & np.isfinite(df))
        if bad_iterate.any():
            raise RuntimeError(
                "drtsafe_masked received non-finite iterates in {} active cells".format(
                    np.count_nonzero(bad_iterate)
                )
            )

        drtsafe_boundary_update(
            vs, work_mask, x_low, x_high, df, f,
            f_low, f_high, drtsafe_val,
        )

    if work_mask.any():
        raise RuntimeError(
            "drtsafe_masked did not converge in {} active cells after {} iterations".format(
                np.count_nonzero(work_mask), max_iterations
            )
        )

    final_f, final_df = function(vs, active_mask, drtsafe_val, *args)
    lower_bound = np.minimum(guess_low, guess_high)
    upper_bound = np.maximum(guess_low, guess_high)
    invalid_root = active_mask & ~(
        np.isfinite(drtsafe_val)
        & np.isfinite(final_f)
        & np.isfinite(final_df)
        & (drtsafe_val > 0)
        & (drtsafe_val >= lower_bound)
        & (drtsafe_val <= upper_bound)
    )
    if invalid_root.any():
        raise RuntimeError(
            "drtsafe_masked returned invalid roots in {} active cells".format(
                np.count_nonzero(invalid_root)
            )
        )

    residual_tolerance = 2 * accuracy * np.maximum(np.abs(final_df), 1.0)
    bad_residual = active_mask & (np.abs(final_f) > residual_tolerance)
    if bad_residual.any():
        raise RuntimeError(
            "drtsafe_masked final residual is too large in {} active cells".format(
                np.count_nonzero(bad_residual)
            )
        )

    return drtsafe_val
```

In the generated C kernel inside `drtsafe_step`, change only the floating-point safeguard:

```c
bool tmp_mask = step_inside || (fabs(2.0 * f[i]) > fabs(dx_old[i] * df[i]));
```

This generated string is the user-approved narrow TDD exception recorded in Global Constraints. Do not add a source-text test for it; compilation and task review are its verification.

- [ ] **Step 5: Run the focused tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest test/test_atmospherefluxes.py -q
```

Expected: all five root-solver tests pass.

- [ ] **Step 6: Commit the root-solver contract**

```bash
git add test/test_atmospherefluxes.py veros_bgc/core/atmospherefluxes.py
git commit -m "fix: validate masked carbonate root solving"
```

---

### Task 2: Restore MOBI carbonate chemistry and positive global bracket

**Files:**
- Modify: `test/test_atmospherefluxes.py`
- Modify: `veros_bgc/core/atmospherefluxes.py:10-55, 160-258`

**Interfaces:**
- Consumes: validated `drtsafe_masked(...)` from Task 1.
- Produces: unchanged `carbon_flux(vs) -> numpy.ndarray` and `co2calc_SWS(vs, temperature, salinity, dic_in, ta_in, co2_in, atmospheric_pressure) -> numpy.ndarray`, with MOBI input clamps, `k1` coefficient, deterministic positive roots, and prior-state independence.

- [ ] **Step 1: Add the Veros state fixture and carbonate regressions**

Append these imports, helper, and tests to `test/test_atmospherefluxes.py`:

```python
import veros_bgc

from veros_bgc.core import atmospherefluxes


def make_carbon_state():
    vs = VerosState(use_plugins=(veros_bgc,))
    vs.nx, vs.ny, vs.nz = 2, 1, 1
    vs.enable_npzd = True
    vs.enable_carbon = True
    vs.allocate_variables()
    vs.maskT[...] = False
    vs.maskT[2:-2, 2:-2, :] = True
    vs.temp[...] = 15.0
    vs.salt[...] = 35.0
    vs.dic[...] = 2300.0
    vs.alkalinity[...] = 2400.0
    vs.hSWS[...] = 1e-8
    vs.forc_temp_surface[...] = 1.0
    vs.surface_taux[...] = 0.0
    vs.surface_tauy[...] = 0.0
    vs.atmospheric_co2 = 280.0
    return vs


def test_saved_pre_failure_tuple_matches_mobi_and_ignores_previous_hsws():
    vs = make_carbon_state()
    shape = vs.hSWS.shape
    temperature = np.full(shape, 24.04129088)
    salinity = np.full(shape, 33.872667)
    dic = np.full(shape, 2114.4811777e-3)
    alkalinity = np.full(shape, 2181.1384228e-3)
    wet = vs.maskT[:, :, -1].astype(bool)
    roots = []
    pco2 = []

    for prior in (-1e-8, 0.0, 1e-12, 3e-9, 5e-9, 3e-8, 5e-7, 2e-6):
        vs.hSWS[...] = prior
        atmospherefluxes.co2calc_SWS(
            vs,
            temperature,
            salinity,
            dic,
            alkalinity,
            280.0,
            1.0,
        )
        roots.append(vs.hSWS[wet].copy())
        pco2.append(vs.pCO2[wet].copy())

    roots = np.asarray(roots)
    pco2 = np.asarray(pco2)
    np.testing.assert_allclose(roots, roots[0], rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(pco2, pco2[0], rtol=0.0, atol=1e-6)
    assert np.all((roots >= 1e-10) & (roots <= 1e-6))
    np.testing.assert_allclose(pco2, 1366.57, rtol=5e-4, atol=0.0)
    assert np.all(vs.dco2star[wet] < 0.0)


def test_carbon_flux_clamps_only_chemistry_temperature_and_salinity(monkeypatch):
    vs = make_carbon_state()
    vs.temp[2, 2, -1, vs.tau] = -5.0
    vs.temp[3, 2, -1, vs.tau] = 50.0
    vs.salt[2, 2, -1, vs.tau] = -3.0
    vs.salt[3, 2, -1, vs.tau] = 50.0
    captured = {}

    def fake_co2calc(vs, temperature, salinity, *args):
        captured["temperature"] = temperature.copy()
        captured["salinity"] = salinity.copy()
        return np.zeros_like(temperature)

    monkeypatch.setattr(atmospherefluxes, "co2calc_SWS", fake_co2calc)
    atmospherefluxes.carbon_flux(vs)

    np.testing.assert_array_equal(
        captured["temperature"][2:-2, 2:-2],
        np.array([[-2.0], [35.0]]),
    )
    np.testing.assert_array_equal(
        captured["salinity"][2:-2, 2:-2],
        np.array([[0.0], [45.0]]),
    )
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest test/test_atmospherefluxes.py::test_saved_pre_failure_tuple_matches_mobi_and_ignores_previous_hsws test/test_atmospherefluxes.py::test_carbon_flux_clamps_only_chemistry_temperature_and_salinity -q
```

Expected: the tuple test fails or raises for negative/small prior guesses and reports approximately 1,201 ppm with the mistyped coefficient; the clamp test observes `-5/50 °C` and `-3/50` salinity instead of the MOBI limits.

- [ ] **Step 3: Implement the MOBI chemistry boundary and coefficient**

In `carbon_flux`, introduce chemistry-only inputs while leaving wind reconstruction and the Schmidt-number calculation unchanged:

```python
    surface_temperature = vs.temp[:, :, -1, vs.tau]
    surface_salinity = vs.salt[:, :, -1, vs.tau]
    chemistry_temperature = np.maximum(-2.0, np.minimum(35.0, surface_temperature))
    chemistry_salinity = np.maximum(0.0, np.minimum(45.0, surface_salinity))

    vs.dco2star = co2calc_SWS(
        vs,
        chemistry_temperature,
        chemistry_salinity,
        vs.dic[:, :, -1, vs.tau] * 1e-3,
        vs.alkalinity[:, :, -1, vs.tau] * 1e-3,
        vs.atmospheric_co2,
        atmospheric_pressure,
    )
```

Correct the literal `k1` transcription:

```python
    k1 = 10 ** (-1.0 * (3670.7 / temperature_in_kelvin - 62.008
                + 9.7944 * np.log(temperature_in_kelvin)
                - 0.0118 * salinity + 0.000116 * salinity**2))
```

Replace the adaptive `hSWS ± boundary` block with the fixed oracle interval and preserve land values:

```python
    in1 = np.full_like(vs.hSWS, 1e-10)
    in2 = np.full_like(vs.hSWS, 1e-6)
    iter_mask = np.empty_like(in1, dtype=np.bool)
    iter_mask[...] = vs.maskT[:, :, -1]

    solved_hsws = drtsafe_masked(
        vs,
        iter_mask,
        ta_iter_SWS,
        in1,
        in2,
        args=(k1, k2, k1p, k2p, k3p, st, ks, kf, ft,
              dic, ta, sit, ksi, pt, bt, kw, kb),
        accuracy=xacc,
    )
    vs.hSWS = np.where(iter_mask, solved_hsws, vs.hSWS)
```

- [ ] **Step 4: Run focused and existing regressions**

Run:

```bash
.venv/bin/python -m pytest test/test_atmospherefluxes.py -q
.venv/bin/python -m pytest test/test_npzd_regressions.py -q
```

Expected: all carbonate tests and all three pull request #8 regressions pass.

- [ ] **Step 5: Commit the carbonate parity correction**

```bash
git add test/test_atmospherefluxes.py veros_bgc/core/atmospherefluxes.py
git commit -m "fix: restore MOBI carbonate chemistry bounds"
```

---

### Task 3: Close zooplankton excretion in the active rules

**Files:**
- Create: `test/test_npzd_mobi_parity.py`
- Modify: `veros_bgc/core/npzd.py:429-524, 633-651`

**Interfaces:**
- Consumes: existing `excretion(vs, zooplankton, nutrient)` and `excretion_dic(vs, zooplankton, nutrient)` functions.
- Produces: active rule `npzd_basic_zooplankton_excretion` in `group_npzd_basic`, plus exactly one carbon rule `npzd_carbon_excretion_dic`; no generated `npzd_basic_zooplankton_excretion_DIC` rule.

- [ ] **Step 1: Create NPZD parity fixtures and failing excretion tests**

Create `test/test_npzd_mobi_parity.py`:

```python
import numpy as np

import veros_bgc
from veros.state import VerosState

from veros_bgc.core.npzd import setupNPZD


DZT = np.array([40.0, 30.0, 20.0, 10.0])
ZW = np.array([-60.0, -30.0, -10.0, 0.0])
KBOT = np.array([[1, 2], [3, 4]])


def make_npzd_state():
    vs = VerosState(use_plugins=(veros_bgc,))
    vs.nx, vs.ny, vs.nz = KBOT.shape[0], KBOT.shape[1], DZT.size
    vs.enable_npzd = True
    vs.enable_carbon = True
    vs.dcaco3 = 2500.0
    vs.npzd_selected_rules = ["group_npzd_basic", "group_carbon_implicit_caco3"]
    vs.allocate_variables()
    vs.dzt[:] = DZT
    vs.zw[:] = ZW
    vs.area_t[:] = 1.0
    vs.kbot[2:-2, 2:-2] = KBOT
    levels = np.arange(vs.nz)[np.newaxis, np.newaxis, :]
    vs.maskT[:] = (vs.kbot[..., np.newaxis] > 0) & (
        vs.kbot[..., np.newaxis] - 1 <= levels
    )
    setupNPZD(vs)
    return vs


def zero_process_rates(vs):
    for collection in (
        vs.net_primary_production,
        vs.recycled,
        vs.mortality,
        vs.grazing,
        vs.npzd_export,
    ):
        for value in collection.values():
            value[...] = 0.0
    vs.excretion_total[...] = 0.0


def selected_primary_changes(vs):
    changes = {
        name: np.zeros_like(data[..., 0])
        for name, data in vs.npzd_tracers.items()
    }
    for rule in vs.npzd_rules:
        output = rule.function(vs, rule.source, rule.sink)
        for name, value in output.items():
            if isinstance(rule.boundary, tuple):
                changes[name][rule.boundary] += value
            else:
                changes[name] += value * rule.boundary
    return changes


def test_selected_excretion_behavior_closes_phosphorus_and_carbon_budgets():
    vs = make_npzd_state()
    zero_process_rates(vs)
    vs.excretion_total[...] = 0.2

    changes = selected_primary_changes(vs)
    wet_cell = (2, 2, -1)

    np.testing.assert_allclose(changes["zooplankton"][wet_cell], -0.2)
    np.testing.assert_allclose(
        changes["po4"][wet_cell],
        vs.redfield_ratio_PN * 0.2,
    )
    np.testing.assert_allclose(changes["DIC"][wet_cell], 1.42)
    np.testing.assert_allclose(
        vs.redfield_ratio_CN * changes["zooplankton"][wet_cell]
        + changes["DIC"][wet_cell],
        0.0,
        rtol=0.0,
        atol=1e-14,
    )
```

- [ ] **Step 2: Run the excretion regression and verify RED**

Run:

```bash
.venv/bin/python -m pytest test/test_npzd_mobi_parity.py::test_selected_excretion_behavior_closes_phosphorus_and_carbon_budgets -q
```

Expected: FAIL because selected behavior adds `1.42` DIC without subtracting zooplankton or returning PO4, so the carbon budget does not close.

- [ ] **Step 3: Register excretion directly in the basic group**

Add `excretion` to the `setup_basic_npzd_rules` import and register it next to the zooplankton rules:

```python
    register_npzd_rule(vs, 'npzd_basic_zooplankton_excretion',
                       (excretion, 'zooplankton', 'po4'),
                       label='Excretion')
```

Add `'npzd_basic_zooplankton_excretion'` exactly once to `group_npzd_basic`. Remove both `register_npzd_common_source_rule` calls for this name from `setupNPZD`, including the local `excretion` import. Leave the generic common-rule machinery itself intact for callers outside this setup.

- [ ] **Step 4: Run excretion and legacy regressions**

Run:

```bash
.venv/bin/python -m pytest test/test_npzd_mobi_parity.py::test_selected_excretion_behavior_closes_phosphorus_and_carbon_budgets -q
.venv/bin/python -m pytest test/test_npzd_regressions.py -q
```

Expected: selected-rule behavior returns the literal zooplankton, PO4, and DIC changes exactly once; the carbon budget closes and all existing tests pass.

- [ ] **Step 5: Commit the excretion correction**

```bash
git add test/test_npzd_mobi_parity.py veros_bgc/core/npzd.py
git commit -m "fix: close zooplankton excretion budgets"
```

---

### Task 4: Replace contaminated total-DIC alkalinity scaling

**Files:**
- Modify: `test/test_npzd_mobi_parity.py`
- Modify: `veros_bgc/core/npzd_rules.py:64-99, 168-198`
- Modify: `veros_bgc/core/npzd.py:530-629`

**Interfaces:**
- Produces: `primary_production_alk(vs, plankton, alkalinity)`, `recycling_alk(vs, plankton, alkalinity)`, `excretion_alk(vs, zooplankton, alkalinity)`, and `bottom_remineralization_alk(vs, source, alkalinity)`; each returns `{alkalinity: numpy.ndarray}` in alkalinity concentration units.
- Removes: active `dic_alk_scale` and `npzd_carbon_dic_alk` post-rule.

- [ ] **Step 1: Add process-local alkalinity and gas-isolation tests**

Append to `test/test_npzd_mobi_parity.py`:

```python
import pytest

from veros_bgc.core import atmospherefluxes


def apply_rule_output(tracers, rule, output):
    for name, value in output.items():
        if isinstance(rule.boundary, tuple):
            tracers[name][rule.boundary] += value
        else:
            tracers[name] += value * rule.boundary


def selected_rule_changes(vs, include_pre=False):
    original = {
        name: data[..., vs.tau].copy()
        for name, data in vs.npzd_tracers.items()
    }
    for name, values in original.items():
        vs.temporary_tracers[name][...] = values

    if include_pre:
        for rule in vs.npzd_pre_rules:
            output = rule.function(vs, rule.source, rule.sink)
            apply_rule_output(vs.temporary_tracers, rule, output)

    for rule in vs.npzd_rules:
        output = rule.function(vs, rule.source, rule.sink)
        apply_rule_output(vs.temporary_tracers, rule, output)

    for rule in vs.npzd_post_rules:
        output = rule.function(vs, rule.source, rule.sink)
        apply_rule_output(vs.temporary_tracers, rule, output)

    return {
        name: vs.temporary_tracers[name] - values
        for name, values in original.items()
    }


@pytest.mark.parametrize(
    "process,value,expected_dic,expected_alkalinity",
    [
        ("primary_production", 2.0, -14.2, 2.0),
        ("detritus_recycling", 3.0, 21.3, -3.0),
        ("phytoplankton_recycling", 4.0, 28.4, -4.0),
        ("excretion", 5.0, 35.5, -5.0),
        ("bottom_remineralization", 6.0, 42.6, -6.0),
    ],
)
def test_selected_organic_processes_have_mobi_alkalinity_stoichiometry(
        process, value, expected_dic, expected_alkalinity):
    vs = make_npzd_state()
    zero_process_rates(vs)

    if process == "primary_production":
        vs.net_primary_production["phytoplankton"][...] = value
    elif process == "detritus_recycling":
        vs.recycled["detritus"][...] = value
    elif process == "phytoplankton_recycling":
        vs.recycled["phytoplankton"][...] = value
    elif process == "excretion":
        vs.excretion_total[...] = value
    elif process == "bottom_remineralization":
        vs.npzd_export["detritus"][...] = value

    changes = selected_rule_changes(vs)
    if process == "bottom_remineralization":
        cell = (2, 2, KBOT[0, 0] - 1)
    else:
        cell = (2, 2, -1)

    np.testing.assert_allclose(changes["DIC"][cell], expected_dic)
    np.testing.assert_allclose(
        changes["alkalinity"][cell],
        expected_alkalinity,
    )


def test_selected_atmospheric_flux_changes_dic_but_not_alkalinity(monkeypatch):
    vs = make_npzd_state()
    zero_process_rates(vs)
    vs.cflux[...] = 2.0
    monkeypatch.setattr(
        atmospherefluxes,
        "carbon_flux",
        lambda state: state.cflux,
    )

    changes = selected_rule_changes(vs, include_pre=True)
    surface_cell = (2, 2, -1)

    np.testing.assert_allclose(
        changes["DIC"][surface_cell],
        2.0 * vs.dt_tracer / vs.dzt[-1],
    )
    np.testing.assert_allclose(changes["alkalinity"][surface_cell], 0.0)
```

- [ ] **Step 2: Run the alkalinity tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest test/test_npzd_mobi_parity.py::test_selected_organic_processes_have_mobi_alkalinity_stoichiometry test/test_npzd_mobi_parity.py::test_selected_atmospheric_flux_changes_dic_but_not_alkalinity -q
```

Expected: all organic cases report the wrong alkalinity sign, and the atmospheric-flux case reports a nonzero alkalinity tendency because the broad `npzd_carbon_dic_alk` post-rule consumes total DIC change.

- [ ] **Step 3: Implement explicit organic alkalinity functions**

Replace `dic_alk_scale` with these functions in `npzd_rules.py`:

```python
@veros_method(inline=True)
def primary_production_alk(vs, plankton, alkalinity):
    """Increase alkalinity by organic nitrogen uptake."""
    return {alkalinity: vs.net_primary_production[plankton]}


@veros_method(inline=True)
def recycling_alk(vs, plankton, alkalinity):
    """Decrease alkalinity when organic matter returns DIC."""
    return {alkalinity: -vs.recycled[plankton]}


@veros_method(inline=True)
def excretion_alk(vs, zooplankton, alkalinity):
    """Decrease alkalinity by the organic excretion return."""
    return {alkalinity: -vs.excretion_total}


@veros_method(inline=True)
def bottom_remineralization_alk(vs, source, alkalinity):
    """Decrease alkalinity where exported organic matter remineralizes."""
    return {alkalinity: -vs.npzd_export[source] * vs.bottom_mask}
```

Register five primary rules in `setup_carbon_npzd_rules` with these exact names and mappings:

```python
    register_npzd_rule(vs, 'npzd_carbon_primary_production_alk',
                       (primary_production_alk, 'phytoplankton', 'alkalinity'),
                       label='Primary production')
    register_npzd_rule(vs, 'npzd_carbon_recycling_detritus_alk',
                       (recycling_alk, 'detritus', 'alkalinity'),
                       label='Remineralization')
    register_npzd_rule(vs, 'npzd_carbon_recycling_phyto_alk',
                       (recycling_alk, 'phytoplankton', 'alkalinity'),
                       label='Fast recycling')
    register_npzd_rule(vs, 'npzd_carbon_excretion_alk',
                       (excretion_alk, 'zooplankton', 'alkalinity'),
                       label='Excretion')
    register_npzd_rule(vs, 'npzd_carbon_detritus_bottom_remineralization_alk',
                       (bottom_remineralization_alk, 'detritus', 'alkalinity'),
                       label='Bottom remineralization', boundary='BOTTOM')
```

Remove the `npzd_carbon_dic_alk` registration. In `group_carbon_implicit_caco3`, replace that member with all five explicit names. Keep the calcite alkalinity production and redistribution rules separate and unchanged.

- [ ] **Step 4: Run all biology parity tests**

Run:

```bash
.venv/bin/python -m pytest test/test_npzd_mobi_parity.py -q
.venv/bin/python -m pytest test/test_npzd_regressions.py -q
```

Expected: every excretion and organic-alkalinity test passes, as do all existing regressions.

- [ ] **Step 5: Commit the organic alkalinity correction**

```bash
git add test/test_npzd_mobi_parity.py veros_bgc/core/npzd_rules.py veros_bgc/core/npzd.py
git commit -m "fix: restore organic alkalinity stoichiometry"
```

---

### Task 5: Include zooplankton in implicit calcite production

**Files:**
- Modify: `test/test_npzd_mobi_parity.py`
- Modify: `veros_bgc/core/npzd_rules.py:103-136`
- Modify: `veros_bgc/core/npzd.py:535-592`

**Interfaces:**
- Produces: `calcite_production(vs, DIC, calcite)` and `calcite_production_alk(vs, alkalinity, calcite)` using phytoplankton mortality, zooplankton mortality, phytoplankton sloppy grazing, and zooplankton sloppy self-grazing.
- Removes: phytoplankton-only wrapper names `calcite_production_phyto` and `calcite_production_phyto_alk` from internal registration.

- [ ] **Step 1: Add literal MOBI calcite-term tests**

Append to `test/test_npzd_mobi_parity.py`:

```python
from veros_bgc.core.npzd_rules import calcite_production, calcite_production_alk


def test_calcite_production_contains_all_and_only_mobi_terms():
    vs = make_npzd_state()
    vs.mortality["phytoplankton"][...] = 1.0
    vs.mortality["zooplankton"][...] = 2.0
    vs.grazing["phytoplankton"][...] = 3.0
    vs.grazing["zooplankton"][...] = 4.0
    vs.grazing["detritus"][...] = 100.0
    vs.assimilation_efficiency = 0.5
    vs.capr = 1.0
    vs.redfield_ratio_CN = 1.0

    carbon = calcite_production(vs, "DIC", "caco3")
    alkalinity = calcite_production_alk(vs, "alkalinity", "caco3")

    np.testing.assert_allclose(carbon["DIC"], -6.5)
    np.testing.assert_allclose(carbon["caco3"], 6.5)
    np.testing.assert_allclose(alkalinity["alkalinity"], -13.0)
```

- [ ] **Step 2: Run the calcite test and verify RED**

Run:

```bash
.venv/bin/python -m pytest test/test_npzd_mobi_parity.py::test_calcite_production_contains_all_and_only_mobi_terms -q
```

Expected: FAIL because the current function requires a phytoplankton argument and omits both zooplankton terms.

- [ ] **Step 3: Implement the complete MOBI expression**

Replace the phytoplankton-specific calcite functions with:

```python
@veros_method(inline=True)
def calcite_production(vs, DIC, calcite):
    """Produce calcite from MOBI mortality and sloppy-grazing pathways."""
    dprca = (
        vs.mortality['phytoplankton']
        + vs.mortality['zooplankton']
        + (
            vs.grazing['phytoplankton']
            + vs.grazing['zooplankton']
        ) * (1 - vs.assimilation_efficiency)
    ) * vs.capr * vs.redfield_ratio_CN
    return {DIC: -dprca, calcite: dprca}


@veros_method(inline=True)
def calcite_production_alk(vs, alkalinity, calcite):
    """Apply the two-equivalent alkalinity cost of calcite production."""
    dic_change = calcite_production(vs, 'DIC', calcite)['DIC']
    return {alkalinity: 2 * dic_change}
```

Update the two calcite registrations in `setup_carbon_npzd_rules` to import and call these signatures directly. Do not reference `vs.grazing['detritus']`.

- [ ] **Step 4: Run the complete focused suites**

Run:

```bash
.venv/bin/python -m pytest test/test_npzd_mobi_parity.py -q
.venv/bin/python -m pytest test/test_npzd_regressions.py -q
```

Expected: all parity and existing calcite-conservation tests pass.

- [ ] **Step 5: Commit the calcite-production correction**

```bash
git add test/test_npzd_mobi_parity.py veros_bgc/core/npzd_rules.py veros_bgc/core/npzd.py
git commit -m "fix: include zooplankton calcite pathways"
```

---

### Task 6: Restore MOBI biological timing and setup constants

**Files:**
- Modify: `test/test_npzd_mobi_parity.py`
- Modify: `veros_bgc/core/npzd.py:13-14, 65-77`
- Modify: `veros_bgc/setup/bgc_global_4deg/bgc_global_four_degree.py:30-64`

**Interfaces:**
- Produces: `solar_declination(vs) -> float or numpy scalar`, using the MOBI `0.22` phase.
- Produces: bundled setup values `dt_bio == 10800.0`, `dt_tracer // dt_bio == 8`, and `dcaco3 == 6500`.

- [ ] **Step 1: Add seasonal and setup parity tests**

Append to `test/test_npzd_mobi_parity.py`:

```python
from veros_bgc.core.npzd import solar_declination
from veros_bgc.setup.bgc_global_4deg.bgc_global_four_degree import GlobalFourDegreeBGC


def test_solar_declination_uses_mobi_phase_at_model_time_zero():
    vs = VerosState(use_plugins=(veros_bgc,))
    vs.time = 0.0

    np.testing.assert_allclose(
        solar_declination(vs),
        -0.3929149002914755,
        rtol=0.0,
        atol=1e-15,
    )


def test_four_degree_setup_uses_eight_bio_steps_and_mobi_calcite_scale():
    setup = GlobalFourDegreeBGC()
    vs = VerosState(use_plugins=(veros_bgc,))

    setup.set_parameter(vs)

    assert vs.dt_tracer == 86400.0
    assert vs.dt_bio == 10800.0
    assert int(vs.dt_tracer // vs.dt_bio) == 8
    assert vs.dcaco3 == 6500
```

- [ ] **Step 2: Run setup tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest test/test_npzd_mobi_parity.py::test_solar_declination_uses_mobi_phase_at_model_time_zero test/test_npzd_mobi_parity.py::test_four_degree_setup_uses_eight_bio_steps_and_mobi_calcite_scale -q
```

Expected: import of `solar_declination` fails, and the setup still reports four biology steps and 2,500 m.

- [ ] **Step 3: Extract the seasonal calculation and correct setup values**

Add this helper immediately before `biogeochemistry`:

```python
@veros_method(inline=True)
def solar_declination(vs):
    """Return MOBI solar declination for the current model time."""
    return np.sin(
        (np.mod(vs.time * time.SECONDS_TO_X['years'], 1) - 0.22)
        * 2.0 * np.pi
    ) * 0.4
```

Replace the inline declaration inside `biogeochemistry` with:

```python
    declin = solar_declination(vs)
```

Change only these setup assignments:

```python
        vs.dt_bio = vs.dt_tracer // 8
        vs.dcaco3 = 6500
```

Do not edit the wind-stress forcing or wind reconstruction.

- [ ] **Step 4: Run all automated model tests**

Run:

```bash
.venv/bin/python -m pytest test/test_npzd_mobi_parity.py -q
.venv/bin/python -m pytest test/test_atmospherefluxes.py test/test_npzd_regressions.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the timing and constant parity changes**

```bash
git add test/test_npzd_mobi_parity.py veros_bgc/core/npzd.py veros_bgc/setup/bgc_global_4deg/bgc_global_four_degree.py
git commit -m "fix: restore MOBI biology timing and phase"
```

---

### Task 7: Add a tested 60-year NetCDF acceptance checker

**Files:**
- Create: `scripts/validate_60_year_run.py`
- Create: `test/test_validate_60_year_run.py`

**Interfaces:**
- Produces: `validate_snapshot(path: pathlib.Path) -> dict`, raising `ValidationError` on any failed acceptance item.
- Produces: `format_markdown(path: pathlib.Path, metrics: dict) -> str`, rendering commit-ready evidence.

- [ ] **Step 1: Write synthetic acceptance-checker tests**

Create `test/test_validate_60_year_run.py`:

```python
import h5netcdf
import numpy as np
import pytest

from scripts.validate_60_year_run import ValidationError, validate_snapshot


FILL_VALUE = -1e18


def write_snapshot(path):
    with h5netcdf.File(path, "w") as outfile:
        outfile.dimensions = {"Time": 60, "zt": 1, "yt": 1, "xt": 2}
        time = outfile.create_variable("Time", ("Time",), float)
        time[:] = np.arange(1, 61) * 360 * 86400.0
        for name, value in (("dic", 2100.0), ("alkalinity", 2300.0)):
            variable = outfile.create_variable(
                name,
                ("Time", "zt", "yt", "xt"),
                float,
                fillvalue=FILL_VALUE,
            )
            variable[:] = value
            variable[:, :, :, 1] = FILL_VALUE
        for name, value in (("pCO2", 400.0), ("hSWS", 3e-8), ("cflux", 0.0)):
            variable = outfile.create_variable(
                name,
                ("Time", "yt", "xt"),
                float,
                fillvalue=FILL_VALUE,
            )
            variable[:] = value
            variable[:, :, 1] = FILL_VALUE


def test_validate_snapshot_accepts_bounded_sixty_year_run(tmp_path):
    snapshot = tmp_path / "4deg.snapshot.nc"
    write_snapshot(snapshot)

    metrics = validate_snapshot(snapshot)

    assert metrics["snapshot_count"] == 60
    assert metrics["status"] == "PASS"
    assert metrics["dic_year_1_max"] == 2100.0
    assert metrics["dic_year_60_max"] == 2100.0


@pytest.mark.parametrize(
    "variable,index,value,message",
    [
        ("pCO2", (3, 0, 0), np.nan, "non-finite"),
        ("hSWS", (3, 0, 0), -1e-9, "outside"),
        ("dic", (3, 0, 0, 0), -1.0, "negative"),
        ("alkalinity", (59, 0, 0, 0), 23000.0, "ten times"),
        ("Time", (1,), 123.0, "annual"),
    ],
)
def test_validate_snapshot_rejects_failed_acceptance_item(
        tmp_path, variable, index, value, message):
    snapshot = tmp_path / "4deg.snapshot.nc"
    write_snapshot(snapshot)
    with h5netcdf.File(snapshot, "a") as outfile:
        outfile.variables[variable][index] = value

    with pytest.raises(ValidationError, match=message):
        validate_snapshot(snapshot)
```

- [ ] **Step 2: Run the validator tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest test/test_validate_60_year_run.py -q
```

Expected: collection fails because `scripts.validate_60_year_run` does not exist.

- [ ] **Step 3: Implement the acceptance checker**

Create `scripts/validate_60_year_run.py` with this behavior:

```python
#!/usr/bin/env python

import argparse
from pathlib import Path

import h5netcdf
import numpy as np


REQUIRED_VARIABLES = ("dic", "alkalinity", "pCO2", "hSWS", "cflux")


class ValidationError(RuntimeError):
    pass


def valid_values(variable, index=slice(None)):
    data = np.asarray(variable[index])
    fill_value = variable.attrs.get("_FillValue")
    valid = np.ones(data.shape, dtype=bool)
    if fill_value is not None:
        valid &= data != fill_value
    return data[valid]


def validate_snapshot(path):
    path = Path(path)
    with h5netcdf.File(path, "r") as infile:
        missing = [name for name in REQUIRED_VARIABLES if name not in infile.variables]
        if missing:
            raise ValidationError("missing required variables: {}".format(", ".join(missing)))

        snapshot_count = infile.variables["Time"].shape[0]
        if snapshot_count != 60:
            raise ValidationError(
                "expected 60 annual snapshots, found {}".format(snapshot_count)
            )

        snapshot_times = np.asarray(infile.variables["Time"][:])
        expected_times = np.arange(1, 61) * 360 * 86400.0
        if not np.allclose(snapshot_times, expected_times, rtol=0.0, atol=1e-6):
            raise ValidationError("snapshot times are not annual 360-day records")

        metrics = {"snapshot_count": snapshot_count}
        for name in REQUIRED_VARIABLES:
            values = valid_values(infile.variables[name])
            if not np.isfinite(values).all():
                raise ValidationError("{} contains non-finite wet values".format(name))
            metrics[name + "_min"] = float(values.min())
            metrics[name + "_max"] = float(values.max())

        hsws = valid_values(infile.variables["hSWS"])
        if np.any((hsws < 1e-10) | (hsws > 1e-6)):
            raise ValidationError("hSWS is outside [1e-10, 1e-6]")

        for name in ("dic", "alkalinity"):
            variable = infile.variables[name]
            values = valid_values(variable)
            if np.any(values < 0):
                raise ValidationError("{} contains negative wet values".format(name))

            year_1_max = float(valid_values(variable, 0).max())
            year_60_max = float(valid_values(variable, 59).max())
            metrics[name + "_year_1_max"] = year_1_max
            metrics[name + "_year_60_max"] = year_60_max
            if not year_60_max < 10 * year_1_max:
                raise ValidationError(
                    "{} year-60 maximum is not below ten times year 1".format(name)
                )

    metrics["status"] = "PASS"
    return metrics


def format_markdown(path, metrics):
    rows = [
        ("Snapshots", metrics["snapshot_count"]),
        ("DIC all-years min", metrics["dic_min"]),
        ("DIC all-years max", metrics["dic_max"]),
        ("DIC year-1 max", metrics["dic_year_1_max"]),
        ("DIC year-60 max", metrics["dic_year_60_max"]),
        ("Alkalinity all-years min", metrics["alkalinity_min"]),
        ("Alkalinity all-years max", metrics["alkalinity_max"]),
        ("Alkalinity year-1 max", metrics["alkalinity_year_1_max"]),
        ("Alkalinity year-60 max", metrics["alkalinity_year_60_max"]),
        ("pCO2 all-years min", metrics["pCO2_min"]),
        ("pCO2 all-years max", metrics["pCO2_max"]),
        ("hSWS all-years min", metrics["hSWS_min"]),
        ("hSWS all-years max", metrics["hSWS_max"]),
        ("cflux all-years min", metrics["cflux_min"]),
        ("cflux all-years max", metrics["cflux_max"]),
    ]
    lines = [
        "# MOBI parity 60-year validation",
        "",
        "Snapshot: `{}`".format(Path(path).resolve()),
        "",
        "| Check | Value |",
        "| --- | ---: |",
    ]
    lines.extend("| {} | {} |".format(label, value) for label, value in rows)
    lines.extend(("", "Acceptance result: **{}**".format(metrics["status"])))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()
    try:
        metrics = validate_snapshot(args.snapshot)
    except ValidationError as error:
        parser.error(str(error))
    if args.markdown:
        print(format_markdown(args.snapshot, metrics))
    else:
        print("PASS")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run validator and full unit suites**

Run:

```bash
.venv/bin/python -m pytest test/test_validate_60_year_run.py -q
.venv/bin/python -m pytest -q
```

Expected: every validator case and every repository test passes.

- [ ] **Step 5: Commit the acceptance checker**

```bash
git add scripts/validate_60_year_run.py test/test_validate_60_year_run.py
git commit -m "test: add 60-year integration acceptance checker"
```

---

### Task 8: Perform whole-branch verification and code review

**Files:**
- Review: every file changed since `master`
- Modify only when a review finding is reproduced by a failing test.

**Interfaces:**
- Produces: a reviewed branch with no unresolved correctness or specification findings.

- [ ] **Step 1: Run fresh automated verification**

Run each command separately:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q veros_bgc test scripts
git diff --check master...HEAD
git status --short --branch
```

Expected: pytest passes; compileall and diff check emit no output; status shows the feature branch with no uncommitted files.

- [ ] **Step 2: Audit approved non-changes**

Run:

```bash
git diff master...HEAD -- veros_bgc/core/atmospherefluxes.py veros_bgc/core/npzd.py veros_bgc/core/npzd_rules.py veros_bgc/setup/bgc_global_4deg/bgc_global_four_degree.py
```

Confirm from the diff that:

- wind speed is still reconstructed from `surface_taux`, `surface_tauy`, and `rho_0` with the existing multiplier;
- `xconv`, the `0.75` multiplier, flux sign, `1e3` conversion, and tracer-step application are unchanged;
- `dpCO2` remains diagnostic and unchanged;
- `rcak`, `rcab`, `bottom_mask`, monitor timestep indexing, and monitor halo slicing are unchanged.

- [ ] **Step 3: Request a whole-change review**

Invoke `superpowers:requesting-code-review` with the approved design, plan, `master` base, and feature-branch head. Require the reviewer to inspect root-solver behavior, rule ordering, stoichiometry, optional Bohrium string, setup parity, validator masking, and preservation of intentional differences.

Expected: no unresolved critical or important findings. Reproduce any reported defect with a focused failing test, fix it through RED/GREEN, rerun Step 1, and commit the correction before proceeding.

---

### Task 9: Run and prove the 60-year four-degree integration

**Files:**
- Generate outside Git: immutable attempt directories, checkpoints, snapshots, logs, `progress.json`, and a merged snapshot below `/home/nutrik/Documents/bugfix/validation/mobi-parity-60y/`
- Create after PASS: `docs/validation/2026-08-10-mobi-parity-60-year.md`

**Interfaces:**
- Consumes: the committed feature-branch package, clean pinned Veros v0.2.3 worktree, `scripts/run_segmented_60_year.py`, and `validate_snapshot` from Task 7.
- Produces: 60 immutable annual records, a manifest-authenticated merged snapshot, and committed compact validation evidence satisfying every approved acceptance criterion.

- [ ] **Step 1: Start the production protocol in a new empty absolute root**

The root below is dedicated to this run. It must be new and empty: do not reuse a prior run root, copy setup files, patch a setup, or add any manual segment/duration/snapshot overrides. The runner's production defaults are exactly 60 annual immutable segments of 360 model days, with one annual snapshot per segment.

Record the identities and confirm the root does not already exist:

```bash
git rev-parse HEAD
git -C /home/nutrik/Documents/bugfix/.worktrees/veros-v0.2.3 rev-parse HEAD
git status --short --branch
test ! -e /home/nutrik/Documents/bugfix/validation/mobi-parity-60y
```

Expected: Veros prints `bcbead95d56341f3ee9f30311c05a546674095e7`, the plugin worktree is clean, and the dedicated output root is absent before the new run. Start it only with the production command:

```bash
.venv/bin/python scripts/run_segmented_60_year.py new /home/nutrik/Documents/bugfix/validation/mobi-parity-60y
```

The runner creates a locked manifest at `progress.json`, records the feature commit/content fingerprint and pinned Veros identity, and commits each segment's snapshot, checkpoint, diagnostics, log, and SHA-256 hashes only after validation. It automatically validates checkpoint time, iteration, and time-level pointers; later segments restart only from a verified private copy of the prior committed checkpoint. Do not modify artifacts under the root.

- [ ] **Step 2: Resume safely after interruption until the run completes**

If the command is interrupted or reports fewer than 60 committed segments, continue with the same root and no overrides:

```bash
.venv/bin/python scripts/run_segmented_60_year.py resume /home/nutrik/Documents/bugfix/validation/mobi-parity-60y
```

Repeat this command as needed. On the final segment the runner automatically merges every committed annual snapshot, verifies the merge schema and exact time sequence, invokes the real 60-year `validate_snapshot` validator (not smoke validation), hashes the merged NetCDF, and stores its path and validation metrics in `progress.json`.

- [ ] **Step 3: Discover, independently validate, and commit the real acceptance evidence**

After the runner reports its merged snapshot, discover the authoritative path from the manifest rather than assuming a fixed filename:

```bash
MERGED_SNAPSHOT="$(.venv/bin/python -c 'import json; from pathlib import Path; root = Path("/home/nutrik/Documents/bugfix/validation/mobi-parity-60y"); print(json.loads((root / "progress.json").read_text(encoding="utf-8"))["merged_snapshot"])')"
.venv/bin/python scripts/validate_60_year_run.py "$MERGED_SNAPSHOT"
.venv/bin/python scripts/validate_60_year_run.py "$MERGED_SNAPSHOT" --markdown
```

Expected: `progress.json` has `status` `complete`, exactly 60 `committed_segments`, and the merged path/hash/validation. The validator prints `PASS`; its Markdown ends in `Acceptance result: **PASS**`; and the merged NetCDF contains exactly 60 annual `Time` records. It also proves finite required wet fields, bounded positive `hSWS`, nonnegative DIC/alkalinity, and year-60 DIC/alkalinity maxima below ten times year 1.

Use `apply_patch` to create `docs/validation/2026-08-10-mobi-parity-60-year.md` from the actual `--markdown` output. Add concrete provenance above the table: feature commit, Veros commit `bcbead95d56341f3ee9f30311c05a546674095e7`, runtime versions, the exact production and validator commands, the manifest-derived merged path, and confirmation that wind reconstruction was retained.

Do not add the NetCDF, immutable attempt files, logs, checkpoint copies, virtual environment, or asset cache to Git. Then run:

```bash
git add docs/validation/2026-08-10-mobi-parity-60-year.md
git commit -m "docs: record passing 60-year MOBI parity run"
```

---

### Task 10: Final verification and GitHub pull request

**Files:**
- Review: all tracked feature-branch changes
- No generated scientific output added to Git

**Interfaces:**
- Produces: pushed feature branch and GitHub pull request against `team-ocean/veros-bgc:master`.

- [ ] **Step 1: Invoke verification-before-completion and rerun every release gate**

Use `superpowers:verification-before-completion`, then run each command separately:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q veros_bgc test scripts
MERGED_SNAPSHOT="$(.venv/bin/python -c 'import json; from pathlib import Path; root = Path("/home/nutrik/Documents/bugfix/validation/mobi-parity-60y"); print(json.loads((root / "progress.json").read_text(encoding="utf-8"))["merged_snapshot"])')"
.venv/bin/python scripts/validate_60_year_run.py "$MERGED_SNAPSHOT"
git diff --check master...HEAD
git status --short --branch
git diff --stat master...HEAD
```

Expected: tests pass; compilation and diff check are silent; the manifest-derived merged snapshot prints `PASS` and has exactly 60 annual records; the worktree is clean; the diff contains source, tests, analysis/design/plan, and compact validation Markdown only.

- [ ] **Step 2: Inspect the final commit series and excluded files**

Run:

```bash
git log --oneline --decorate master..HEAD
git diff --name-only master...HEAD
```

Expected: focused documentation and RED/GREEN commits are visible; no `.venv`, `.nc`, `run.log`, copied setup, cache, bytecode, or asset file is listed.

- [ ] **Step 3: Publish intentionally through the GitHub workflow**

Invoke `github:yeet` to confirm scope, push `agent/fix-mobi-parity-carbon-divergence`, and create the pull request through the connected GitHub app. Target `master` in `team-ocean/veros-bgc`. The pull-request description must include:

- the negative-`hSWS` → invalid `pCO2` → inward `cflux` → DIC/alkalinity overflow failure chain;
- the fixed positive bracket and solver validation;
- all corrected MOBI discrepancies;
- explicit retention of wind reconstruction and verified non-bugs;
- focused and full test commands/results;
- the 60-year acceptance criteria and Markdown evidence link;
- the local-only status of large NetCDF output.

Expected: GitHub returns a pull-request URL. If `github:yeet` opens it as a draft, leave it as a draft only until connector-visible checks and the description are confirmed, then mark it ready for review without changing code.
