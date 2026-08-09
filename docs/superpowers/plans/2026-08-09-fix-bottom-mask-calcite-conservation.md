# Bottom Mask and Calcite Conservation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore conservative bottom-boundary and calcite behavior in Veros-BGC and make NPZD surface/bottom diagnostics select the current timestep correctly.

**Architecture:** Keep the existing rule engine and the multiplication-based boundary update introduced by `3860c6d`. Correct the mask at its declaration, correct the two vertical indices used to construct the seafloor calcite coefficient, and make diagnostics slice space and time before applying masks. Exercise the changes through a real Veros v0.2.3 plugin state rather than source-text assertions or framework mocks.

**Tech Stack:** Python 3.11, NumPy 1.23.5, pytest 8, pytest-forked, Veros v0.2.3 (`bcbead95d56341f3ee9f30311c05a546674095e7`).

## Global Constraints

- Implement only in `/home/nutrik/Documents/bugfix/veros-bgc`; do not modify `/home/nutrik/Documents/bugfix/veros`.
- Validate compatibility against Veros tag `v0.2.3`, commit `bcbead95d56341f3ee9f30311c05a546674095e7`.
- Preserve the multiplication-based primary-rule update and full-field bottom remineralization introduced by `3860c6d285670ba166d5286c369a2ccc5091a474`.
- Make `bottom_mask` Boolean at its source rather than adding casts at every use.
- Calcite redistribution must integrate to exactly one, within floating-point tolerance, for every wet column including a one-cell-deep column.
- Diagnostics must report only active wet cells from `vs.tau`; stored historical/future timesteps must not influence their means.
- Include `BUG_ANALYSIS_3860c6d.md` in the pull request as the approved design and root-cause record.
- Do not depend on external forcing assets or a 50-year integration in the automated tests.

---

### Task 1: Add regression coverage and repair bottom-boundary behavior

**Files:**
- Create: `test/test_npzd_regressions.py`
- Modify: `veros_bgc/variables.py:10-12`
- Modify: `veros_bgc/core/npzd.py:545-556`
- Modify: `veros_bgc/diagnostics/npzd_monitor.py:160-164`

**Interfaces:**
- Consumes: `veros.state.VerosState(use_plugins=(veros_bgc,))`, `setup_basic_npzd_rules(vs)`, `setup_carbon_npzd_rules(vs)`, and `NPZDMonitor.output(vs)` from Veros v0.2.3 and Veros-BGC.
- Produces: a Boolean `vs.bottom_mask`; an `rcak` field satisfying `sum(rcak * dzt, axis=2) == 1` in every wet column; surface and bottom diagnostic means drawn from the current timestep only.

- [ ] **Step 1: Write a real Veros v0.2.3 regression fixture and failing tests**

Create `test/test_npzd_regressions.py` with a small allocated plugin state. Use literal grid inputs and expected means so the test does not reproduce the implementation formula to calculate its expectations.

```python
import numpy as np
from loguru import logger

import veros_bgc
from veros.state import VerosState

from veros_bgc.core.npzd import (
    setup_basic_npzd_rules,
    setup_carbon_npzd_rules,
)
from veros_bgc.diagnostics.npzd_monitor import NPZDMonitor


DZT = np.array([40.0, 30.0, 20.0, 10.0])
ZW = np.array([-60.0, -30.0, -10.0, 0.0])
KBOT = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [1, 3, 4],
    [2, 4, 1],
])


def make_state(kbot=KBOT):
    vs = VerosState(use_plugins=(veros_bgc,))
    vs.nx, vs.ny, vs.nz = kbot.shape[0], kbot.shape[1], DZT.size
    vs.enable_npzd = True
    vs.enable_carbon = True
    vs.dcaco3 = 2500.0
    vs.allocate_variables()

    vs.dzt[:] = DZT
    vs.zw[:] = ZW
    vs.area_t[:] = 1.0
    vs.kbot[2:-2, 2:-2] = kbot

    levels = np.arange(vs.nz)[np.newaxis, np.newaxis, :]
    vs.maskT[:] = (vs.kbot[..., np.newaxis] > 0) & (
        vs.kbot[..., np.newaxis] - 1 <= levels
    )
    return vs


def test_bottom_mask_is_boolean_and_selects_one_cell_per_wet_column():
    vs = make_state()
    setup_basic_npzd_rules(vs)

    assert vs.bottom_mask.dtype == np.dtype("bool")
    assert np.count_nonzero(vs.bottom_mask) == 12
    assert vs.bottom_mask[2, 2, 0]
    assert vs.bottom_mask[3, 4, 3]
    assert vs.bottom_mask[5, 3, 3]
    assert vs.bottom_mask[5, 4, 0]


def test_calcite_redistribution_conserves_every_wet_column():
    vs = make_state()
    setup_basic_npzd_rules(vs)
    setup_carbon_npzd_rules(vs)

    column_fraction = np.sum(
        vs.rcak * vs.dzt[np.newaxis, np.newaxis, :],
        axis=2,
    )
    np.testing.assert_allclose(
        column_fraction[2:-2, 2:-2],
        np.ones(KBOT.shape),
        rtol=0.0,
        atol=1e-14,
    )
    assert np.count_nonzero(vs.rcak[vs.maskT == 0]) == 0


def test_monitor_reports_current_surface_and_bottom_means():
    diagnostic_kbot = np.array([
        [1, 2, 3],
        [2, 3, 1],
        [1, 3, 2],
        [2, 1, 3],
    ])
    vs = make_state(diagnostic_kbot)
    setup_basic_npzd_rules(vs)

    vs.po4[:] = 1000.0
    vs.po4[:, :, :, vs.tau] = 0.0
    vs.po4[2:-2, 2:-2, -1, vs.tau] = np.arange(1.0, 13.0).reshape(4, 3)
    bottom_values = np.arange(101.0, 113.0).reshape(4, 3)
    for i in range(4):
        for j in range(3):
            vs.po4[i + 2, j + 2, diagnostic_kbot[i, j] - 1, vs.tau] = bottom_values[i, j]

    monitor = NPZDMonitor(None)
    monitor.surface_out = ["po4"]
    monitor.bottom_out = ["po4"]
    monitor.initialize(vs)

    messages = []
    sink = logger.add(messages.append, level=0, format="{message}")
    try:
        monitor.output(vs)
    finally:
        logger.remove(sink)

    rendered = "\n".join(str(message) for message in messages)
    assert "mean po4 surface concentration: 6.5 mmol/m^3" in rendered
    assert "mean po4 bottom concentration: 106.5 mmol/m^3" in rendered
```

- [ ] **Step 2: Run each regression test and verify the expected failures**

Run:

```bash
.venv/bin/python -m pytest test/test_npzd_regressions.py::test_bottom_mask_is_boolean_and_selects_one_cell_per_wet_column -q
.venv/bin/python -m pytest test/test_npzd_regressions.py::test_calcite_redistribution_conserves_every_wet_column -q
.venv/bin/python -m pytest test/test_npzd_regressions.py::test_monitor_reports_current_surface_and_bottom_means -q
```

Expected:

- The mask test fails because `bottom_mask.dtype` is `int8`.
- The conservation test fails because each wet-column integral differs from `1.0`.
- The diagnostic test fails because integer advanced indexing and unsliced timesteps produce means other than `6.5` and `106.5`.

- [ ] **Step 3: Make the bottom mask Boolean**

In `veros_bgc/variables.py`, change only the `bottom_mask` dtype:

```python
('bottom_mask', Variable(
    'Bottom mask', T_GRID, '', 'Bottom mask', dtype='bool'
)),
```

- [ ] **Step 4: Correct both calcite bottom-coefficient indices**

In `setup_carbon_npzd_rules` in `veros_bgc/core/npzd.py`, use the top level and the upper face/current thickness for all lower bottom cells:

```python
rcab = np.empty_like(vs.dic[..., 0])
rcab[:, :, -1] = 1 / vs.dzt[-1]
rcab[:, :, :-1] = np.exp(zw[1:] / vs.dcaco3) / vs.dzt[:-1]
```

Keep the existing Boolean-indexed merge and mask reset:

```python
vs.rcak[vs.bottom_mask] = rcab[vs.bottom_mask]
vs.rcak[...] *= vs.maskT
```

- [ ] **Step 5: Slice diagnostic data before applying masks**

In `veros_bgc/diagnostics/npzd_monitor.py`, replace the direct four-dimensional indexing with explicit current-timestep surface and bottom fields:

```python
for var in self.surface_out:
    surface_mask = vs.maskT[:, :, -1].astype(np.bool)
    surface = vs.npzd_tracers[var][:, :, -1, vs.tau]
    logger.diagnostic(' mean {} surface concentration: {} mmol/m^3'.format(
        var, surface[surface_mask].mean()))

for var in self.bottom_out:
    bottom = vs.npzd_tracers[var][:, :, :, vs.tau]
    logger.diagnostic(' mean {} bottom concentration: {} mmol/m^3'.format(
        var, bottom[vs.bottom_mask].mean()))
```

- [ ] **Step 6: Run focused and full verification**

Run:

```bash
.venv/bin/python -m pytest test/test_npzd_regressions.py -q
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q veros_bgc test
git diff --check
```

Expected: three tests pass, full pytest reports three passing tests, compileall exits zero, and `git diff --check` has no output.

- [ ] **Step 7: Commit the implementation**

Stage only the task files and commit:

```bash
git add test/test_npzd_regressions.py veros_bgc/variables.py veros_bgc/core/npzd.py veros_bgc/diagnostics/npzd_monitor.py
git commit -m "fix bottom calcite conservation"
```
