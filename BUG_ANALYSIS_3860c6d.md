# Analysis of the long-term numerical divergence after `3860c6d`

## Executive summary

Commit `3860c6d285670ba166d5286c369a2ccc5091a474` fixed a real array-shape bug in bottom remineralization, but it fixed two symptoms rather than the underlying cause. The underlying problem is that `vs.bottom_mask` is declared with dtype `int8` even though several call sites use it as a Boolean index.

The update introduced by `3860c6d` is correct for finite array values. The long-term integration failure is caused by another use of the same integer mask in the implicit calcite redistribution setup:

```python
vs.rcak[vs.bottom_mask] = rcab[vs.bottom_mask]
```

Because `bottom_mask` is an integer array containing zeroes and ones, NumPy interprets it as an array of integer indices rather than as a Boolean mask. The intended ocean-bottom cells are therefore not updated. The resulting `rcak` field does not conserve calcite, which creates a persistent artificial sink of dissolved inorganic carbon (DIC) and alkalinity. On the vertical grid bundled with the 4-degree setup, only approximately 4.7% to 87.5% of calcite production is returned to the modeled water column, depending on water depth, instead of 100%.

There are also two independent indexing errors in the calculation of the replacement bottom coefficient `rcab`. Both must be corrected together with the mask dtype. Changing only the mask dtype makes the faulty `rcab` values active and can produce either under- or over-redistribution.

No source files were modified during the investigation. This document is the only requested repository addition.

## Scope and repository versions

The following repositories were inspected:

- Veros-BGC: `/home/nutrik/Documents/bugfix/veros-bgc`
- Veros: `/home/nutrik/Documents/bugfix/veros`

The regression commit is:

```text
commit: 3860c6d285670ba166d5286c369a2ccc5091a474
parent: 63c354d297d3794a3a47c6ec7687d69163d26b1b
date:   2020-10-28T11:18:30+01:00
title:  fix bug where temporary arrays were blown up to 5 dimensions
```

The compatible Veros version was inspected at the requested tag without checking it out or modifying its worktree:

```text
tag:    v0.2.3
commit: bcbead95d56341f3ee9f30311c05a546674095e7
```

Veros v0.2.3 allocates plugin variables using their declared dtype. Its distributed implementation explicitly supports both `int8` and Boolean arrays (`MPI.CHAR` and `MPI.BOOL`, respectively), so declaring `bottom_mask` as Boolean is supported by this version.

## What commit `3860c6d` changed

The commit changed only two files and two coupled operations.

### Primary-rule application

Before the commit, every rule update used its boundary directly as an index:

```python
vs.temporary_tracers[key][boundary] += value * vs.dt_bio
```

After the commit, tuple and slice boundaries are still indexed, while an array boundary is applied by multiplication:

```python
if isinstance(boundary, (tuple, slice)):
    vs.temporary_tracers[key][boundary] += value * vs.dt_bio
else:
    vs.temporary_tracers[key] += value * vs.dt_bio * boundary
```

Location: `veros_bgc/core/npzd.py`, lines 158-164.

### Bottom remineralization

Before the commit, bottom remineralization indexed `npzd_export`:

```python
return {sink: vs.npzd_export[source][vs.bottom_mask] * scale}
```

After the commit, it returns a full three-dimensional field multiplied by the mask:

```python
return {sink: vs.npzd_export[source] * vs.bottom_mask * scale}
```

Location: `veros_bgc/core/npzd_rules.py`, lines 173-186.

For finite inputs, the combination of these two new operations is exactly equivalent to updating only the cells selected by a proper Boolean mask. An in-memory numerical comparison produced zero difference. Consequently, this new update branch is not itself the source of the gradual divergence.

One secondary robustness issue remains: multiplication is not a strict substitute for selection when invalid floating-point values are present, because `0 * NaN` and `0 * Inf` are not guaranteed to become zero. This was not needed to explain the observed conservation failure, but using a genuinely Boolean mask or `where` operation is safer if invalid values can occur outside the active domain.

## Why the original code produced five-dimensional arrays

`bottom_mask` is declared in `veros_bgc/variables.py` as follows:

```python
('bottom_mask', Variable(
    'Bottom mask', T_GRID, '', 'Bottom mask', dtype='int8'
)),
```

It is populated with a Boolean expression in `setup_basic_npzd_rules`:

```python
vs.bottom_mask[:, :, :] = (
    np.arange(vs.nz)[np.newaxis, np.newaxis, :]
    == (vs.kbot - 1)[:, :, np.newaxis]
)
```

Assignment into the declared `int8` variable converts `False` and `True` to integer zero and one. NumPy does not infer that such an integer array is intended to be a mask.

For an array with shape `(nx, ny, nz)`, this operation:

```python
array[bottom_mask]
```

uses every zero and one in `bottom_mask` as an index into the first dimension. Its output shape is:

```text
(nx, ny, nz, ny, nz)
```

An in-memory reproduction with an array of shape `(6, 4, 3)` produced:

```text
integer-mask result shape: (6, 4, 3, 4, 3)
Boolean-mask result shape: (2,)
```

This exactly explains the five-dimensional temporary arrays mentioned in the commit title.

## Remaining integration bug: `rcak` bottom indexing

The same integer mask is still used as an index in `setup_carbon_npzd_rules`:

```python
# merge bottom into level k and reset every cell outside ocean
vs.rcak[vs.bottom_mask] = rcab[vs.bottom_mask]
vs.rcak[...] *= vs.maskT
```

Location: `veros_bgc/core/npzd.py`, lines 554-556.

With the current `int8` dtype, this statement does not select cells for which `bottom_mask == 1`. It performs advanced integer indexing and writes through the first axis at indices zero and one. The actual interior bottom cells, whose x indices normally start at two because of Veros ghost cells, retain the uncorrected open-water dissolution profile.

An in-memory assignment test confirmed that none of the intended interior bottom cells received its corresponding `rcab` value when the `int8` mask was used. Converting the same mask to Boolean made the assignment update exactly the intended cells.

Git history shows that the faulty calcite setup predates `3860c6d`; it originates in commit `f79920f3`. The regression commit therefore did not create this line. Instead, it made bottom detritus remineralization operate correctly, which sustained nutrient recycling and biological production and exposed or accelerated the latent calcite conservation error.

## How the mask bug causes long-term divergence

The bundled 4-degree setup enables both NPZD and carbon:

```python
vs.enable_npzd = True
vs.enable_carbon = True
```

Its `npzd.yml` selects both `group_npzd_basic` and `group_carbon_implicit_caco3`, so all of the following rules are active.

1. `calcite_production` subtracts `dprca` from DIC and stores the same amount in the temporary `caco3` tracer.
2. `calcite_production_alk` subtracts `2 * dprca` from alkalinity.
3. `post_redistribute_calcite` integrates the temporary calcite production over depth and redistributes it using `vs.rcak`.
4. `post_redistribute_calcite_alk` restores twice the redistributed amount to alkalinity.
5. `pre_reset_calcite` resets temporary calcite before the next model timestep.

For this process to conserve carbon and alkalinity in each wet column, the following invariant must hold:

```python
np.sum(vs.rcak * vs.dzt[np.newaxis, np.newaxis, :], axis=2) == 1
```

The base exponential profile represents dissolution within the resolved water column. The special `rcab` value at the bottom cell must additionally capture the fraction that would otherwise sink below the seafloor.

Because the bottom replacement is currently not applied to normal interior cells, the depth-integrated fraction is less than one. Using the exact vertical spacing and `dcaco3 = 2500` from the bundled 4-degree setup gave:

```text
minimum recovered fraction: 0.046866212922495265
maximum recovered fraction: 0.8750697878014175
```

Thus approximately 12.5% to 95.3% of calcite-associated DIC and alkalinity can be lost in each affected column, depending on water depth. The configured calcite production ratio is relatively small (`capr = 0.022`), so the inventory error accumulates gradually rather than necessarily failing immediately. This is consistent with divergence after a multi-decadal integration.

The carbonate chemistry routine uses the evolving DIC and alkalinity in a nonlinear root solve for `hSWS`. The solver does not verify that its initial interval brackets a root and returns its current value after the maximum iteration count without a convergence error. A slowly drifting DIC/alkalinity state therefore has a direct route to invalid carbonate chemistry and eventual numerical divergence.

## Additional `rcab` calculation errors

The existing code is:

```python
rcab = np.empty_like(vs.dic[..., 0])
rcab[:, : -1] = 1 / vs.dzt[-1]
rcab[:, :, :-1] = np.exp(zw[:-1] / vs.dcaco3) / vs.dzt[1:]
```

It has two indexing errors.

### Missing vertical index

This line:

```python
rcab[:, : -1] = 1 / vs.dzt[-1]
```

selects all x cells, all but the last y row, and every vertical level. The intended operation is to initialize only the top vertical level:

```python
rcab[:, :, -1] = 1 / vs.dzt[-1]
```

Because `rcab` is created with `empty_like`, any top-level elements not selected by the erroneous assignment remain uninitialized.

### Shifted coordinate and cell thickness

This line:

```python
rcab[:, :, :-1] = np.exp(zw[:-1] / vs.dcaco3) / vs.dzt[1:]
```

uses the lower face of the bottom cell and divides by the thickness of the cell above. To capture all calcite reaching the bottom cell, the bottom-cell coefficient must use the upper face of that cell and the thickness of the bottom cell itself:

```python
rcab[:, :, :-1] = np.exp(zw[1:] / vs.dcaco3) / vs.dzt[:-1]
```

Changing only `bottom_mask` to Boolean would activate the existing erroneous `rcab` values. On the bundled vertical grid, the resulting column-integrated recovery ranges from approximately 0.9626 to 1.3551, so some columns would still lose calcite while others would create it.

With both `rcab` corrections, the tested recovery fraction was between `0.9999999999999999` and `1.0` for every possible wet bottom level.

## Required source changes

### 1. Declare `bottom_mask` as Boolean

In `veros_bgc/variables.py`, change:

```python
dtype='int8'
```

to:

```python
dtype='bool'
```

This fixes the underlying semantic mismatch everywhere `bottom_mask` is used as an index.

### 2. Correct the bottom calcite coefficients

In `setup_carbon_npzd_rules` in `veros_bgc/core/npzd.py`, replace the current `rcab` construction with:

```python
rcab = np.empty_like(vs.dic[..., 0])
rcab[:, :, -1] = 1 / vs.dzt[-1]
rcab[:, :, :-1] = np.exp(zw[1:] / vs.dcaco3) / vs.dzt[:-1]
```

The existing assignment can then correctly use the Boolean mask:

```python
vs.rcak[vs.bottom_mask] = rcab[vs.bottom_mask]
```

Both changes are required. Applying either one alone does not restore conservation.

### 3. Keep or simplify the `3860c6d` bottom-update code

The multiplication-based update added by `3860c6d` is correct for finite values and may be retained. Once `bottom_mask` is Boolean, it remains equivalent to the intended masked update.

It could later be refactored to make the boundary contract explicit, but such a refactor is not necessary to fix the identified conservation defect. If invalid values outside the active domain are possible, a Boolean `where` operation is preferable to multiplication because multiplication does not suppress `NaN` or infinity.

## Diagnostic indexing problems

`veros_bgc/diagnostics/npzd_monitor.py` contains two additional integer-mask problems:

```python
vs.npzd_tracers[var][vs.maskT[:, :, -1]].mean()
vs.npzd_tracers[var][vs.bottom_mask].mean()
```

`maskT` is intentionally `int8` in Veros v0.2.3, so the first expression also performs integer advanced indexing. Making the plugin's `bottom_mask` Boolean fixes the mask type in the second expression, but both expressions still operate on the full tracer array, including all stored timesteps. They do not explicitly select the current surface or bottom concentration represented by their log labels.

They should follow the equivalent pattern:

```python
surface_mask = vs.maskT[:, :, -1].astype(np.bool)
surface = vs.npzd_tracers[var][:, :, -1, vs.tau]
surface_mean = surface[surface_mask].mean()

bottom = vs.npzd_tracers[var][:, :, :, vs.tau]
bottom_mean = bottom[vs.bottom_mask].mean()
```

These diagnostic errors do not update model state and therefore are not the cause of integration divergence, but their output cannot currently be trusted.

## Recommended regression tests

### Mask dtype and indexing semantics

After setup, verify:

```python
assert vs.bottom_mask.dtype == np.dtype('bool')
assert vs.rcak[vs.bottom_mask].ndim == 1
assert vs.rcak[vs.bottom_mask].size == np.count_nonzero(vs.bottom_mask)
```

The test should include bottom cells whose x index is greater than one so it detects accidental integer indexing.

### Calcite column conservation

For every wet column, verify:

```python
column_fraction = np.sum(
    vs.rcak * vs.dzt[np.newaxis, np.newaxis, :],
    axis=2,
)

np.testing.assert_allclose(
    column_fraction[vs.kbot > 0],
    1.0,
    rtol=0.0,
    atol=1e-14,
)
```

The test should exercise multiple bottom levels, including shallow and deep columns.

### Bottom remineralization conservation

On a small synthetic grid with physical transport disabled, initialize detritus in one bottom cell and run one biological substep. Verify that:

- detritus export is removed only from the bottom cell;
- PO4 receives the exported amount scaled by `redfield_ratio_PN`;
- DIC receives the exported amount scaled by `redfield_ratio_CN` when carbon is enabled;
- no non-bottom cell changes through a bottom-boundary rule;
- the finite-field result agrees with an explicit Boolean-index reference calculation.

### Diagnostic selection

Fill surface, subsurface, current-timestep, and old-timestep cells with distinct values. Verify that surface and bottom diagnostics use only the requested spatial cells at `vs.tau`.

### Short integration invariant

A shortened deterministic integration should monitor:

- global DIC inventory;
- global alkalinity inventory;
- integrated temporary calcite production;
- integrated calcite redistribution;
- finiteness and convergence of `hSWS`, `pCO2`, and `cflux`.

In the absence of external carbon flux, integrated calcite production and redistribution should balance to numerical precision.

## Verification performed during this investigation

The investigation used read-only Git inspection and in-memory NumPy calculations. No source patch was applied.

The final numerical checks established:

```text
int8 index shape: (6, 4, 3, 4, 3)
new commit update equals intended finite Boolean update: PASS
int8 rcak assignment misses intended cells: PASS
current calcite recovery range: 0.046866212922495265 0.8750697878014175
recovery after mask-only fix with source rcab: 0.9625533695224905 1.3550960368129603
recovery with corrected rcab: 0.9999999999999999 1.0
```

The full 50-year model integration was not rerun. The compatible Veros runtime and external 4-degree forcing assets were not installed in the inspected workspace. The diagnosis instead reproduces the exact indexing semantics and proves the violated column-conservation invariant using the setup's actual vertical grid and parameters.

## Final diagnosis

The source-level defect is a family of inconsistent mask and vertical-index assumptions in Veros-BGC:

1. `bottom_mask` is stored as `int8` but used as a Boolean index.
2. Commit `3860c6d` corrected that misuse in bottom remineralization without correcting the mask at its source.
3. The remaining integer-index use prevents the calcite profile from being closed at the seafloor.
4. The replacement bottom profile itself has two independent indexing errors.
5. Together these errors violate DIC and alkalinity conservation and provide the mechanism for delayed numerical divergence.

The required fix is to make `bottom_mask` Boolean and correct both `rcab` indices, accompanied by a column-conservation regression test.
