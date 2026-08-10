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
