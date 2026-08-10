import numpy as np
import pytest

from veros.state import VerosState

import veros_bgc

from veros_bgc.core import atmospherefluxes
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
    np.testing.assert_allclose(roots[1:], roots[:-1], rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(pco2[1:], pco2[:-1], rtol=0.0, atol=1e-6)
    assert np.all((roots >= 1e-10) & (roots <= 1e-6))
    np.testing.assert_allclose(pco2, 1366.57, rtol=5e-4, atol=0.0)
    assert np.all(vs.dco2star[wet] < 0.0)


def test_co2calc_preserves_hsws_over_land_cells():
    vs = make_carbon_state()
    wet = vs.maskT[:, :, -1].astype(bool)
    land_hsws = -7e-8
    vs.hSWS[~wet] = land_hsws
    shape = vs.hSWS.shape

    atmospherefluxes.co2calc_SWS(
        vs,
        np.full(shape, 24.04129088),
        np.full(shape, 33.872667),
        np.full(shape, 2114.4811777e-3),
        np.full(shape, 2181.1384228e-3),
        280.0,
        1.0,
    )

    np.testing.assert_array_equal(vs.hSWS[~wet], land_hsws)


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
