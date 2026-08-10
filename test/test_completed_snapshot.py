import h5netcdf
import numpy as np
import pytest

import veros_bgc
from veros.state import VerosState

from veros_bgc.diagnostics.completed_snapshot import CompletedStateSnapshot
from veros_bgc.setup.bgc_global_4deg.bgc_global_four_degree import GlobalFourDegreeBGC


ANNUAL_SECONDS = 360 * 86400.0
CARBON_DIAGNOSTIC_FIELDS = (
    "wind_speed",
    "hSWS",
    "co2star",
    "dco2star",
    "pCO2",
    "dpCO2",
    "cflux",
)


def make_sentinel_state():
    vs = VerosState(use_plugins=(veros_bgc,))
    vs.nx, vs.ny, vs.nz = 2, 1, 1
    vs.enable_npzd = True
    vs.enable_carbon = True
    vs.allocate_variables()
    vs.maskT[...] = False
    vs.maskT[2, 2, :] = True

    vs.temp[..., vs.tau] = 10.0
    vs.salt[..., vs.tau] = 30.0
    vs.dic[..., vs.tau] = 1900.0
    vs.alkalinity[..., vs.tau] = 2000.0

    vs.temp[..., vs.taup1] = 24.04129088
    vs.salt[..., vs.taup1] = 33.872667
    vs.dic[..., vs.taup1] = 2114.4811777
    vs.alkalinity[..., vs.taup1] = 2181.1384228

    vs.hSWS[...] = 7e-9
    vs.pCO2[...] = -123.0
    vs.cflux[...] = 456.0
    vs.forc_temp_surface[...] = 1.0
    vs.surface_taux[...] = 0.1
    vs.surface_tauy[...] = 0.05
    vs.atmospheric_co2 = 280.0
    vs.time = ANNUAL_SECONDS
    return vs


@pytest.mark.parametrize("output_days", (360.0, 21600.0))
def test_completed_snapshot_labels_and_writes_one_consistent_completed_state(
    tmp_path, output_days
):
    vs = make_sentinel_state()
    vs.time = output_days * 86400.0
    output_path = tmp_path / "sentinel.snapshot.nc"
    diagnostic = CompletedStateSnapshot(vs)
    diagnostic.output_path = str(output_path)
    diagnostic.output_frequency = ANNUAL_SECONDS
    diagnostic.output_variables = [
        "dic",
        "alkalinity",
        "pCO2",
        "hSWS",
        "cflux",
        "maskT",
    ]

    original_pointers = (vs.taum1, vs.tau, vs.taup1)
    original_fields = {
        name: (getattr(vs, name), getattr(vs, name).copy())
        for name in CARBON_DIAGNOSTIC_FIELDS
    }

    diagnostic.initialize(vs)
    diagnostic.output(vs)

    assert (vs.taum1, vs.tau, vs.taup1) == original_pointers
    for name, (original_object, original_values) in original_fields.items():
        assert getattr(vs, name) is original_object
        np.testing.assert_array_equal(getattr(vs, name), original_values)

    with h5netcdf.File(output_path, "r") as infile:
        np.testing.assert_array_equal(infile.variables["Time"][:], [output_days])
        assert tuple(infile.variables["maskT"].dimensions) == ("zt", "yt", "xt")
        np.testing.assert_array_equal(infile.variables["maskT"][:], [[[1, 0]]])
        np.testing.assert_allclose(infile.variables["dic"][0, 0, 0, 0], 2114.4811777)
        np.testing.assert_allclose(infile.variables["alkalinity"][0, 0, 0, 0], 2181.1384228)
        np.testing.assert_allclose(
            infile.variables["hSWS"][0, 0, 0],
            2.884768663987437e-08,
            rtol=0.0,
            atol=1e-18,
        )
        np.testing.assert_allclose(
            infile.variables["pCO2"][0, 0, 0],
            1366.5741001291553,
            rtol=0.0,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            infile.variables["cflux"][0, 0, 0],
            -0.000922145353804244,
            rtol=0.0,
            atol=1e-15,
        )


def test_four_degree_annual_snapshot_uses_completed_state_diagnostic():
    setup = GlobalFourDegreeBGC()
    vs = setup.state
    setup.set_parameter(vs)
    vs.allocate_variables()
    vs.create_diagnostics()

    setup.set_diagnostics(vs)

    assert isinstance(vs.diagnostics["snapshot"], CompletedStateSnapshot)
    assert vs.diagnostics["snapshot"].output_frequency == ANNUAL_SECONDS
