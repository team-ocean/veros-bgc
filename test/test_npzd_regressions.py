import numpy as np
from loguru import logger

import veros_bgc
from veros.logs import setup_logging
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

    setup_logging(loglevel="warning")
    messages = []
    sink = logger.add(messages.append, level=0, format="{message}")
    try:
        monitor.output(vs)
    finally:
        logger.remove(sink)

    rendered = "\n".join(str(message) for message in messages)
    assert "mean po4 surface concentration: 6.5 mmol/m^3" in rendered
    assert "mean po4 bottom concentration: 106.5 mmol/m^3" in rendered
