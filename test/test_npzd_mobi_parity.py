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
    zero = np.zeros_like(vs.excretion_total)
    vs.net_primary_production = {"phytoplankton": zero.copy()}
    vs.recycled = {
        "phytoplankton": zero.copy(),
        "detritus": zero.copy(),
    }
    vs.mortality = {
        "phytoplankton": zero.copy(),
        "zooplankton": zero.copy(),
    }
    vs.grazing = {
        "phytoplankton": zero.copy(),
        "detritus": zero.copy(),
    }
    vs.sloppy_feeding = {"phytoplankton": zero.copy(), "zooplankton": zero.copy()}
    vs.npzd_export = {"detritus": zero.copy()}
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
