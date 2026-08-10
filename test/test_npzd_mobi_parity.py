import numpy as np
import pytest

import veros_bgc
from veros.state import VerosState

from veros_bgc.core import atmospherefluxes
from veros_bgc.core.npzd import setupNPZD
from veros_bgc.core.npzd_rules import calcite_production, calcite_production_alk


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
        "zooplankton": zero.copy(),
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
    vs.dt_tracer = 10.0
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


@pytest.mark.parametrize(
    "mortality_key,grazing_key,rate,assimilation_efficiency,"
    "expected_dic,expected_calcite,expected_alkalinity",
    [
        ("phytoplankton", None, 2.0, 0.1, -3.0, 3.0, -6.0),
        ("zooplankton", None, 3.0, 0.8, -4.5, 4.5, -9.0),
        (None, "phytoplankton", 4.0, 0.2, -4.8, 4.8, -9.6),
        (None, "zooplankton", 7.0, 0.6, -4.2, 4.2, -8.4),
    ],
)
def test_calcite_production_contains_all_and_only_mobi_terms(
        mortality_key, grazing_key, rate, assimilation_efficiency, expected_dic,
        expected_calcite, expected_alkalinity):
    vs = make_npzd_state()
    zero_process_rates(vs)
    vs.grazing["detritus"][...] = 100.0
    vs.assimilation_efficiency = assimilation_efficiency
    vs.capr = 0.25
    vs.redfield_ratio_CN = 6.0

    if mortality_key is not None:
        vs.mortality[mortality_key][...] = rate
    if grazing_key is not None:
        vs.grazing[grazing_key][...] = rate

    carbon = calcite_production(vs, "DIC", "caco3")
    alkalinity = calcite_production_alk(vs, "alkalinity", "caco3")

    np.testing.assert_allclose(carbon["DIC"], expected_dic)
    np.testing.assert_allclose(carbon["caco3"], expected_calcite)
    np.testing.assert_allclose(alkalinity["alkalinity"], expected_alkalinity)
