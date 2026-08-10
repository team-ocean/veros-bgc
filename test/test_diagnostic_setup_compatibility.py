from pathlib import Path

from veros_bgc.setup.bgc_global_4deg.bgc_global_four_degree import GlobalFourDegreeBGC


ANNUAL_SECONDS = 360 * 86400.0


def test_real_four_degree_setup_initializes_configured_netcdf_diagnostics(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    setup = GlobalFourDegreeBGC()

    setup.setup()

    diagnostics = setup.state.diagnostics
    assert {
        name: type(diagnostics[name]).__name__
        for name in ("snapshot", "averages", "energy", "overturning")
    } == {
        "snapshot": "CompletedStateSnapshot",
        "averages": "CompatibleAverages",
        "energy": "CompatibleEnergy",
        "overturning": "CompatibleOverturning",
    }
    assert diagnostics["snapshot"].output_frequency == ANNUAL_SECONDS
    assert diagnostics["averages"].output_frequency == ANNUAL_SECONDS
    assert diagnostics["averages"].sampling_frequency == 86400
    assert diagnostics["energy"].output_frequency == ANNUAL_SECONDS
    assert diagnostics["energy"].sampling_frequency == 86400
    assert diagnostics["overturning"].output_frequency == ANNUAL_SECONDS
    assert diagnostics["overturning"].sampling_frequency == setup.state.dt_tracer
    assert sorted(path.name for path in Path(tmp_path).glob("4deg.*.nc")) == [
        "4deg.averages.nc",
        "4deg.energy.nc",
        "4deg.overturning.nc",
        "4deg.snapshot.nc",
    ]
