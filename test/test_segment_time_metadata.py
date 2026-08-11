import h5netcdf

from veros_bgc.setup.bgc_global_4deg.bgc_global_four_degree import GlobalFourDegreeBGC


def test_pinned_compatible_outputs_persist_time_units(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    setup = GlobalFourDegreeBGC()
    setup.setup()

    for name in ("snapshot", "averages", "energy", "overturning"):
        with h5netcdf.File(tmp_path / "4deg.{}.nc".format(name), "r") as infile:
            assert infile.variables["Time"].attrs["units"] == "days"
