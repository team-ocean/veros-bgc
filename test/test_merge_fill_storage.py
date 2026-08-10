import h5netcdf

from veros_bgc.snapshot_merge import merge_snapshots


def _snapshot(path, day):
    with h5netcdf.File(path, "w") as outfile:
        outfile.dimensions = {"Time": None, "x": 2}
        outfile.resize_dimension("Time", 1)
        time = outfile.create_variable("Time", ("Time",), "f8", chunks=(1,))
        time.attrs["units"] = "days"
        time[:] = [day]
        mask = outfile.create_variable(
            "maskT",
            ("x",),
            "i1",
            chunks=(2,),
            compression="gzip",
            compression_opts=1,
            fillvalue=-128,
        )
        mask.attrs["_FillValue"] = 0
        mask[:] = [1, 0]


def test_merge_preserves_hdf5_fill_property_separately_from_fillvalue_attribute(tmp_path):
    first = tmp_path / "first.nc"
    second = tmp_path / "second.nc"
    merged = (tmp_path / "merged.nc").resolve()
    _snapshot(first, 1)
    _snapshot(second, 2)

    merge_snapshots([first, second], merged, expected_records=2, interval_days=1)

    with h5netcdf.File(merged, "r") as infile:
        mask = infile.variables["maskT"]
        assert mask._h5ds.fillvalue == -128
        assert mask.attrs["_FillValue"] == 0
