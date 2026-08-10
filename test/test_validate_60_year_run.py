import h5netcdf
import numpy as np
import pytest

from scripts.validate_60_year_run import ValidationError, validate_snapshot


FILL_VALUE = -1e18


def write_snapshot(path):
    with h5netcdf.File(path, "w") as outfile:
        outfile.dimensions = {"Time": 60, "zt": 1, "yt": 1, "xt": 2}
        time = outfile.create_variable("Time", ("Time",), float)
        time[:] = np.arange(1, 61) * 360 * 86400.0
        for name, value in (("dic", 2100.0), ("alkalinity", 2300.0)):
            variable = outfile.create_variable(
                name,
                ("Time", "zt", "yt", "xt"),
                float,
                fillvalue=FILL_VALUE,
            )
            variable[:] = value
            variable[:, :, :, 1] = FILL_VALUE
        for name, value in (("pCO2", 400.0), ("hSWS", 3e-8), ("cflux", 0.0)):
            variable = outfile.create_variable(
                name,
                ("Time", "yt", "xt"),
                float,
                fillvalue=FILL_VALUE,
            )
            variable[:] = value
            variable[:, :, 1] = FILL_VALUE


def test_validate_snapshot_accepts_bounded_sixty_year_run(tmp_path):
    snapshot = tmp_path / "4deg.snapshot.nc"
    write_snapshot(snapshot)

    metrics = validate_snapshot(snapshot)

    assert metrics["snapshot_count"] == 60
    assert metrics["status"] == "PASS"
    assert metrics["dic_year_1_max"] == 2100.0
    assert metrics["dic_year_60_max"] == 2100.0


@pytest.mark.parametrize(
    "variable,index,value,message",
    [
        ("pCO2", (3, 0, 0), np.nan, "non-finite"),
        ("hSWS", (3, 0, 0), -1e-9, "outside"),
        ("dic", (3, 0, 0, 0), -1.0, "negative"),
        ("alkalinity", (59, 0, 0, 0), 23000.0, "ten times"),
        ("Time", (1,), 123.0, "annual"),
    ],
)
def test_validate_snapshot_rejects_failed_acceptance_item(
    tmp_path, variable, index, value, message
):
    snapshot = tmp_path / "4deg.snapshot.nc"
    write_snapshot(snapshot)
    with h5netcdf.File(snapshot, "a") as outfile:
        outfile.variables[variable][index] = value

    with pytest.raises(ValidationError, match=message):
        validate_snapshot(snapshot)
