import subprocess
import sys
from pathlib import Path

import h5netcdf
import numpy as np
import pytest

from scripts.validate_60_year_run import ValidationError, validate_snapshot


FILL_VALUE = -1e18
STRING_DTYPE = "S16"
FIELD_DIMENSIONS = {
    "dic": ("Time", "zt", "yt", "xt"),
    "alkalinity": ("Time", "zt", "yt", "xt"),
    "pCO2": ("Time", "yt", "xt"),
    "hSWS": ("Time", "yt", "xt"),
    "cflux": ("Time", "yt", "xt"),
}
FIELD_VALUES = {
    "dic": 2100.0,
    "alkalinity": 2300.0,
    "pCO2": 400.0,
    "hSWS": 3e-8,
    "cflux": 0.0,
}
SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_60_year_run.py"


def write_nonnumeric_snapshot(path, target):
    with h5netcdf.File(path, "w") as outfile:
        outfile.dimensions = {"Time": 60, "zt": 1, "yt": 1, "xt": 2}
        time_dtype = STRING_DTYPE if target == "Time" else float
        time = outfile.create_variable("Time", ("Time",), time_dtype)
        time.attrs["units"] = "days"
        time.attrs["time_origin"] = "01-JAN-1900 00:00:00"
        if target == "Time":
            time[:] = np.asarray([b"day"] * 60, dtype=STRING_DTYPE)
        else:
            time[:] = np.arange(1, 61) * 360.0

        mask = outfile.create_variable("maskT", ("zt", "yt", "xt"), np.int8)
        mask[:] = 0
        mask[0, 0, 0] = 1

        for name, dimensions in FIELD_DIMENSIONS.items():
            if name == target:
                variable = outfile.create_variable(name, dimensions, STRING_DTYPE)
                variable[:] = np.full(variable.shape, b"bad", dtype=STRING_DTYPE)
                continue
            variable = outfile.create_variable(
                name,
                dimensions,
                float,
                fillvalue=FILL_VALUE,
            )
            variable[:] = FIELD_VALUES[name]
            variable[..., 1] = FILL_VALUE


@pytest.mark.parametrize("target", ("Time",) + tuple(FIELD_DIMENSIONS))
def test_validate_snapshot_rejects_nonnumeric_time_or_required_field(tmp_path, target):
    snapshot = tmp_path / "4deg.snapshot.nc"
    write_nonnumeric_snapshot(snapshot, target)

    with pytest.raises(ValidationError, match="{}.*numeric".format(target)):
        validate_snapshot(snapshot)


def test_cli_reports_nonnumeric_field_validation_error(tmp_path):
    snapshot = tmp_path / "4deg.snapshot.nc"
    write_nonnumeric_snapshot(snapshot, "pCO2")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(snapshot)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "error: pCO2 must have a numeric dtype" in result.stderr
