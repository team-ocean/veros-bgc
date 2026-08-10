import subprocess
import sys
from pathlib import Path

import h5netcdf
import numpy as np
import pytest

from scripts.validate_60_year_run import (
    ValidationError,
    format_markdown,
    validate_snapshot,
)


FILL_VALUE = -1e18
RECORDS = 60
THREED_DIMENSIONS = ("Time", "zt", "yt", "xt")
SURFACE_DIMENSIONS = ("Time", "yt", "xt")
SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_60_year_run.py"


def series_for(name, record_count):
    record = np.arange(1, record_count + 1, dtype=float)
    if name == "dic":
        return 2000.0 + record
    if name == "alkalinity":
        return 2300.0 + record
    if name == "pCO2":
        return 400.0 + record
    if name == "hSWS":
        return np.linspace(3e-8, 9e-7, record_count)
    return -30.0 + record


def field_data(name, shape, record_count):
    data = np.full(shape, series_for(name, record_count)[0], dtype=float)
    if shape[0] == record_count:
        data[:] = series_for(name, record_count).reshape(
            (record_count,) + (1,) * (len(shape) - 1)
        )
    if shape[-1] == 2:
        data[..., 1] = FILL_VALUE
    return data


def write_snapshot(
    path,
    record_count=RECORDS,
    missing_variables=(),
    variable_dimensions=None,
    time_dimensions=("Time",),
    time_units="days",
):
    variable_dimensions = variable_dimensions or {}
    missing_variables = set(missing_variables)
    dimensions = {"Time": record_count, "zt": 1, "yt": 1, "xt": 2}
    with h5netcdf.File(path, "w") as outfile:
        outfile.dimensions = dimensions
        if "Time" not in missing_variables:
            time = outfile.create_variable("Time", time_dimensions, float)
            time.attrs["units"] = time_units
            time.attrs["time_origin"] = "01-JAN-1900 00:00:00"
            time[:] = (np.arange(1, record_count + 1) * 360.0).reshape(
                (record_count,) + (1,) * (len(time_dimensions) - 1)
            )
        for name in ("dic", "alkalinity", "pCO2", "hSWS", "cflux"):
            if name in missing_variables:
                continue
            default_dimensions = (
                THREED_DIMENSIONS if name in ("dic", "alkalinity") else SURFACE_DIMENSIONS
            )
            dimensions = variable_dimensions.get(name, default_dimensions)
            variable = outfile.create_variable(name, dimensions, float, fillvalue=FILL_VALUE)
            variable[:] = field_data(name, variable.shape, record_count)


def field_index(variable, record=3):
    return (record,) + (0,) * (variable.ndim - 1)


def test_validate_snapshot_accepts_veros_days_schema(tmp_path):
    snapshot = tmp_path / "4deg.snapshot.nc"
    write_snapshot(snapshot)

    metrics = validate_snapshot(snapshot)

    assert metrics == {
        "snapshot_count": 60,
        "dic_min": 2001.0,
        "dic_max": 2060.0,
        "alkalinity_min": 2301.0,
        "alkalinity_max": 2360.0,
        "pCO2_min": 401.0,
        "pCO2_max": 460.0,
        "hSWS_min": 3e-8,
        "hSWS_max": 9e-7,
        "cflux_min": -29.0,
        "cflux_max": 30.0,
        "dic_year_1_max": 2001.0,
        "dic_year_60_max": 2060.0,
        "alkalinity_year_1_max": 2301.0,
        "alkalinity_year_60_max": 2360.0,
        "status": "PASS",
    }


@pytest.mark.parametrize(
    "index,value",
    [(1, 360.0), (24, 9001.0), (3, np.nan), (4, np.inf)],
    ids=("duplicate", "misaligned", "nan", "infinite"),
)
def test_validate_snapshot_rejects_duplicate_misaligned_or_nonfinite_time(
    tmp_path, index, value
):
    snapshot = tmp_path / "4deg.snapshot.nc"
    write_snapshot(snapshot)
    with h5netcdf.File(snapshot, "a") as outfile:
        outfile.variables["Time"][index] = value

    with pytest.raises(ValidationError, match="annual"):
        validate_snapshot(snapshot)


def test_validate_snapshot_rejects_offset_time_coordinate(tmp_path):
    snapshot = tmp_path / "4deg.snapshot.nc"
    write_snapshot(snapshot)
    with h5netcdf.File(snapshot, "a") as outfile:
        outfile.variables["Time"][:] += 0.5

    with pytest.raises(ValidationError, match="annual"):
        validate_snapshot(snapshot)


def test_validate_snapshot_rejects_surface_field_without_time_dimension(tmp_path):
    snapshot = tmp_path / "4deg.snapshot.nc"
    write_snapshot(snapshot, variable_dimensions={"pCO2": ("yt", "xt")})

    with pytest.raises(ValidationError, match="pCO2.*dimensions"):
        validate_snapshot(snapshot)


def test_validate_snapshot_rejects_empty_intermediate_annual_record(tmp_path):
    snapshot = tmp_path / "4deg.snapshot.nc"
    write_snapshot(snapshot)
    with h5netcdf.File(snapshot, "a") as outfile:
        outfile.variables["dic"][17, :, :, :] = FILL_VALUE

    with pytest.raises(ValidationError, match="dic.*record 18.*no wet values"):
        validate_snapshot(snapshot)


@pytest.mark.parametrize("missing", [("Time",), ("cflux",)])
def test_validate_snapshot_rejects_missing_coordinate_or_required_field(tmp_path, missing):
    snapshot = tmp_path / "4deg.snapshot.nc"
    write_snapshot(snapshot, missing_variables=missing)

    with pytest.raises(ValidationError, match="missing"):
        validate_snapshot(snapshot)


def test_validate_snapshot_rejects_wrong_snapshot_count(tmp_path):
    snapshot = tmp_path / "4deg.snapshot.nc"
    write_snapshot(snapshot, record_count=59)

    with pytest.raises(ValidationError, match="expected 60"):
        validate_snapshot(snapshot)


@pytest.mark.parametrize(
    "variable_dimensions",
    [{"dic": ("Time", "yt", "xt")}, {"pCO2": ("Time", "zt", "yt", "xt")}],
)
def test_validate_snapshot_rejects_wrong_required_field_dimensions(
    tmp_path, variable_dimensions
):
    snapshot = tmp_path / "4deg.snapshot.nc"
    write_snapshot(snapshot, variable_dimensions=variable_dimensions)

    with pytest.raises(ValidationError, match="dimensions"):
        validate_snapshot(snapshot)


def test_validate_snapshot_rejects_wrong_time_dimensions(tmp_path):
    snapshot = tmp_path / "4deg.snapshot.nc"
    write_snapshot(snapshot, time_dimensions=("Time", "zt"))

    with pytest.raises(ValidationError, match="Time.*dimensions"):
        validate_snapshot(snapshot)


def test_validate_snapshot_rejects_non_days_time_units(tmp_path):
    snapshot = tmp_path / "4deg.snapshot.nc"
    write_snapshot(snapshot, time_units="seconds")

    with pytest.raises(ValidationError, match="Time.*units"):
        validate_snapshot(snapshot)


def test_validate_snapshot_rejects_time_varying_mask(tmp_path):
    snapshot = tmp_path / "4deg.snapshot.nc"
    write_snapshot(snapshot)
    with h5netcdf.File(snapshot, "a") as outfile:
        outfile.variables["dic"][17, 0, 0, 1] = 2018.0

    with pytest.raises(ValidationError, match="dic.*mask.*changes"):
        validate_snapshot(snapshot)


def test_validate_snapshot_rejects_mismatched_threed_masks(tmp_path):
    snapshot = tmp_path / "4deg.snapshot.nc"
    write_snapshot(snapshot)
    with h5netcdf.File(snapshot, "a") as outfile:
        outfile.variables["alkalinity"][:, 0, 0, 1] = series_for("alkalinity", 60)

    with pytest.raises(ValidationError, match="dic.*alkalinity.*masks"):
        validate_snapshot(snapshot)


def test_validate_snapshot_rejects_mismatched_surface_masks(tmp_path):
    snapshot = tmp_path / "4deg.snapshot.nc"
    write_snapshot(snapshot)
    with h5netcdf.File(snapshot, "a") as outfile:
        outfile.variables["cflux"][:, 0, 1] = series_for("cflux", 60)

    with pytest.raises(ValidationError, match="pCO2.*hSWS.*cflux.*masks"):
        validate_snapshot(snapshot)


@pytest.mark.parametrize("name", ("dic", "alkalinity", "pCO2", "hSWS", "cflux"))
def test_validate_snapshot_rejects_nonfinite_wet_values_in_each_required_field(tmp_path, name):
    snapshot = tmp_path / "4deg.snapshot.nc"
    write_snapshot(snapshot)
    with h5netcdf.File(snapshot, "a") as outfile:
        outfile.variables[name][field_index(outfile.variables[name])] = np.nan

    with pytest.raises(ValidationError, match="{}.*non-finite".format(name)):
        validate_snapshot(snapshot)


def test_validate_snapshot_rejects_infinite_required_field_value(tmp_path):
    snapshot = tmp_path / "4deg.snapshot.nc"
    write_snapshot(snapshot)
    with h5netcdf.File(snapshot, "a") as outfile:
        outfile.variables["pCO2"][3, 0, 0] = np.inf

    with pytest.raises(ValidationError, match="pCO2.*non-finite"):
        validate_snapshot(snapshot)


def test_validate_snapshot_accepts_exact_hsws_bounds(tmp_path):
    snapshot = tmp_path / "4deg.snapshot.nc"
    write_snapshot(snapshot)
    with h5netcdf.File(snapshot, "a") as outfile:
        outfile.variables["hSWS"][0, 0, 0] = 1e-10
        outfile.variables["hSWS"][59, 0, 0] = 1e-6

    assert validate_snapshot(snapshot)["status"] == "PASS"


@pytest.mark.parametrize("value", (9e-11, 1.01e-6))
def test_validate_snapshot_rejects_hsws_outside_inclusive_bounds(tmp_path, value):
    snapshot = tmp_path / "4deg.snapshot.nc"
    write_snapshot(snapshot)
    with h5netcdf.File(snapshot, "a") as outfile:
        outfile.variables["hSWS"][3, 0, 0] = value

    with pytest.raises(ValidationError, match="hSWS.*outside"):
        validate_snapshot(snapshot)


def test_validate_snapshot_rejects_negative_dic(tmp_path):
    snapshot = tmp_path / "4deg.snapshot.nc"
    write_snapshot(snapshot)
    with h5netcdf.File(snapshot, "a") as outfile:
        outfile.variables["dic"][3, 0, 0, 0] = -1.0

    with pytest.raises(ValidationError, match="dic.*negative"):
        validate_snapshot(snapshot)


def test_validate_snapshot_rejects_negative_alkalinity(tmp_path):
    snapshot = tmp_path / "4deg.snapshot.nc"
    write_snapshot(snapshot)
    with h5netcdf.File(snapshot, "a") as outfile:
        outfile.variables["alkalinity"][3, 0, 0, 0] = -1.0

    with pytest.raises(ValidationError, match="alkalinity.*negative"):
        validate_snapshot(snapshot)


@pytest.mark.parametrize("value", (20010.0, 20011.0))
def test_validate_snapshot_rejects_dic_growth_at_or_above_ten_times_year_one(
    tmp_path, value
):
    snapshot = tmp_path / "4deg.snapshot.nc"
    write_snapshot(snapshot)
    with h5netcdf.File(snapshot, "a") as outfile:
        outfile.variables["dic"][59, 0, 0, 0] = value

    with pytest.raises(ValidationError, match="dic.*ten times"):
        validate_snapshot(snapshot)


@pytest.mark.parametrize("value", (23010.0, 23011.0))
def test_validate_snapshot_rejects_alkalinity_growth_at_or_above_ten_times_year_one(
    tmp_path, value
):
    snapshot = tmp_path / "4deg.snapshot.nc"
    write_snapshot(snapshot)
    with h5netcdf.File(snapshot, "a") as outfile:
        outfile.variables["alkalinity"][59, 0, 0, 0] = value

    with pytest.raises(ValidationError, match="alkalinity.*ten times"):
        validate_snapshot(snapshot)


def test_format_markdown_renders_every_literal_metric_row(tmp_path):
    snapshot = tmp_path / "4deg.snapshot.nc"
    write_snapshot(snapshot)

    markdown = format_markdown(snapshot, validate_snapshot(snapshot))

    assert [line for line in markdown.splitlines() if line.startswith("|")] == [
        "| Check | Value |",
        "| --- | ---: |",
        "| Snapshots | 60 |",
        "| DIC all-years min | 2001.0 |",
        "| DIC all-years max | 2060.0 |",
        "| DIC year-1 max | 2001.0 |",
        "| DIC year-60 max | 2060.0 |",
        "| Alkalinity all-years min | 2301.0 |",
        "| Alkalinity all-years max | 2360.0 |",
        "| Alkalinity year-1 max | 2301.0 |",
        "| Alkalinity year-60 max | 2360.0 |",
        "| pCO2 all-years min | 401.0 |",
        "| pCO2 all-years max | 460.0 |",
        "| hSWS all-years min | 3e-08 |",
        "| hSWS all-years max | 9e-07 |",
        "| cflux all-years min | -29.0 |",
        "| cflux all-years max | 30.0 |",
    ]
    assert "# MOBI parity 60-year validation" in markdown
    assert "Acceptance result: **PASS**" in markdown


def test_cli_renders_markdown_for_valid_snapshot(tmp_path):
    snapshot = tmp_path / "4deg.snapshot.nc"
    write_snapshot(snapshot)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(snapshot), "--markdown"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "| hSWS all-years max | 9e-07 |" in result.stdout
    assert result.stderr == ""


def test_cli_reports_validation_error_deterministically(tmp_path):
    snapshot = tmp_path / "4deg.snapshot.nc"
    write_snapshot(snapshot, missing_variables=("pCO2",))

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(snapshot)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "error: missing required variables: pCO2" in result.stderr
