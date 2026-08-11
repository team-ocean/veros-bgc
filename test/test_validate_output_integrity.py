import h5netcdf
import numpy as np
import pytest

from scripts.validate_60_year_run import ValidationError, validate_snapshot


FILL_VALUE = -1e18
RECORDS = 60
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


def write_snapshot(path, include_mask=True, mask_dimensions=("zt", "yt", "xt"), mask_dtype=np.int8):
    with h5netcdf.File(path, "w") as outfile:
        outfile.dimensions = {"Time": RECORDS, "zt": 1, "yt": 1, "xt": 2}
        time = outfile.create_variable("Time", ("Time",), float)
        time.attrs["units"] = "days"
        time[:] = np.arange(1, RECORDS + 1) * 360.0

        if include_mask:
            mask = outfile.create_variable("maskT", mask_dimensions, mask_dtype)
            if np.issubdtype(np.dtype(mask_dtype), np.number):
                mask[:] = 0
                mask[(0,) * len(mask_dimensions)] = 1
            else:
                mask[:] = np.full(mask.shape, b"bad", dtype=mask_dtype)

        for name, dimensions in FIELD_DIMENSIONS.items():
            variable = outfile.create_variable(name, dimensions, float, fillvalue=FILL_VALUE)
            variable[:] = FIELD_VALUES[name]
            variable[..., 1] = FILL_VALUE


def test_validate_snapshot_requires_authoritative_maskt(tmp_path):
    snapshot = tmp_path / "snapshot.nc"
    write_snapshot(snapshot, include_mask=False)

    with pytest.raises(ValidationError, match="missing.*maskT"):
        validate_snapshot(snapshot)


def test_validate_snapshot_requires_exact_static_maskt_dimensions(tmp_path):
    snapshot = tmp_path / "snapshot.nc"
    write_snapshot(snapshot, mask_dimensions=("Time", "zt", "yt", "xt"))

    with pytest.raises(ValidationError, match="maskT.*dimensions"):
        validate_snapshot(snapshot)


def test_validate_snapshot_rejects_nonnumeric_maskt(tmp_path):
    snapshot = tmp_path / "snapshot.nc"
    write_snapshot(snapshot, mask_dtype="S16")

    with pytest.raises(ValidationError, match="maskT.*numeric"):
        validate_snapshot(snapshot)


@pytest.mark.parametrize("name", tuple(FIELD_DIMENSIONS))
def test_validate_snapshot_compares_each_required_field_to_maskt(tmp_path, name):
    snapshot = tmp_path / "snapshot.nc"
    write_snapshot(snapshot)
    with h5netcdf.File(snapshot, "a") as outfile:
        variable = outfile.variables[name]
        variable[..., 0] = FILL_VALUE
        variable[..., 1] = FIELD_VALUES[name]

    with pytest.raises(ValidationError, match="{}.*maskT".format(name)):
        validate_snapshot(snapshot)


def test_validate_snapshot_rejects_disjoint_threed_family_that_matches_itself(tmp_path):
    snapshot = tmp_path / "snapshot.nc"
    write_snapshot(snapshot)
    with h5netcdf.File(snapshot, "a") as outfile:
        for name in ("dic", "alkalinity"):
            outfile.variables[name][..., 0] = FILL_VALUE
            outfile.variables[name][..., 1] = FIELD_VALUES[name]

    with pytest.raises(ValidationError, match="dic.*maskT"):
        validate_snapshot(snapshot)


def test_validate_snapshot_rejects_disjoint_surface_family_that_matches_itself(tmp_path):
    snapshot = tmp_path / "snapshot.nc"
    write_snapshot(snapshot)
    with h5netcdf.File(snapshot, "a") as outfile:
        for name in ("pCO2", "hSWS", "cflux"):
            outfile.variables[name][..., 0] = FILL_VALUE
            outfile.variables[name][..., 1] = FIELD_VALUES[name]

    with pytest.raises(ValidationError, match="pCO2.*maskT"):
        validate_snapshot(snapshot)


def test_validate_snapshot_rejects_uniformly_missing_authoritative_wet_cell(tmp_path):
    snapshot = tmp_path / "snapshot.nc"
    write_snapshot(snapshot)
    with h5netcdf.File(snapshot, "a") as outfile:
        outfile.variables["maskT"][0, 0, 1] = 1

    with pytest.raises(ValidationError, match="dic.*maskT"):
        validate_snapshot(snapshot)
