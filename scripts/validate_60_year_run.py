#!/usr/bin/env python

import argparse
from pathlib import Path

import h5netcdf
import numpy as np


REQUIRED_VARIABLES = ("dic", "alkalinity", "pCO2", "hSWS", "cflux")
EXPECTED_DIMENSIONS = {
    "Time": ("Time",),
    "dic": ("Time", "zt", "yt", "xt"),
    "alkalinity": ("Time", "zt", "yt", "xt"),
    "pCO2": ("Time", "yt", "xt"),
    "hSWS": ("Time", "yt", "xt"),
    "cflux": ("Time", "yt", "xt"),
}
SNAPSHOT_COUNT = 60


class ValidationError(RuntimeError):
    pass


def validate_numeric_dtype(name, variable):
    if not np.issubdtype(variable.dtype, np.number):
        raise ValidationError("{} must have a numeric dtype".format(name))


def wet_mask(variable, index):
    data = np.asarray(variable[index])
    fill_value = variable.attrs.get("_FillValue")
    mask = np.ones(data.shape, dtype=bool)
    if fill_value is not None:
        mask &= data != fill_value
    return data, mask


def validate_field(name, variable, snapshot_count):
    expected_dimensions = EXPECTED_DIMENSIONS[name]
    if tuple(variable.dimensions) != expected_dimensions:
        raise ValidationError(
            "{} has dimensions {}, expected {}".format(
                name, tuple(variable.dimensions), expected_dimensions
            )
        )
    validate_numeric_dtype(name, variable)
    if variable.shape[0] != snapshot_count:
        raise ValidationError(
            "{} has {} records, expected {}".format(name, variable.shape[0], snapshot_count)
        )

    reference_mask = None
    values_by_record = []
    for record in range(snapshot_count):
        data, mask = wet_mask(variable, record)
        values = data[mask]
        if values.size == 0:
            raise ValidationError(
                "{} record {} contains no wet values".format(name, record + 1)
            )
        if not np.isfinite(values).all():
            raise ValidationError(
                "{} record {} contains non-finite wet values".format(name, record + 1)
            )
        if reference_mask is None:
            reference_mask = mask
        elif not np.array_equal(mask, reference_mask):
            raise ValidationError("{} wet mask changes between annual records".format(name))
        if name == "hSWS" and np.any((values < 1e-10) | (values > 1e-6)):
            raise ValidationError("hSWS is outside [1e-10, 1e-6]")
        if name in ("dic", "alkalinity") and np.any(values < 0):
            raise ValidationError("{} contains negative wet values".format(name))
        values_by_record.append(values)

    return reference_mask, values_by_record


def validate_snapshot(path):
    path = Path(path)
    try:
        with h5netcdf.File(path, "r") as infile:
            required = ("Time",) + REQUIRED_VARIABLES
            missing = [name for name in required if name not in infile.variables]
            if missing:
                raise ValidationError("missing required variables: {}".format(", ".join(missing)))

            time = infile.variables["Time"]
            if tuple(time.dimensions) != EXPECTED_DIMENSIONS["Time"]:
                raise ValidationError(
                    "Time has dimensions {}, expected {}".format(
                        tuple(time.dimensions), EXPECTED_DIMENSIONS["Time"]
                    )
                )
            validate_numeric_dtype("Time", time)
            if time.attrs.get("units") != "days":
                raise ValidationError("Time units must be days")

            snapshot_count = time.shape[0]
            if snapshot_count != SNAPSHOT_COUNT:
                raise ValidationError(
                    "expected {} annual snapshots, found {}".format(
                        SNAPSHOT_COUNT, snapshot_count
                    )
                )

            snapshot_times = np.asarray(time[:])
            expected_times = np.arange(1, SNAPSHOT_COUNT + 1) * 360.0
            if not np.allclose(snapshot_times, expected_times, rtol=0.0, atol=1e-6):
                raise ValidationError("snapshot times are not annual 360-day records")

            masks = {}
            record_values = {}
            metrics = {"snapshot_count": snapshot_count}
            for name in REQUIRED_VARIABLES:
                masks[name], record_values[name] = validate_field(
                    name, infile.variables[name], snapshot_count
                )
                metrics[name + "_min"] = float(
                    min(values.min() for values in record_values[name])
                )
                metrics[name + "_max"] = float(
                    max(values.max() for values in record_values[name])
                )

            if not np.array_equal(masks["dic"], masks["alkalinity"]):
                raise ValidationError("dic and alkalinity wet masks do not match")
            if not (
                np.array_equal(masks["pCO2"], masks["hSWS"])
                and np.array_equal(masks["pCO2"], masks["cflux"])
            ):
                raise ValidationError("pCO2, hSWS, and cflux wet masks do not match")

            for name in ("dic", "alkalinity"):
                year_1_max = float(record_values[name][0].max())
                year_60_max = float(record_values[name][-1].max())
                metrics[name + "_year_1_max"] = year_1_max
                metrics[name + "_year_60_max"] = year_60_max
                if not year_60_max < 10 * year_1_max:
                    raise ValidationError(
                        "{} year-60 maximum is not below ten times year 1".format(name)
                    )
    except OSError as error:
        raise ValidationError("could not read snapshot {}: {}".format(path, error)) from error

    metrics["status"] = "PASS"
    return metrics


def format_markdown(path, metrics):
    rows = [
        ("Snapshots", metrics["snapshot_count"]),
        ("DIC all-years min", metrics["dic_min"]),
        ("DIC all-years max", metrics["dic_max"]),
        ("DIC year-1 max", metrics["dic_year_1_max"]),
        ("DIC year-60 max", metrics["dic_year_60_max"]),
        ("Alkalinity all-years min", metrics["alkalinity_min"]),
        ("Alkalinity all-years max", metrics["alkalinity_max"]),
        ("Alkalinity year-1 max", metrics["alkalinity_year_1_max"]),
        ("Alkalinity year-60 max", metrics["alkalinity_year_60_max"]),
        ("pCO2 all-years min", metrics["pCO2_min"]),
        ("pCO2 all-years max", metrics["pCO2_max"]),
        ("hSWS all-years min", metrics["hSWS_min"]),
        ("hSWS all-years max", metrics["hSWS_max"]),
        ("cflux all-years min", metrics["cflux_min"]),
        ("cflux all-years max", metrics["cflux_max"]),
    ]
    lines = [
        "# MOBI parity 60-year validation",
        "",
        "Snapshot: `{}`".format(Path(path).resolve()),
        "",
        "| Check | Value |",
        "| --- | ---: |",
    ]
    lines.extend("| {} | {} |".format(label, value) for label, value in rows)
    lines.extend(("", "Acceptance result: **{}**".format(metrics["status"])))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()
    try:
        metrics = validate_snapshot(args.snapshot)
    except ValidationError as error:
        parser.error(str(error))
    if args.markdown:
        print(format_markdown(args.snapshot, metrics))
    else:
        print("PASS")


if __name__ == "__main__":
    main()
