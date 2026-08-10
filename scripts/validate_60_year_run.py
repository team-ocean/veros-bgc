#!/usr/bin/env python

import argparse
from pathlib import Path

import h5netcdf
import numpy as np


REQUIRED_VARIABLES = ("dic", "alkalinity", "pCO2", "hSWS", "cflux")


class ValidationError(RuntimeError):
    pass


def valid_values(variable, index=slice(None)):
    data = np.asarray(variable[index])
    fill_value = variable.attrs.get("_FillValue")
    valid = np.ones(data.shape, dtype=bool)
    if fill_value is not None:
        valid &= data != fill_value
    return data[valid]


def validate_snapshot(path):
    path = Path(path)
    with h5netcdf.File(path, "r") as infile:
        missing = [name for name in REQUIRED_VARIABLES if name not in infile.variables]
        if missing:
            raise ValidationError("missing required variables: {}".format(", ".join(missing)))

        snapshot_count = infile.variables["Time"].shape[0]
        if snapshot_count != 60:
            raise ValidationError(
                "expected 60 annual snapshots, found {}".format(snapshot_count)
            )

        snapshot_times = np.asarray(infile.variables["Time"][:])
        expected_times = np.arange(1, 61) * 360 * 86400.0
        if not np.allclose(snapshot_times, expected_times, rtol=0.0, atol=1e-6):
            raise ValidationError("snapshot times are not annual 360-day records")

        metrics = {"snapshot_count": snapshot_count}
        for name in REQUIRED_VARIABLES:
            values = valid_values(infile.variables[name])
            if not np.isfinite(values).all():
                raise ValidationError("{} contains non-finite wet values".format(name))
            metrics[name + "_min"] = float(values.min())
            metrics[name + "_max"] = float(values.max())

        hsws = valid_values(infile.variables["hSWS"])
        if np.any((hsws < 1e-10) | (hsws > 1e-6)):
            raise ValidationError("hSWS is outside [1e-10, 1e-6]")

        for name in ("dic", "alkalinity"):
            variable = infile.variables[name]
            values = valid_values(variable)
            if np.any(values < 0):
                raise ValidationError("{} contains negative wet values".format(name))

            year_1_max = float(valid_values(variable, 0).max())
            year_60_max = float(valid_values(variable, 59).max())
            metrics[name + "_year_1_max"] = year_1_max
            metrics[name + "_year_60_max"] = year_60_max
            if not year_60_max < 10 * year_1_max:
                raise ValidationError(
                    "{} year-60 maximum is not below ten times year 1".format(name)
                )

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
