import os
from pathlib import Path
import tempfile

import h5netcdf
import numpy as np

from veros_bgc.run_protocol import (
    ProtocolError,
    load_manifest,
    validate_manifest,
    write_manifest_atomic,
)


def _attributes(variable):
    return {name: np.asarray(value) for name, value in variable.attrs.items()}


def _values_equal(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape or left.dtype.kind != right.dtype.kind:
        return False
    if left.dtype.kind in "fc":
        return np.array_equal(left, right, equal_nan=True)
    return np.array_equal(left, right)


def _attributes_equal(left, right):
    return left.keys() == right.keys() and all(
        _values_equal(left[name], right[name]) for name in left
    )


def _schema(variable):
    return tuple(variable.dimensions), np.dtype(variable.dtype).str, _attributes(variable)


def _read_segment(path, reference=None):
    try:
        with h5netcdf.File(path, "r") as infile:
            if "Time" not in infile.variables:
                raise ProtocolError("snapshot schema is missing Time: {}".format(path))
            dimensions = {
                name: len(dimension) for name, dimension in infile.dimensions.items()
            }
            schemas = {name: _schema(variable) for name, variable in infile.variables.items()}
            if schemas["Time"][0] != ("Time",) or dimensions.get("Time") != 1:
                raise ProtocolError("snapshot schema must contain one Time record: {}".format(path))
            time = np.asarray(infile.variables["Time"][:])
            if time.shape != (1,) or not np.isfinite(time).all():
                raise ProtocolError("snapshot Time must be one finite record: {}".format(path))

            static = {}
            timed = {}
            for name, variable in infile.variables.items():
                data = np.asarray(variable[:])
                if "Time" in variable.dimensions:
                    if variable.dimensions[0] != "Time" or data.shape[0] != 1:
                        raise ProtocolError(
                            "snapshot schema requires Time as the leading dimension: {}".format(name)
                        )
                    timed[name] = data
                else:
                    static[name] = data

            current = {
                "dimensions": dimensions,
                "schemas": schemas,
                "static": static,
                "timed": timed,
                "global_attributes": {
                    name: np.asarray(value) for name, value in infile.attrs.items()
                },
            }
    except OSError as error:
        raise ProtocolError("could not read snapshot {}: {}".format(path, error)) from error

    if reference is not None:
        reference_dimensions = dict(reference["dimensions"])
        current_dimensions = dict(current["dimensions"])
        reference_dimensions["Time"] = 1
        current_dimensions["Time"] = 1
        if current_dimensions != reference_dimensions:
            raise ProtocolError("snapshot schema dimensions do not match")
        if current["schemas"].keys() != reference["schemas"].keys():
            raise ProtocolError("snapshot schema variables do not match")
        for name in reference["schemas"]:
            left_dims, left_dtype, left_attrs = reference["schemas"][name]
            right_dims, right_dtype, right_attrs = current["schemas"][name]
            if (left_dims, left_dtype) != (right_dims, right_dtype) or not _attributes_equal(
                left_attrs, right_attrs
            ):
                raise ProtocolError("snapshot schema mismatch for {}".format(name))
        if not _attributes_equal(
            reference["global_attributes"], current["global_attributes"]
        ):
            raise ProtocolError("snapshot schema global attributes do not match")
        for name, expected in reference["static"].items():
            if not _values_equal(current["static"][name], expected):
                label = "mask mismatch" if name == "maskT" else "static mismatch"
                raise ProtocolError("snapshot {} for {}".format(label, name))
    return current


def _records_equal(left, right):
    return all(_values_equal(left[name], right[name]) for name in left)


def _copy_attributes(target, attributes):
    for name, value in attributes.items():
        if name != "_FillValue":
            target.attrs[name] = value.item() if value.ndim == 0 else value


def merge_snapshots(paths, output_path, expected_records=60, interval_days=360.0):
    if expected_records <= 0 or interval_days <= 0:
        raise ValueError("expected record count and interval must be positive")
    paths = [Path(path) for path in paths]
    if not paths:
        raise ProtocolError("no committed snapshots to merge")
    output_path = Path(output_path)
    if not output_path.is_absolute():
        raise ValueError("merged output path must be absolute")
    if output_path.exists():
        raise ProtocolError("merged output already exists: {}".format(output_path))

    reference = None
    records = {}
    for path in paths:
        current = _read_segment(path, reference=reference)
        if reference is None:
            reference = current
        time = float(current["timed"]["Time"][0])
        duplicate = next(
            (known for known in records if np.isclose(known, time, rtol=0.0, atol=1e-6)),
            None,
        )
        if duplicate is not None:
            if not _records_equal(records[duplicate], current["timed"]):
                raise ProtocolError("conflicting duplicate snapshot record at {} days".format(time))
            continue
        records[time] = current["timed"]

    ordered_times = np.asarray(sorted(records))
    expected_times = np.arange(1, expected_records + 1, dtype=float) * interval_days
    if ordered_times.shape != expected_times.shape or not np.allclose(
        ordered_times, expected_times, rtol=0.0, atol=1e-6
    ):
        raise ProtocolError(
            "snapshot time gap: found {}, expected {}".format(
                ordered_times.tolist(), expected_times.tolist()
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".{}-".format(output_path.name),
            suffix=".tmp",
            dir=str(output_path.parent),
            delete=False,
        ) as outfile:
            temporary = Path(outfile.name)
        temporary.unlink()
        with h5netcdf.File(temporary, "w") as outfile:
            for name, size in reference["dimensions"].items():
                outfile.dimensions[name] = expected_records if name == "Time" else size
            _copy_attributes(outfile, reference["global_attributes"])
            for name, (dims, dtype, attributes) in reference["schemas"].items():
                fill_value = attributes.get("_FillValue")
                if fill_value is not None:
                    fill_value = fill_value.item()
                variable = outfile.create_variable(name, dims, np.dtype(dtype), fillvalue=fill_value)
                _copy_attributes(variable, attributes)
                if "Time" in dims:
                    variable[:] = np.concatenate(
                        [records[time][name] for time in ordered_times], axis=0
                    )
                else:
                    variable[:] = reference["static"][name]
        os.replace(str(temporary), str(output_path))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return output_path


def _allocate_merged_path(root):
    merged_root = root / "merged"
    merged_root.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while True:
        path = merged_root / "mobi-parity.snapshot-attempt-{:03d}.nc".format(attempt)
        if not path.exists():
            return path
        attempt += 1


def finalize_run(root, validator):
    root = Path(root)
    if not root.is_absolute():
        raise ValueError("output root must be an explicit absolute path")
    manifest = load_manifest(root)
    config = validate_manifest(root, manifest)
    if manifest["status"] != "complete":
        raise ProtocolError("cannot merge an incomplete run")
    if "merged_snapshot" in manifest:
        merged = Path(manifest["merged_snapshot"])
        if not merged.is_file():
            raise ProtocolError("committed merged snapshot does not exist")
        validator(merged)
        return merged

    merged = _allocate_merged_path(root)
    merge_snapshots(
        [entry["snapshot"] for entry in manifest["committed_segments"]],
        merged,
        expected_records=config.total_segments,
        interval_days=config.segment_seconds / 86400.0,
    )
    metrics = validator(merged)
    manifest["merged_snapshot"] = str(merged)
    manifest["validation"] = metrics
    write_manifest_atomic(root, manifest)
    return merged
