import os
from pathlib import Path
import tempfile

import h5netcdf
import numpy as np

from veros_bgc.run_protocol import (
    ProtocolError,
    _fsync_directory,
    fsync_file,
    load_manifest,
    require_artifact,
    sha256_file,
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


def _storage(variable):
    dataset = variable._h5ds
    return {
        "chunks": dataset.chunks,
        "compression": dataset.compression,
        "compression_opts": dataset.compression_opts,
        "shuffle": dataset.shuffle,
        "fletcher32": dataset.fletcher32,
        "fillvalue": np.asarray(dataset.fillvalue),
    }


def _storage_equal(left, right):
    return (
        left["chunks"] == right["chunks"]
        and left["compression"] == right["compression"]
        and left["compression_opts"] == right["compression_opts"]
        and left["shuffle"] == right["shuffle"]
        and left["fletcher32"] == right["fletcher32"]
        and _values_equal(left["fillvalue"], right["fillvalue"])
    )


def _schema(variable):
    return (
        tuple(variable.dimensions),
        np.dtype(variable.dtype).str,
        _attributes(variable),
        _storage(variable),
    )


def _compare_snapshot_schema(reference, current):
    if current["dimensions"] != reference["dimensions"]:
        raise ProtocolError("snapshot schema dimensions do not match")
    if current["schemas"].keys() != reference["schemas"].keys():
        raise ProtocolError("snapshot schema variables do not match")
    for name in reference["schemas"]:
        left_dims, left_dtype, left_attrs, left_storage = reference["schemas"][name]
        right_dims, right_dtype, right_attrs, right_storage = current["schemas"][name]
        if (
            (left_dims, left_dtype) != (right_dims, right_dtype)
            or not _attributes_equal(left_attrs, right_attrs)
            or not _storage_equal(left_storage, right_storage)
        ):
            raise ProtocolError("snapshot schema or storage mismatch for {}".format(name))
    if not _attributes_equal(
        reference["global_attributes"], current["global_attributes"]
    ):
        raise ProtocolError("snapshot schema global attributes do not match")
    for name, expected in reference["static"].items():
        if not _values_equal(current["static"][name], expected):
            label = "mask mismatch" if name == "maskT" else "static mismatch"
            raise ProtocolError("snapshot {} for {}".format(label, name))


def _read_snapshot(path, expected_records=1, reference=None):
    try:
        with h5netcdf.File(path, "r") as infile:
            if "Time" not in infile.variables:
                raise ProtocolError("snapshot schema is missing Time: {}".format(path))
            dimensions = {
                name: {"size": len(dimension), "unlimited": dimension.isunlimited()}
                for name, dimension in infile.dimensions.items()
            }
            schemas = {name: _schema(variable) for name, variable in infile.variables.items()}
            if (
                schemas["Time"][0] != ("Time",)
                or dimensions.get("Time", {}).get("size") != expected_records
            ):
                raise ProtocolError(
                    "snapshot schema must contain {} Time records: {}".format(
                        expected_records, path
                    )
                )
            time = np.asarray(infile.variables["Time"][:])
            if time.shape != (expected_records,) or not np.isfinite(time).all():
                raise ProtocolError("snapshot Time records must be finite: {}".format(path))

            static = {}
            timed = {}
            for name, variable in infile.variables.items():
                data = np.asarray(variable[:])
                if "Time" in variable.dimensions:
                    if variable.dimensions[0] != "Time" or data.shape[0] != expected_records:
                        raise ProtocolError(
                            "snapshot schema requires Time as the leading dimension: {}".format(name)
                        )
                    timed[name] = data
                else:
                    static[name] = data
            if "maskT" not in static:
                raise ProtocolError("snapshot schema requires static maskT")

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
        _compare_snapshot_schema(reference, current)
    return current


def _records_equal(left, right):
    return left.keys() == right.keys() and all(
        _values_equal(left[name], right[name]) for name in left
    )


def _copy_attributes(target, attributes):
    for name, value in attributes.items():
        if name != "_FillValue":
            target.attrs[name] = value.item() if value.ndim == 0 else value


def _create_variable(outfile, name, dims, dtype, attributes, storage):
    kwargs = {}
    if storage["chunks"] is not None:
        kwargs["chunks"] = storage["chunks"]
    if storage["compression"] is not None:
        kwargs["compression"] = storage["compression"]
        kwargs["compression_opts"] = storage["compression_opts"]
    if storage["shuffle"]:
        kwargs["shuffle"] = True
    if storage["fletcher32"]:
        kwargs["fletcher32"] = True
    if "_FillValue" in attributes:
        kwargs["fillvalue"] = storage["fillvalue"].item()
    variable = outfile.create_variable(name, dims, np.dtype(dtype), **kwargs)
    _copy_attributes(variable, attributes)
    if "_FillValue" in attributes:
        variable.attrs["_FillValue"] = attributes["_FillValue"].item()
    return variable


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
        current = _read_snapshot(path, reference=reference)
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
            for name, definition in reference["dimensions"].items():
                if definition["unlimited"]:
                    outfile.dimensions[name] = None
                else:
                    size = expected_records if name == "Time" else definition["size"]
                    outfile.dimensions[name] = size
            for name, definition in reference["dimensions"].items():
                if definition["unlimited"]:
                    size = expected_records if name == "Time" else definition["size"]
                    outfile.resize_dimension(name, size)
            _copy_attributes(outfile, reference["global_attributes"])
            for name, (dims, dtype, attributes, storage) in reference["schemas"].items():
                variable = _create_variable(outfile, name, dims, dtype, attributes, storage)
                if "Time" in dims:
                    variable[:] = np.concatenate(
                        [records[time][name] for time in ordered_times], axis=0
                    )
                else:
                    variable[:] = reference["static"][name]
        fsync_file(temporary)
        os.replace(str(temporary), str(output_path))
        fsync_file(output_path)
        _fsync_directory(output_path.parent)
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


def _validate_merged_snapshot(path, committed_snapshots, config):
    reference = _read_snapshot(committed_snapshots[0])
    merged = _read_snapshot(path, expected_records=config.total_segments)
    reference_dimensions = dict(reference["dimensions"])
    merged_dimensions = dict(merged["dimensions"])
    reference_dimensions["Time"] = dict(reference_dimensions["Time"])
    reference_dimensions["Time"]["size"] = config.total_segments
    merged_for_comparison = dict(merged)
    merged_for_comparison["dimensions"] = merged_dimensions
    reference_for_comparison = dict(reference)
    reference_for_comparison["dimensions"] = reference_dimensions
    _compare_snapshot_schema(reference_for_comparison, merged_for_comparison)
    expected_times = (
        np.arange(1, config.total_segments + 1, dtype=float)
        * config.segment_seconds
        / 86400.0
    )
    if not np.allclose(
        merged["timed"]["Time"], expected_times, rtol=0.0, atol=1e-6
    ):
        raise ProtocolError("committed merged snapshot has invalid Time records")


def finalize_run(root, validator):
    root = Path(root)
    if not root.is_absolute():
        raise ValueError("output root must be an explicit absolute path")
    manifest = load_manifest(root)
    config = validate_manifest(root, manifest)
    if manifest["status"] != "complete":
        raise ProtocolError("cannot merge an incomplete run")
    committed = [entry["snapshot"] for entry in manifest["committed_segments"]]
    if "merged_snapshot" in manifest:
        merged = require_artifact(manifest["merged_snapshot"], "merged snapshot", root)
        expected_hash = manifest.get("merged_sha256")
        if not isinstance(expected_hash, str) or sha256_file(merged) != expected_hash:
            raise ProtocolError("merged snapshot SHA-256 does not match committed evidence")
        _validate_merged_snapshot(merged, committed, config)
        validator(merged)
        return merged

    merged = _allocate_merged_path(root)
    merge_snapshots(
        committed,
        merged,
        expected_records=config.total_segments,
        interval_days=config.segment_seconds / 86400.0,
    )
    _validate_merged_snapshot(merged, committed, config)
    metrics = validator(merged)
    fsync_file(merged)
    _fsync_directory(merged.parent)
    manifest["merged_snapshot"] = str(merged)
    manifest["merged_sha256"] = sha256_file(merged)
    manifest["validation"] = metrics
    write_manifest_atomic(root, manifest)
    return merged
