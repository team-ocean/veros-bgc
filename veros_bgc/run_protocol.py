import json
import os
from dataclasses import dataclass
from pathlib import Path
import tempfile

import h5netcdf
import h5py
import numpy as np


DAY = 86400.0
MANIFEST_NAME = "progress.json"
MANIFEST_VERSION = 1


class ProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunConfig:
    root: Path
    total_segments: int = 60
    segment_seconds: float = 360 * DAY
    snapshot_seconds: float = 360 * DAY

    def __post_init__(self):
        if not self.root.is_absolute():
            raise ValueError("output root must be an explicit absolute path")
        if self.total_segments <= 0:
            raise ValueError("total_segments must be positive")
        if self.segment_seconds <= 0 or self.snapshot_seconds <= 0:
            raise ValueError("segment and snapshot durations must be positive")
        if self.snapshot_seconds != self.segment_seconds:
            raise ValueError(
                "snapshot duration must equal segment duration for one committed record per segment"
            )


@dataclass(frozen=True)
class AttemptPaths:
    directory: Path
    snapshot: Path
    checkpoint: Path
    log: Path
    diagnostics: dict


@dataclass(frozen=True)
class SegmentSpec:
    index: int
    start_time: float
    end_time: float
    input_checkpoint: object
    paths: AttemptPaths
    config: RunConfig


def new_manifest(config):
    return {
        "version": MANIFEST_VERSION,
        "status": "running",
        "config": {
            "output_root": str(config.root),
            "total_segments": config.total_segments,
            "segment_seconds": config.segment_seconds,
            "snapshot_seconds": config.snapshot_seconds,
        },
        "committed_segments": [],
    }


def manifest_path(root):
    return Path(root) / MANIFEST_NAME


def load_manifest(root):
    try:
        with manifest_path(root).open("r", encoding="utf-8") as infile:
            return json.load(infile)
    except (OSError, ValueError) as error:
        raise ProtocolError("could not read progress manifest: {}".format(error)) from error


def write_manifest_atomic(root, manifest):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    destination = manifest_path(root)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=".{}.".format(MANIFEST_NAME),
            suffix=".tmp",
            dir=str(root),
            delete=False,
        ) as outfile:
            temporary = Path(outfile.name)
            json.dump(manifest, outfile, indent=2, sort_keys=True)
            outfile.write("\n")
            outfile.flush()
            os.fsync(outfile.fileno())
        os.replace(str(temporary), str(destination))
        directory_fd = os.open(str(root), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def allocate_attempt(root, segment_index):
    root = Path(root)
    attempts_root = root / "attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    prefix = "segment-{:03d}-attempt-".format(segment_index)
    attempt_number = 1
    while True:
        directory = attempts_root / "{}{:03d}".format(prefix, attempt_number)
        try:
            directory.mkdir()
        except FileExistsError:
            attempt_number += 1
            continue
        break

    return AttemptPaths(
        directory=directory,
        snapshot=directory / "completed.snapshot.nc",
        checkpoint=directory / "completed.restart.h5",
        log=directory / "run.log",
        diagnostics={
            name: directory / "{}.nc".format(name)
            for name in ("averages", "energy", "overturning")
        },
    )


def _config_from_manifest(root, manifest):
    if manifest.get("version") != MANIFEST_VERSION:
        raise ProtocolError("unsupported progress manifest version")
    config_data = manifest.get("config", {})
    manifest_root = Path(config_data.get("output_root", ""))
    if manifest_root != root:
        raise ProtocolError("manifest output root does not match requested output root")
    try:
        return RunConfig(
            root=root,
            total_segments=config_data["total_segments"],
            segment_seconds=config_data["segment_seconds"],
            snapshot_seconds=config_data["snapshot_seconds"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ProtocolError("invalid run configuration: {}".format(error)) from error


def _require_artifact(path, label, root):
    path = Path(path)
    if not path.is_absolute() or not path.is_relative_to(root):
        raise ProtocolError("{} path is not absolute under output root".format(label))
    if not path.is_file():
        raise ProtocolError("{} does not exist: {}".format(label, path))
    return path


def _read_snapshot_time(path):
    try:
        with h5netcdf.File(path, "r") as infile:
            if "Time" not in infile.variables:
                raise ProtocolError("snapshot has no Time variable: {}".format(path))
            variable = infile.variables["Time"]
            if tuple(variable.dimensions) != ("Time",) or variable.shape != (1,):
                raise ProtocolError("snapshot must contain exactly one Time record: {}".format(path))
            if variable.attrs.get("units") != "days":
                raise ProtocolError("snapshot Time units must be days: {}".format(path))
            value = np.asarray(variable[:])
            if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
                raise ProtocolError("snapshot Time must be finite and numeric: {}".format(path))
            return float(value[0]) * DAY
    except OSError as error:
        raise ProtocolError("could not read snapshot {}: {}".format(path, error)) from error


def _read_checkpoint_time(path):
    try:
        with h5py.File(path, "r") as infile:
            if "snapshot" not in infile or "time" not in infile["snapshot"].attrs:
                raise ProtocolError("checkpoint has no snapshot time: {}".format(path))
            value = infile["snapshot"].attrs["time"]
            if not np.isscalar(value) or not np.isfinite(value):
                raise ProtocolError("checkpoint time must be finite: {}".format(path))
            return float(value)
    except OSError as error:
        raise ProtocolError("could not read checkpoint {}: {}".format(path, error)) from error


def _validate_artifact_times(snapshot, checkpoint, expected_time):
    snapshot_time = _read_snapshot_time(snapshot)
    if not np.isclose(snapshot_time, expected_time, rtol=0.0, atol=1e-6):
        raise ProtocolError(
            "snapshot time {} does not match expected {}".format(snapshot_time, expected_time)
        )
    checkpoint_time = _read_checkpoint_time(checkpoint)
    if not np.isclose(checkpoint_time, expected_time, rtol=0.0, atol=1e-6):
        raise ProtocolError(
            "checkpoint time {} does not match expected {}".format(checkpoint_time, expected_time)
        )


def validate_manifest(root, manifest):
    root = Path(root)
    config = _config_from_manifest(root, manifest)
    if manifest.get("status") not in ("running", "complete"):
        raise ProtocolError("invalid progress status")
    entries = manifest.get("committed_segments")
    if not isinstance(entries, list) or len(entries) > config.total_segments:
        raise ProtocolError("invalid committed segment list")

    for expected_index, entry in enumerate(entries):
        expected_start = expected_index * config.segment_seconds
        expected_end = expected_start + config.segment_seconds
        if entry.get("index") != expected_index:
            raise ProtocolError("committed segment indices are not contiguous")
        if not np.isclose(entry.get("start_time", np.nan), expected_start):
            raise ProtocolError("committed segment start times are not contiguous")
        if not np.isclose(entry.get("end_time", np.nan), expected_end):
            raise ProtocolError("committed segment end times are not contiguous")
        snapshot = _require_artifact(entry.get("snapshot", ""), "snapshot", root)
        checkpoint = _require_artifact(entry.get("checkpoint", ""), "checkpoint", root)
        _validate_artifact_times(snapshot, checkpoint, expected_end)

    expected_status = "complete" if len(entries) == config.total_segments else "running"
    if manifest["status"] != expected_status:
        raise ProtocolError("progress status disagrees with committed segment count")
    return config


def _entry_from_spec(spec):
    return {
        "index": spec.index,
        "start_time": spec.start_time,
        "end_time": spec.end_time,
        "attempt_directory": str(spec.paths.directory),
        "snapshot": str(spec.paths.snapshot),
        "checkpoint": str(spec.paths.checkpoint),
        "log": str(spec.paths.log),
        "diagnostics": {
            name: str(path) for name, path in sorted(spec.paths.diagnostics.items())
        },
    }


def run_new(config, executor, max_segments=None):
    root = config.root
    if manifest_path(root).exists():
        raise ProtocolError("progress manifest already exists; use resume")
    if root.exists() and any(root.iterdir()):
        raise ProtocolError("new run output root must be empty")
    write_manifest_atomic(root, new_manifest(config))
    return resume_run(root, executor, max_segments=max_segments)


def resume_run(root, executor, max_segments=None):
    root = Path(root)
    if not root.is_absolute():
        raise ValueError("output root must be an explicit absolute path")
    if max_segments is not None and max_segments < 0:
        raise ValueError("max_segments must be nonnegative")

    manifest = load_manifest(root)
    config = validate_manifest(root, manifest)
    if manifest["status"] == "complete" or max_segments == 0:
        return manifest

    first_index = len(manifest["committed_segments"])
    stop_index = config.total_segments
    if max_segments is not None:
        stop_index = min(stop_index, first_index + max_segments)

    for index in range(first_index, stop_index):
        start_time = index * config.segment_seconds
        end_time = start_time + config.segment_seconds
        previous = manifest["committed_segments"][-1] if index else None
        input_checkpoint = Path(previous["checkpoint"]) if previous else None
        spec = SegmentSpec(
            index=index,
            start_time=start_time,
            end_time=end_time,
            input_checkpoint=input_checkpoint,
            paths=allocate_attempt(root, index),
            config=config,
        )
        executor(spec)
        snapshot = _require_artifact(spec.paths.snapshot, "snapshot", root)
        checkpoint = _require_artifact(spec.paths.checkpoint, "checkpoint", root)
        _validate_artifact_times(snapshot, checkpoint, end_time)
        manifest["committed_segments"].append(_entry_from_spec(spec))
        if len(manifest["committed_segments"]) == config.total_segments:
            manifest["status"] = "complete"
        write_manifest_atomic(root, manifest)

    return manifest
