import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import tempfile

import h5netcdf
import h5py
import numpy as np

import veros
from veros import runtime_settings


DAY = 86400.0
MANIFEST_NAME = "progress.json"
MANIFEST_VERSION = 2
PLUGIN_ROOT = Path(__file__).resolve().parents[1]
VEROS_ROOT = Path(veros.__file__).resolve().parents[1]


class ProtocolError(RuntimeError):
    pass


def _canonical_root(root, must_exist=False):
    root = Path(root)
    if not root.is_absolute():
        raise ValueError("output root must be an explicit absolute path")
    absolute = Path(os.path.abspath(str(root)))
    try:
        resolved = root.resolve(strict=must_exist)
    except OSError as error:
        raise ProtocolError("could not resolve output root: {}".format(error)) from error
    if resolved != absolute:
        raise ProtocolError("output root must be canonical and contain no symlink components")
    return resolved


@dataclass(frozen=True)
class RunConfig:
    root: Path
    total_segments: int = 60
    segment_seconds: float = 360 * DAY
    snapshot_seconds: float = 360 * DAY

    def __post_init__(self):
        root = _canonical_root(self.root)
        object.__setattr__(self, "root", root)
        if not isinstance(self.total_segments, int) or isinstance(self.total_segments, bool):
            raise ValueError("total_segments must be an integer")
        if self.total_segments <= 0:
            raise ValueError("total_segments must be positive")
        if self.segment_seconds <= 0 or self.snapshot_seconds <= 0:
            raise ValueError("segment and snapshot durations must be positive")
        if self.snapshot_seconds != self.segment_seconds:
            raise ValueError(
                "snapshot duration must equal segment duration for one committed record per segment"
            )
        steps = self.segment_seconds / DAY
        if not float(steps).is_integer():
            raise ValueError("segment duration must contain an exact number of tracer steps")


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


def _git_commit(root):
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ProtocolError("could not identify Git commit for {}: {}".format(root, error)) from error
    return result.stdout.strip()


def _content_fingerprint(root):
    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in (root / "veros_bgc").rglob("*")
        if path.is_file() and path.suffix in (".py", ".yml", ".yaml")
    )
    if not paths:
        raise ProtocolError("no plugin science files found for identity")
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as infile:
            for chunk in iter(lambda: infile.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def scientific_identity():
    return {
        "veros_git_commit": _git_commit(VEROS_ROOT),
        "plugin_git_commit": _git_commit(PLUGIN_ROOT),
        "plugin_content_sha256": _content_fingerprint(PLUGIN_ROOT),
        "runtime": {
            "backend": runtime_settings.backend,
            "linear_solver": runtime_settings.linear_solver,
            "num_proc": list(runtime_settings.num_proc),
            "python": "{}.{}.{}".format(*sys.version_info[:3]),
            "numpy": np.__version__,
            "h5py": h5py.__version__,
            "h5netcdf": h5netcdf.__version__,
        },
        "configured_tracer_step_seconds": DAY,
    }


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
        "scientific_identity": scientific_identity(),
        "committed_segments": [],
    }


def manifest_path(root):
    return Path(root) / MANIFEST_NAME


def load_manifest(root):
    try:
        with manifest_path(root).open("r", encoding="utf-8") as infile:
            manifest = json.load(infile)
    except (OSError, ValueError) as error:
        raise ProtocolError("could not read progress manifest: {}".format(error)) from error
    if not isinstance(manifest, dict):
        raise ProtocolError("progress manifest must be a JSON object")
    return manifest


def _fsync_directory(path):
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_file(path):
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
        _fsync_directory(root)
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
    config_data = manifest.get("config")
    if not isinstance(config_data, dict):
        raise ProtocolError("invalid run configuration")
    try:
        manifest_root = Path(config_data["output_root"])
        if manifest_root != root:
            raise ProtocolError("manifest output root does not match requested output root")
        return RunConfig(
            root=root,
            total_segments=config_data["total_segments"],
            segment_seconds=config_data["segment_seconds"],
            snapshot_seconds=config_data["snapshot_seconds"],
        )
    except ProtocolError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise ProtocolError("invalid run configuration: {}".format(error)) from error


def _require_directory(path, label, root):
    if not isinstance(path, (str, os.PathLike)):
        raise ProtocolError("{} path must be a string".format(label))
    path = Path(path)
    if not path.is_absolute():
        raise ProtocolError("{} path is not absolute under output root".format(label))
    absolute = Path(os.path.abspath(str(path)))
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ProtocolError("{} does not exist: {}".format(label, error)) from error
    if resolved != absolute or not resolved.is_relative_to(root) or resolved == root:
        raise ProtocolError("{} path is not canonical or contains a symlink".format(label))
    if not resolved.is_dir():
        raise ProtocolError("{} is not a directory: {}".format(label, resolved))
    return resolved


def require_artifact(path, label, root):
    if not isinstance(path, (str, os.PathLike)):
        raise ProtocolError("{} path must be a string".format(label))
    path = Path(path)
    if not path.is_absolute():
        raise ProtocolError("{} path is not absolute under output root".format(label))
    absolute = Path(os.path.abspath(str(path)))
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ProtocolError("{} does not exist: {}".format(label, error)) from error
    if resolved != absolute or not resolved.is_relative_to(root) or resolved == root:
        raise ProtocolError("{} path is not canonical or contains a symlink".format(label))
    if not resolved.is_file():
        raise ProtocolError("{} does not exist: {}".format(label, resolved))
    return resolved


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_sha256(path, expected, label):
    if not isinstance(expected, str) or len(expected) != 64:
        raise ProtocolError("invalid committed SHA-256 for {}".format(label))
    actual = sha256_file(path)
    if actual != expected:
        raise ProtocolError("{} SHA-256 does not match committed evidence".format(label))


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


def _schema_scalar(value):
    array = np.asarray(value)
    return {"shape": list(array.shape), "dtype": array.dtype.str}


def _validate_restart_schema(schema):
    if not isinstance(schema, dict) or "snapshot" not in schema:
        raise ProtocolError("executor returned no configured snapshot restart schema")
    normalized = {}
    for group_name, group_schema in sorted(schema.items()):
        if not isinstance(group_name, str) or not isinstance(group_schema, dict):
            raise ProtocolError("invalid configured restart schema")
        attributes = group_schema.get("attributes")
        datasets = group_schema.get("datasets")
        if not isinstance(attributes, dict) or not isinstance(datasets, dict):
            raise ProtocolError("invalid configured restart group schema")
        normalized[group_name] = {
            "attributes": {name: dict(value) for name, value in sorted(attributes.items())},
            "datasets": {name: dict(value) for name, value in sorted(datasets.items())},
        }
        for collection in normalized[group_name].values():
            for name, item in collection.items():
                if not isinstance(name, str) or set(item) != {"shape", "dtype"}:
                    raise ProtocolError("invalid configured restart item schema")
                if not isinstance(item["shape"], list) or not isinstance(item["dtype"], str):
                    raise ProtocolError("invalid configured restart item schema")
    required = {"itt", "time", "tau", "taum1", "taup1"}
    if not required.issubset(normalized["snapshot"]["attributes"]):
        raise ProtocolError("configured restart schema lacks required snapshot attributes")
    return normalized


def _expected_step_state(expected_time):
    steps_float = expected_time / DAY
    if not float(steps_float).is_integer():
        raise ProtocolError("expected time is not an exact tracer-step boundary")
    steps = int(steps_float)
    pointers = ((0, 1, 2), (1, 2, 0), (2, 0, 1))[steps % 3]
    return steps, pointers


def _validate_checkpoint(path, schema, expected_time):
    schema = _validate_restart_schema(schema)
    try:
        with h5py.File(path, "r") as infile:
            if set(infile.keys()) != set(schema):
                raise ProtocolError("checkpoint restart groups do not match configured schema")
            for group_name, expected_group in schema.items():
                group = infile[group_name]
                if not isinstance(group, h5py.Group):
                    raise ProtocolError("checkpoint restart group is not a group")
                if set(group.attrs) != set(expected_group["attributes"]):
                    raise ProtocolError(
                        "checkpoint attributes do not match configured schema for {}".format(group_name)
                    )
                if set(group.keys()) != set(expected_group["datasets"]):
                    raise ProtocolError(
                        "checkpoint datasets do not match configured schema for {}".format(group_name)
                    )
                for name, expected in expected_group["attributes"].items():
                    actual = _schema_scalar(group.attrs[name])
                    if actual != expected:
                        raise ProtocolError("checkpoint attribute schema mismatch for {}/{}".format(group_name, name))
                for name, expected in expected_group["datasets"].items():
                    dataset = group[name]
                    actual = {"shape": list(dataset.shape), "dtype": dataset.dtype.str}
                    if actual != expected:
                        raise ProtocolError("checkpoint dataset schema mismatch for {}/{}".format(group_name, name))

            snapshot = infile["snapshot"]
            time_value = float(snapshot.attrs["time"])
            if not np.isfinite(time_value) or not np.isclose(
                time_value, expected_time, rtol=0.0, atol=1e-6
            ):
                raise ProtocolError("checkpoint time does not match expected completed time")
            expected_itt, (expected_taum1, expected_tau, expected_taup1) = _expected_step_state(
                expected_time
            )
            if int(snapshot.attrs["itt"]) != expected_itt:
                raise ProtocolError("checkpoint itt does not match elapsed tracer steps")
            actual_pointers = tuple(
                int(snapshot.attrs[name]) for name in ("taum1", "tau", "taup1")
            )
            if actual_pointers != (expected_taum1, expected_tau, expected_taup1):
                raise ProtocolError("checkpoint time-level pointer permutation is invalid")
    except OSError as error:
        raise ProtocolError("could not read checkpoint {}: {}".format(path, error)) from error


def _validate_artifact_times(snapshot, checkpoint, schema, expected_time):
    snapshot_time = _read_snapshot_time(snapshot)
    if not np.isclose(snapshot_time, expected_time, rtol=0.0, atol=1e-6):
        raise ProtocolError(
            "snapshot time {} does not match expected {}".format(snapshot_time, expected_time)
        )
    _validate_checkpoint(checkpoint, schema, expected_time)


def _entry_artifacts(entry, root):
    attempt_directory = _require_directory(entry.get("attempt_directory"), "attempt directory", root)
    snapshot = require_artifact(entry.get("snapshot"), "snapshot", root)
    checkpoint = require_artifact(entry.get("checkpoint"), "checkpoint", root)
    log = require_artifact(entry.get("log"), "log", root)
    diagnostics = entry.get("diagnostics")
    if not isinstance(diagnostics, dict) or set(diagnostics) != {
        "averages",
        "energy",
        "overturning",
    }:
        raise ProtocolError("invalid committed diagnostic artifact mapping")
    artifacts = {"snapshot": snapshot, "checkpoint": checkpoint, "log": log}
    for name, path in diagnostics.items():
        artifacts[name] = require_artifact(path, name, root)
    if any(path.parent != attempt_directory for path in artifacts.values()):
        raise ProtocolError("committed artifact is outside its attempt directory")
    return attempt_directory, artifacts


def validate_manifest(root, manifest, verify_identity=True):
    try:
        root = _canonical_root(root, must_exist=True)
        if not isinstance(manifest, dict):
            raise ProtocolError("progress manifest must be an object")
        config = _config_from_manifest(root, manifest)
        if verify_identity and manifest.get("scientific_identity") != scientific_identity():
            raise ProtocolError("scientific identity differs from committed run")
        if manifest.get("status") not in ("running", "complete"):
            raise ProtocolError("invalid progress status")
        entries = manifest.get("committed_segments")
        if not isinstance(entries, list) or len(entries) > config.total_segments:
            raise ProtocolError("invalid committed segment list")
        schema = manifest.get("restart_schema")
        if entries:
            schema = _validate_restart_schema(schema)
        elif schema is not None:
            schema = _validate_restart_schema(schema)

        for expected_index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ProtocolError("committed segment entry must be an object")
            expected_start = expected_index * config.segment_seconds
            expected_end = expected_start + config.segment_seconds
            if entry.get("index") != expected_index:
                raise ProtocolError("committed segment indices are not contiguous")
            if not np.isclose(
                entry.get("start_time", np.nan), expected_start, rtol=0.0, atol=1e-6
            ):
                raise ProtocolError("committed segment start times are not contiguous")
            if not np.isclose(
                entry.get("end_time", np.nan), expected_end, rtol=0.0, atol=1e-6
            ):
                raise ProtocolError("committed segment end times are not contiguous")
            _, artifacts = _entry_artifacts(entry, root)
            hashes = entry.get("sha256")
            if not isinstance(hashes, dict) or set(hashes) != set(artifacts):
                raise ProtocolError("invalid committed artifact SHA-256 mapping")
            for name, path in artifacts.items():
                _check_sha256(path, hashes[name], name)
            _validate_artifact_times(
                artifacts["snapshot"], artifacts["checkpoint"], schema, expected_end
            )

        expected_status = "complete" if len(entries) == config.total_segments else "running"
        if manifest["status"] != expected_status:
            raise ProtocolError("progress status disagrees with committed segment count")
        return config
    except ProtocolError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ProtocolError("invalid progress manifest: {}".format(error)) from error


def _entry_from_spec(spec, hashes):
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
        "sha256": hashes,
    }


def _execution_contract(result):
    if not isinstance(result, dict):
        raise ProtocolError("segment executor returned no configured restart contract")
    schema = _validate_restart_schema(result.get("restart_schema"))
    runtime_config = result.get("runtime_config")
    if not isinstance(runtime_config, dict) or runtime_config.get("dt_tracer") != DAY:
        raise ProtocolError("configured tracer timestep differs from protocol identity")
    return schema


def _sync_attempt(spec, root):
    entry = {
        "attempt_directory": str(spec.paths.directory),
        "snapshot": str(spec.paths.snapshot),
        "checkpoint": str(spec.paths.checkpoint),
        "log": str(spec.paths.log),
        "diagnostics": {name: str(path) for name, path in spec.paths.diagnostics.items()},
    }
    directory, artifacts = _entry_artifacts(entry, root)
    for path in artifacts.values():
        fsync_file(path)
    _fsync_directory(directory)
    return artifacts, {name: sha256_file(path) for name, path in artifacts.items()}


def run_new(config, executor, max_segments=None):
    root = config.root
    if manifest_path(root).exists():
        raise ProtocolError("progress manifest already exists; use resume")
    if root.exists() and any(root.iterdir()):
        raise ProtocolError("new run output root must be empty")
    write_manifest_atomic(root, new_manifest(config))
    return resume_run(root, executor, max_segments=max_segments)


def resume_run(root, executor, max_segments=None):
    root = _canonical_root(root, must_exist=True)
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
        schema = _execution_contract(executor(spec))
        if "restart_schema" in manifest and schema != manifest["restart_schema"]:
            raise ProtocolError("configured restart schema changed between segments")
        manifest.setdefault("restart_schema", schema)
        artifacts, hashes = _sync_attempt(spec, root)
        _validate_artifact_times(
            artifacts["snapshot"], artifacts["checkpoint"], schema, end_time
        )
        manifest["committed_segments"].append(_entry_from_spec(spec, hashes))
        if len(manifest["committed_segments"]) == config.total_segments:
            manifest["status"] = "complete"
        write_manifest_atomic(root, manifest)

    return manifest
