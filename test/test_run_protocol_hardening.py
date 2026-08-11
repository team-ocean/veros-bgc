import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import h5netcdf
import h5py
import numpy as np
import pytest

from scripts import run_segmented_60_year
from veros_bgc.run_protocol import (
    DAY,
    ProtocolError,
    RunConfig,
    load_manifest,
    new_manifest,
    resume_run,
    run_new,
    write_manifest_atomic,
)
from veros_bgc.snapshot_merge import finalize_run, merge_snapshots


FAKE_RESTART_SCHEMA = {
    "snapshot": {
        "attributes": {
            name: {"dtype": "<i8" if name != "time" else "<f8", "shape": []}
            for name in ("itt", "time", "tau", "taum1", "taup1")
        },
        "datasets": {"state": {"dtype": "<f8", "shape": [2, 3]}},
    },
    "tracer_monitor": {
        "attributes": {"tempm1": {"dtype": "<f8", "shape": []}},
        "datasets": {},
    },
}


def _pointers(steps):
    return ((0, 1, 2), (1, 2, 0), (2, 0, 1))[steps % 3]


def _write_fake_attempt(spec):
    steps = int(spec.end_time / DAY)
    taum1, tau, taup1 = _pointers(steps)
    with h5netcdf.File(spec.paths.snapshot, "w") as outfile:
        outfile.dimensions = {"Time": None, "x": 2}
        outfile.resize_dimension("Time", 1)
        time = outfile.create_variable("Time", ("Time",), "f8", chunks=(1,))
        time.attrs["units"] = "days"
        time[:] = [steps]
        state = outfile.create_variable(
            "state",
            ("Time", "x"),
            "f8",
            chunks=(1, 2),
            compression="gzip",
            compression_opts=1,
            shuffle=True,
            fletcher32=True,
            fillvalue=-9.0,
        )
        state[:] = [[steps, steps + 0.5]]
        mask = outfile.create_variable(
            "maskT", ("x",), "i1", chunks=(2,), compression="gzip", compression_opts=1
        )
        mask[:] = [1, 0]
    with h5py.File(spec.paths.checkpoint, "w") as outfile:
        snapshot = outfile.create_group("snapshot")
        snapshot.attrs.update(
            itt=steps,
            time=spec.end_time,
            tau=tau,
            taum1=taum1,
            taup1=taup1,
        )
        snapshot.create_dataset("state", data=np.arange(6, dtype="f8").reshape(2, 3))
        monitor = outfile.create_group("tracer_monitor")
        monitor.attrs["tempm1"] = 0.0
    spec.paths.log.write_text("complete\n", encoding="utf-8")
    for path in spec.paths.diagnostics.values():
        path.write_bytes(b"diagnostic")
    return {
        "restart_schema": FAKE_RESTART_SCHEMA,
        "runtime_config": {"dt_tracer": DAY},
    }


class FakeExecutor:
    def __init__(self):
        self.calls = []

    def __call__(self, spec):
        self.calls.append(spec)
        return _write_fake_attempt(spec)


def _one_committed_segment(tmp_path, total_segments=2, segment_seconds=DAY):
    root = (tmp_path / "run").resolve()
    executor = FakeExecutor()
    run_new(
        RunConfig(
            root=root,
            total_segments=total_segments,
            segment_seconds=segment_seconds,
            snapshot_seconds=segment_seconds,
        ),
        executor,
        max_segments=1,
    )
    return root, executor.calls[0]


@pytest.mark.parametrize(
    "mutation",
    (
        ("missing_group", "tracer_monitor"),
        ("missing_dataset", "state"),
        ("dataset_shape", "state"),
        ("dataset_dtype", "state"),
        ("missing_attr", "time"),
        ("missing_attr", "itt"),
        ("missing_attr", "tau"),
        ("missing_attr", "taum1"),
        ("missing_attr", "taup1"),
        ("bad_pointer_permutation", None),
        ("bad_itt", None),
    ),
)
def test_resume_rejects_structurally_invalid_checkpoint_before_executor(tmp_path, mutation):
    root, spec = _one_committed_segment(tmp_path)
    kind, name = mutation
    with h5py.File(spec.paths.checkpoint, "r+") as checkpoint:
        if kind == "missing_group":
            del checkpoint[name]
        elif kind == "missing_dataset":
            del checkpoint["snapshot"][name]
        elif kind == "dataset_shape":
            del checkpoint["snapshot"][name]
            checkpoint["snapshot"].create_dataset(name, shape=(3, 2), dtype="f8")
        elif kind == "dataset_dtype":
            del checkpoint["snapshot"][name]
            checkpoint["snapshot"].create_dataset(name, shape=(2, 3), dtype="i8")
        elif kind == "missing_attr":
            del checkpoint["snapshot"].attrs[name]
        elif kind == "bad_pointer_permutation":
            checkpoint["snapshot"].attrs.update(taum1=0, tau=1, taup1=2)
        elif kind == "bad_itt":
            checkpoint["snapshot"].attrs["itt"] = 2

    executor = FakeExecutor()
    with pytest.raises(ProtocolError):
        resume_run(root, executor)
    assert executor.calls == []


@pytest.mark.parametrize("artifact", ("snapshot", "checkpoint"))
def test_resume_rejects_artifact_changed_after_commit_before_executor(tmp_path, artifact):
    root, spec = _one_committed_segment(tmp_path)
    path = getattr(spec.paths, artifact)
    with path.open("ab") as outfile:
        outfile.write(b"post-commit mutation")
    executor = FakeExecutor()
    with pytest.raises(ProtocolError, match="SHA-256"):
        resume_run(root, executor)
    assert executor.calls == []


def test_manifest_pins_and_reverifies_scientific_identity_before_executor(tmp_path):
    root, _ = _one_committed_segment(tmp_path)
    manifest = load_manifest(root)
    assert manifest["scientific_identity"]["veros_git_commit"]
    assert manifest["scientific_identity"]["plugin_content_sha256"]
    manifest["scientific_identity"]["runtime"]["backend"] = "changed"
    write_manifest_atomic(root, manifest)
    executor = FakeExecutor()
    with pytest.raises(ProtocolError, match="scientific identity"):
        resume_run(root, executor)
    assert executor.calls == []


@pytest.mark.parametrize("artifact", ("snapshot", "checkpoint"))
def test_resume_rejects_symlink_escape_before_executor(tmp_path, artifact):
    root, spec = _one_committed_segment(tmp_path)
    original = getattr(spec.paths, artifact)
    external = tmp_path / "external"
    external.write_bytes(original.read_bytes())
    original.unlink()
    original.symlink_to(external)
    executor = FakeExecutor()
    with pytest.raises(ProtocolError, match="symlink|canonical"):
        resume_run(root, executor)
    assert executor.calls == []


@pytest.mark.parametrize("entry_index,field", ((0, "end_time"), (30, "start_time"), (59, "end_time")))
def test_manifest_rejects_one_second_continuity_mutations(tmp_path, entry_index, field):
    root = (tmp_path / "run").resolve()
    run_new(RunConfig(root=root), FakeExecutor())
    manifest = load_manifest(root)
    manifest["committed_segments"][entry_index][field] += 1.0
    write_manifest_atomic(root, manifest)
    with pytest.raises(ProtocolError, match="contiguous"):
        resume_run(root, FakeExecutor())


@pytest.mark.parametrize("bad_entries", (None, {}, [None]))
def test_malformed_manifest_entries_raise_protocol_error(tmp_path, bad_entries):
    root = tmp_path.resolve()
    manifest = new_manifest(RunConfig(root=root))
    manifest["committed_segments"] = bad_entries
    write_manifest_atomic(root, manifest)
    with pytest.raises(ProtocolError):
        resume_run(root, FakeExecutor())


def test_required_artifacts_are_fsynced_before_segment_manifest_replace(tmp_path, monkeypatch):
    root = (tmp_path / "run").resolve()
    events = []
    real_fsync = os.fsync
    real_replace = os.replace

    def observe_fsync(fd):
        events.append(("fsync", Path(os.readlink("/proc/self/fd/{}".format(fd))).name))
        return real_fsync(fd)

    def observe_replace(source, destination):
        events.append(("replace", Path(destination).name))
        return real_replace(source, destination)

    monkeypatch.setattr(os, "fsync", observe_fsync)
    monkeypatch.setattr(os, "replace", observe_replace)
    run_new(
        RunConfig(root=root, total_segments=1, segment_seconds=DAY, snapshot_seconds=DAY),
        FakeExecutor(),
    )
    final_replace = max(i for i, event in enumerate(events) if event == ("replace", "progress.json"))
    for name in (
        "completed.snapshot.nc",
        "completed.restart.h5",
        "run.log",
        "averages.nc",
        "energy.nc",
        "overturning.nc",
    ):
        assert ("fsync", name) in events[:final_replace]


def test_nonproduction_resume_requires_smoke_before_executor(tmp_path, monkeypatch):
    root = (tmp_path / "run").resolve()
    write_manifest_atomic(
        root,
        new_manifest(
            RunConfig(root=root, total_segments=2, segment_seconds=DAY, snapshot_seconds=DAY)
        ),
    )

    class ForbiddenExecutor:
        def __call__(self, spec):
            raise AssertionError("executor must not run")

    monkeypatch.setattr(run_segmented_60_year, "VerosSegmentExecutor", ForbiddenExecutor)
    with pytest.raises(SystemExit):
        run_segmented_60_year.main(["resume", str(root)])


def test_production_resume_rejects_smoke_before_executor(tmp_path, monkeypatch):
    root = (tmp_path / "run").resolve()
    write_manifest_atomic(root, new_manifest(RunConfig(root=root)))

    class ForbiddenExecutor:
        def __call__(self, spec):
            raise AssertionError("executor must not run")

    monkeypatch.setattr(run_segmented_60_year, "VerosSegmentExecutor", ForbiddenExecutor)
    with pytest.raises(SystemExit):
        run_segmented_60_year.main(["resume", str(root), "--smoke"])


def test_direct_cli_launch_succeeds(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/run_segmented_60_year.py", "--help"],
        cwd=str(repository),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0, result.stderr
    assert "new" in result.stdout and "resume" in result.stdout


def _write_merge_snapshot(path, day, value, chunk=(1, 2)):
    with h5netcdf.File(path, "w") as outfile:
        outfile.dimensions = {"Time": None, "x": 2}
        outfile.resize_dimension("Time", 1)
        time = outfile.create_variable("Time", ("Time",), "f8", chunks=(1,))
        time.attrs["units"] = "days"
        time[:] = [day]
        state = outfile.create_variable(
            "state",
            ("Time", "x"),
            "f4",
            chunks=chunk,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            fletcher32=True,
            fillvalue=-7.0,
        )
        state[:] = [[value, value + 1]]
        mask = outfile.create_variable("maskT", ("x",), "i1", chunks=(2,), fillvalue=-1)
        mask[:] = [1, 0]


def test_merge_preserves_unlimited_dimension_and_hdf5_storage(tmp_path):
    paths = []
    for day in (1, 2):
        path = tmp_path / "{}.nc".format(day)
        _write_merge_snapshot(path, day, day)
        paths.append(path)
    merged = (tmp_path / "merged.nc").resolve()
    merge_snapshots(paths, merged, expected_records=2, interval_days=1)
    with h5netcdf.File(merged, "r") as infile:
        assert infile.dimensions["Time"].isunlimited()
        assert len(infile.dimensions["Time"]) == 2
        dataset = infile.variables["state"]._h5ds
        assert dataset.chunks == (1, 2)
        assert dataset.compression == "gzip"
        assert dataset.compression_opts == 4
        assert dataset.shuffle is True
        assert dataset.fletcher32 is True
        assert dataset.fillvalue == -7.0
        assert dataset.dtype == np.dtype("f4")


def test_merge_rejects_unlimited_or_storage_schema_mismatch(tmp_path):
    first = tmp_path / "first.nc"
    second = tmp_path / "second.nc"
    _write_merge_snapshot(first, 1, 1, chunk=(1, 2))
    _write_merge_snapshot(second, 2, 2, chunk=(1, 1))
    with pytest.raises(ProtocolError, match="schema|storage"):
        merge_snapshots([first, second], (tmp_path / "merged.nc").resolve(), 2, 1)


def test_committed_merged_snapshot_rejects_symlink_or_hash_change(tmp_path):
    root = (tmp_path / "run").resolve()
    run_new(
        RunConfig(root=root, total_segments=1, segment_seconds=DAY, snapshot_seconds=DAY),
        FakeExecutor(),
    )
    merged = finalize_run(root, lambda path: {"status": "PASS"})
    manifest = load_manifest(root)
    assert manifest["merged_sha256"] == hashlib.sha256(merged.read_bytes()).hexdigest()
    external = tmp_path / "external-merged.nc"
    external.write_bytes(merged.read_bytes())
    merged.unlink()
    merged.symlink_to(external)
    with pytest.raises(ProtocolError, match="symlink|canonical|SHA-256"):
        finalize_run(root, lambda path: {"status": "PASS"})


def test_merged_file_and_directory_are_fsynced_before_manifest_commit(tmp_path, monkeypatch):
    root = (tmp_path / "run").resolve()
    run_new(
        RunConfig(root=root, total_segments=1, segment_seconds=DAY, snapshot_seconds=DAY),
        FakeExecutor(),
    )
    events = []
    real_fsync = os.fsync
    real_replace = os.replace

    def observe_fsync(fd):
        events.append(("fsync", Path(os.readlink("/proc/self/fd/{}".format(fd))).name))
        return real_fsync(fd)

    def observe_replace(source, destination):
        events.append(("replace", Path(destination).name))
        return real_replace(source, destination)

    monkeypatch.setattr(os, "fsync", observe_fsync)
    monkeypatch.setattr(os, "replace", observe_replace)
    merged = finalize_run(root, lambda path: {"status": "PASS"})
    manifest_replace = max(i for i, event in enumerate(events) if event == ("replace", "progress.json"))
    merged_fsync = events.index(("fsync", merged.name))
    directory_fsync = events.index(("fsync", "merged"))
    assert merged_fsync < manifest_replace
    assert directory_fsync < manifest_replace
