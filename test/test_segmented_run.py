import json
import os
from pathlib import Path

import h5netcdf
import h5py
import pytest

from veros_bgc.run_protocol import (
    MANIFEST_NAME,
    ProtocolError,
    RunConfig,
    allocate_attempt,
    load_manifest,
    new_manifest,
    resume_run,
    run_new,
    write_manifest_atomic,
)
from veros_bgc.snapshot_merge import finalize_run, merge_snapshots


DAY = 86400.0
FILL_VALUE = -1e33
TEST_RESTART_SCHEMA = {
    "snapshot": {
        "attributes": {
            "itt": {"dtype": "<i8", "shape": []},
            "time": {"dtype": "<f8", "shape": []},
            "tau": {"dtype": "<i8", "shape": []},
            "taum1": {"dtype": "<i8", "shape": []},
            "taup1": {"dtype": "<i8", "shape": []},
        },
        "datasets": {},
    }
}


def _pointers(steps):
    return ((0, 1, 2), (1, 2, 0), (2, 0, 1))[steps % 3]


def write_attempt_artifacts(spec, checkpoint_time=None, snapshot_time=None):
    checkpoint_time = spec.end_time if checkpoint_time is None else checkpoint_time
    snapshot_time = spec.end_time if snapshot_time is None else snapshot_time
    steps = int(spec.end_time / DAY)
    taum1, tau, taup1 = _pointers(steps)
    with h5netcdf.File(spec.paths.snapshot, "w") as outfile:
        outfile.dimensions["Time"] = 1
        time = outfile.create_variable("Time", ("Time",), float)
        time.attrs["units"] = "days"
        time[:] = [snapshot_time / DAY]
    with h5py.File(spec.paths.checkpoint, "w") as outfile:
        snapshot = outfile.create_group("snapshot")
        snapshot.attrs.update(
            itt=steps,
            time=checkpoint_time,
            tau=tau,
            taum1=taum1,
            taup1=taup1,
        )
    spec.paths.log.write_text("completed\n", encoding="utf-8")
    for path in spec.paths.diagnostics.values():
        path.write_bytes(b"diagnostic")
    return {
        "restart_schema": TEST_RESTART_SCHEMA,
        "runtime_config": {"dt_tracer": DAY},
    }


def write_completed_snapshot(path, time_days, value, mask=(1, 0), pco2_dims=None):
    with h5netcdf.File(path, "w") as outfile:
        outfile.dimensions = {"Time": 1, "zt": 1, "yt": 1, "xt": 2}
        time = outfile.create_variable("Time", ("Time",), float)
        time.attrs["units"] = "days"
        time[:] = [time_days]
        maskt = outfile.create_variable("maskT", ("zt", "yt", "xt"), int)
        maskt[:] = [[mask]]
        for name in ("dic", "alkalinity"):
            variable = outfile.create_variable(
                name, ("Time", "zt", "yt", "xt"), float, fillvalue=FILL_VALUE
            )
            variable[:] = [[[[value, FILL_VALUE]]]]
        for name in ("pCO2", "hSWS", "cflux"):
            dims = pco2_dims if name == "pCO2" and pco2_dims else ("Time", "yt", "xt")
            variable = outfile.create_variable(name, dims, float, fillvalue=FILL_VALUE)
            surface_value = 1e-8 if name == "hSWS" else value
            if dims == ("Time", "yt", "xt"):
                variable[:] = [[[surface_value, FILL_VALUE]]]
            else:
                variable[:] = [[[[surface_value, FILL_VALUE]]]]


class RecordingExecutor:
    def __init__(self, fail_once_at=None):
        self.calls = []
        self.fail_once_at = fail_once_at

    def __call__(self, spec):
        self.calls.append(spec)
        result = write_attempt_artifacts(spec)
        if self.fail_once_at == spec.index:
            self.fail_once_at = None
            raise RuntimeError("simulated interruption")
        return result


def test_attempt_paths_are_absolute_fresh_and_confined_to_output_root(tmp_path):
    root = tmp_path.resolve()
    config = RunConfig(root=root)
    manifest = new_manifest(config)
    first = allocate_attempt(root, segment_index=0)
    second = allocate_attempt(root, segment_index=0)

    assert first.directory != second.directory
    assert first.directory.name == "segment-000-attempt-001"
    assert second.directory.name == "segment-000-attempt-002"
    for attempt in (first, second):
        paths = [attempt.directory, attempt.snapshot, attempt.checkpoint, attempt.log]
        paths.extend(attempt.diagnostics.values())
        assert all(path.is_absolute() for path in paths)
        assert all(path.is_relative_to(root) for path in paths)
        assert len(paths) == len(set(paths))
    assert manifest["config"] == {
        "output_root": str(root),
        "total_segments": 60,
        "segment_seconds": 360 * DAY,
        "snapshot_seconds": 360 * DAY,
    }


def test_run_config_requires_an_explicit_absolute_output_root(tmp_path):
    relative = Path(os.path.relpath(tmp_path, Path.cwd()))
    with pytest.raises(ValueError, match="absolute"):
        RunConfig(root=relative)


def test_manifest_replacement_is_atomic_and_keeps_prior_progress_on_failure(
    tmp_path, monkeypatch
):
    root = tmp_path.resolve()
    original = new_manifest(RunConfig(root=root, total_segments=2))
    write_manifest_atomic(root, original)
    updated = json.loads(json.dumps(original))
    updated["status"] = "complete"
    real_replace = os.replace
    observations = []

    def observe_replace(source, destination):
        observations.append(load_manifest(root)["status"])
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", observe_replace)
    write_manifest_atomic(root, updated)
    assert observations == ["running"]
    assert load_manifest(root)["status"] == "complete"
    assert not list(root.glob(".{}.*.tmp".format(MANIFEST_NAME)))

    monkeypatch.setattr(
        os,
        "replace",
        lambda source, destination: (_ for _ in ()).throw(OSError("interrupted")),
    )
    with pytest.raises(OSError, match="interrupted"):
        write_manifest_atomic(root, original)
    assert load_manifest(root)["status"] == "complete"
    assert not list(root.glob(".{}.*.tmp".format(MANIFEST_NAME)))


def test_interrupted_run_resumes_from_last_commit_without_rerunning_it(tmp_path):
    root = tmp_path.resolve()
    config = RunConfig(root=root, total_segments=3, segment_seconds=DAY, snapshot_seconds=DAY)
    interrupted = RecordingExecutor(fail_once_at=1)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_new(config, interrupted)

    after_failure = load_manifest(root)
    assert [entry["index"] for entry in after_failure["committed_segments"]] == [0]
    assert [call.index for call in interrupted.calls] == [0, 1]
    orphan = interrupted.calls[-1]
    assert orphan.paths.snapshot.exists()
    assert orphan.paths.checkpoint.exists()

    resumed = RecordingExecutor()
    final = resume_run(root, resumed)
    assert [call.index for call in resumed.calls] == [1, 2]
    assert resumed.calls[0].paths.directory.name == "segment-001-attempt-002"
    assert resumed.calls[0].input_checkpoint != interrupted.calls[0].paths.checkpoint
    assert resumed.calls[0].input_checkpoint.name == "verified-input.restart.h5"
    assert (
        resumed.calls[0].input_checkpoint.read_bytes()
        == interrupted.calls[0].paths.checkpoint.read_bytes()
    )
    assert resumed.calls[1].input_checkpoint != resumed.calls[0].paths.checkpoint
    assert resumed.calls[1].input_checkpoint.name == "verified-input.restart.h5"
    assert (
        resumed.calls[1].input_checkpoint.read_bytes()
        == resumed.calls[0].paths.checkpoint.read_bytes()
    )
    assert final["status"] == "complete"
    assert [entry["end_time"] for entry in final["committed_segments"]] == [DAY, 2 * DAY, 3 * DAY]
    assert all(
        Path(entry["snapshot"]).is_absolute() and Path(entry["checkpoint"]).is_absolute()
        for entry in final["committed_segments"]
    )


def test_attempt_is_not_committed_when_checkpoint_time_disagrees(tmp_path):
    root = tmp_path.resolve()
    config = RunConfig(root=root, total_segments=1, segment_seconds=DAY, snapshot_seconds=DAY)

    def wrong_checkpoint(spec):
        return write_attempt_artifacts(spec, checkpoint_time=spec.start_time)

    with pytest.raises(ProtocolError, match="checkpoint time"):
        run_new(config, wrong_checkpoint)
    assert load_manifest(root)["committed_segments"] == []


def test_resume_validates_committed_artifacts_before_calling_executor(tmp_path):
    root = tmp_path.resolve()
    config = RunConfig(root=root, total_segments=2, segment_seconds=DAY, snapshot_seconds=DAY)
    run_new(config, RecordingExecutor(), max_segments=1)
    manifest = load_manifest(root)
    Path(manifest["committed_segments"][0]["snapshot"]).unlink()
    executor = RecordingExecutor()
    with pytest.raises(ProtocolError, match="snapshot does not exist"):
        resume_run(root, executor)
    assert executor.calls == []


def test_merge_sorts_records_and_preserves_one_static_mask(tmp_path):
    paths = []
    for time_days in (3, 1, 2):
        path = tmp_path / "{}.nc".format(time_days)
        write_completed_snapshot(path, time_days, value=time_days)
        paths.append(path)
    output = tmp_path / "merged.nc"
    merge_snapshots(paths, output, expected_records=3, interval_days=1)
    with h5netcdf.File(output, "r") as infile:
        assert infile.variables["Time"][:].tolist() == [1.0, 2.0, 3.0]
        assert infile.variables["dic"][:, 0, 0, 0].tolist() == [1.0, 2.0, 3.0]
        assert tuple(infile.variables["maskT"].dimensions) == ("zt", "yt", "xt")
        assert infile.variables["maskT"][:].tolist() == [[[1, 0]]]


def test_merge_rejects_snapshot_time_gaps(tmp_path):
    paths = []
    for time_days in (1, 3):
        path = tmp_path / "{}.nc".format(time_days)
        write_completed_snapshot(path, time_days, value=time_days)
        paths.append(path)
    with pytest.raises(ProtocolError, match="gap"):
        merge_snapshots(paths, tmp_path / "merged.nc", expected_records=2, interval_days=1)


def test_merge_rejects_conflicting_duplicate_records(tmp_path):
    first = tmp_path / "first.nc"
    second = tmp_path / "second.nc"
    write_completed_snapshot(first, 1, value=1)
    write_completed_snapshot(second, 1, value=2)
    with pytest.raises(ProtocolError, match="conflicting duplicate"):
        merge_snapshots([first, second], tmp_path / "merged.nc", expected_records=1, interval_days=1)


@pytest.mark.parametrize("mismatch", ("schema", "mask"))
def test_merge_rejects_schema_and_static_mask_mismatches(tmp_path, mismatch):
    first = tmp_path / "first.nc"
    second = tmp_path / "second.nc"
    write_completed_snapshot(first, 1, value=1)
    if mismatch == "schema":
        write_completed_snapshot(second, 2, value=2, pco2_dims=("Time", "zt", "yt", "xt"))
    else:
        write_completed_snapshot(second, 2, value=2, mask=(0, 1))
    with pytest.raises(ProtocolError, match=mismatch):
        merge_snapshots([first, second], tmp_path / "merged.nc", expected_records=2, interval_days=1)


def test_finalize_merges_committed_snapshots_then_hands_off_to_validator(tmp_path):
    root = tmp_path.resolve()
    config = RunConfig(root=root, total_segments=2, segment_seconds=DAY, snapshot_seconds=DAY)

    def executor(spec):
        result = write_attempt_artifacts(spec)
        write_completed_snapshot(spec.paths.snapshot, spec.end_time / DAY, spec.index + 1)
        return result

    run_new(config, executor)
    calls = []

    def validator(path):
        calls.append(Path(path))
        with h5netcdf.File(path, "r") as infile:
            assert infile.variables["Time"][:].tolist() == [1.0, 2.0]
        return {"status": "PASS", "snapshot_count": 2}

    merged = finalize_run(root, validator)
    assert len(calls) == 1
    assert str(calls[0]).startswith("/proc/self/fd/")
    assert merged.is_file()
    assert merged.name == "mobi-parity.snapshot-attempt-001.nc"
    manifest = load_manifest(root)
    assert manifest["merged_snapshot"] == str(merged)
    assert manifest["validation"]["status"] == "PASS"
