import hashlib
import json
import os
from pathlib import Path
import stat

import h5netcdf
import h5py
import pytest
import scipy

from veros_bgc import run_protocol
from veros_bgc.run_protocol import DAY, ProtocolError, RunConfig
from veros_bgc.snapshot_merge import finalize_run


EXPECTED_VEROS_COMMIT = "bcbead95d56341f3ee9f30311c05a546674095e7"
RESTART_SCHEMA = {
    "snapshot": {
        "attributes": {
            "itt": {"dtype": "<i8", "shape": []},
            "time": {"dtype": "<f8", "shape": []},
            "tau": {"dtype": "<i8", "shape": []},
            "taum1": {"dtype": "<i8", "shape": []},
            "taup1": {"dtype": "<i8", "shape": []},
        },
        "datasets": {"state": {"dtype": "<f8", "shape": [2]}},
    }
}


def _pointers(steps):
    return ((0, 1, 2), (1, 2, 0), (2, 0, 1))[steps % 3]


def _write_attempt(spec):
    steps = int(spec.end_time / DAY)
    taum1, tau, taup1 = _pointers(steps)
    with h5netcdf.File(spec.paths.snapshot, "w") as outfile:
        outfile.dimensions = {"Time": None, "x": 2}
        outfile.resize_dimension("Time", 1)
        time = outfile.create_variable("Time", ("Time",), "f8", chunks=(1,))
        time.attrs["units"] = "days"
        time[:] = [steps]
        state = outfile.create_variable("state", ("Time", "x"), "f8", chunks=(1, 2))
        state[:] = [[steps, steps + 1]]
        mask = outfile.create_variable("maskT", ("x",), "i1", chunks=(2,))
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
        snapshot.create_dataset("state", data=[steps, steps + 1], dtype="f8")
    spec.paths.log.write_text("complete\n", encoding="utf-8")
    for path in spec.paths.diagnostics.values():
        path.write_bytes(b"diagnostic")
    return {
        "restart_schema": RESTART_SCHEMA,
        "runtime_config": {"dt_tracer": DAY},
    }


class Executor:
    def __init__(self, before_write=None, result_extra=None):
        self.calls = []
        self.before_write = before_write
        self.result_extra = result_extra or {}

    def __call__(self, spec):
        self.calls.append(spec)
        if self.before_write is not None:
            self.before_write(spec)
        result = _write_attempt(spec)
        result["runtime_config"].update(self.result_extra)
        return result


def _running_manifest(tmp_path, total_segments=2):
    root = (tmp_path / "run").resolve()
    executor = Executor()
    run_protocol.run_new(
        RunConfig(
            root=root,
            total_segments=total_segments,
            segment_seconds=DAY,
            snapshot_seconds=DAY,
        ),
        executor,
        max_segments=1,
    )
    return root


def test_new_run_rejects_any_veros_commit_other_than_exact_pin(tmp_path, monkeypatch):
    real_git_commit = run_protocol._git_commit

    def wrong_veros_commit(root):
        if Path(root) == run_protocol.VEROS_ROOT:
            return "0" * 40
        return real_git_commit(root)

    monkeypatch.setattr(run_protocol, "_git_commit", wrong_veros_commit)
    with pytest.raises(ProtocolError, match="pinned Veros commit"):
        run_protocol.new_manifest(RunConfig(root=(tmp_path / "run").resolve()))


def test_new_run_rejects_dirty_tracked_veros_at_pinned_head(tmp_path, monkeypatch):
    monkeypatch.setattr(run_protocol, "_git_tracked_clean", lambda root: False, raising=False)
    with pytest.raises(ProtocolError, match="clean tracked Veros"):
        run_protocol.new_manifest(RunConfig(root=(tmp_path / "run").resolve()))


def test_plugin_fingerprint_includes_validator_and_entrypoint(tmp_path):
    root = tmp_path.resolve()
    (root / "veros_bgc").mkdir()
    (root / "scripts").mkdir()
    (root / "veros_bgc" / "science.py").write_text("science = 1\n", encoding="utf-8")
    validator = root / "scripts" / "validate_60_year_run.py"
    entrypoint = root / "scripts" / "run_segmented_60_year.py"
    validator.write_text("validator = 1\n", encoding="utf-8")
    entrypoint.write_text("entrypoint = 1\n", encoding="utf-8")
    original = run_protocol._content_fingerprint(root)
    validator.write_text("validator = 2\n", encoding="utf-8")
    validator_changed = run_protocol._content_fingerprint(root)
    validator.write_text("validator = 1\n", encoding="utf-8")
    entrypoint.write_text("entrypoint = 2\n", encoding="utf-8")
    entrypoint_changed = run_protocol._content_fingerprint(root)
    assert validator_changed != original
    assert entrypoint_changed != original


@pytest.mark.parametrize("drift", ("scipy", "solver"))
def test_runtime_science_drift_fails_before_executor(tmp_path, monkeypatch, drift):
    root = (tmp_path / "run").resolve()
    manifest = run_protocol.new_manifest(
        RunConfig(root=root, total_segments=2, segment_seconds=DAY, snapshot_seconds=DAY)
    )
    run_protocol.write_manifest_atomic(root, manifest)
    if drift == "scipy":
        monkeypatch.setattr(scipy, "__version__", "changed")
    else:
        monkeypatch.setattr(
            run_protocol,
            "_resolved_solver_identity",
            lambda: "changed.module.ChangedSolver",
            raising=False,
        )
    executor = Executor()
    with pytest.raises(ProtocolError, match="scientific identity"):
        run_protocol.resume_run(root, executor)
    assert executor.calls == []


def test_scientific_identity_is_stable_and_records_resolved_solver():
    first = run_protocol.scientific_identity()
    second = run_protocol.scientific_identity()
    assert first == second
    assert first["runtime"]["scipy"] == "1.10.1"
    assert first["runtime"]["resolved_linear_solver"] == (
        "veros.core.streamfunction.solvers.scipy.SciPySolver"
    )


def test_attempts_parent_is_fsynced_before_manifest_replace(tmp_path, monkeypatch):
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
    run_protocol.run_new(
        RunConfig(root=root, total_segments=1, segment_seconds=DAY, snapshot_seconds=DAY),
        Executor(),
    )
    replace_index = max(
        index for index, event in enumerate(events) if event == ("replace", "progress.json")
    )
    attempt_index = max(
        index for index, event in enumerate(events) if event[0] == "fsync" and "attempt-" in event[1]
    )
    parent_index = max(
        index for index, event in enumerate(events) if event == ("fsync", "attempts")
    )
    assert attempt_index < parent_index < replace_index


def test_checkpoint_replaced_after_manifest_validation_never_reaches_executor(
    tmp_path, monkeypatch
):
    root = _running_manifest(tmp_path)
    manifest = run_protocol.load_manifest(root)
    checkpoint = Path(manifest["committed_segments"][0]["checkpoint"])
    external = tmp_path / "replacement.restart.h5"
    external.write_bytes(checkpoint.read_bytes())
    real_validate = run_protocol.validate_manifest
    replaced = []

    def replace_after_validation(root_arg, manifest_arg, verify_identity=True):
        result = real_validate(root_arg, manifest_arg, verify_identity=verify_identity)
        if not replaced:
            checkpoint.unlink()
            checkpoint.symlink_to(external)
            replaced.append(True)
        return result

    monkeypatch.setattr(run_protocol, "validate_manifest", replace_after_validation)
    executor = Executor()
    with pytest.raises(ProtocolError, match="symlink|stable|SHA-256"):
        run_protocol.resume_run(root, executor)
    assert executor.calls == []


def test_executor_consumes_private_checkpoint_and_source_replacement_aborts_commit(tmp_path):
    root = _running_manifest(tmp_path)
    manifest = run_protocol.load_manifest(root)
    source = Path(manifest["committed_segments"][0]["checkpoint"])
    expected_hash = manifest["committed_segments"][0]["sha256"]["checkpoint"]
    observed = {}

    def replace_source_during_execution(spec):
        observed["private_path"] = spec.input_checkpoint
        observed["private_hash"] = hashlib.sha256(spec.input_checkpoint.read_bytes()).hexdigest()
        observed["mode"] = stat.S_IMODE(spec.input_checkpoint.stat().st_mode)
        replacement = source.with_name("replacement.restart.h5")
        replacement.write_bytes(source.read_bytes())
        source.unlink()
        source.symlink_to(replacement)

    executor = Executor(before_write=replace_source_during_execution)
    with pytest.raises(ProtocolError, match="changed during|stable|symlink"):
        run_protocol.resume_run(root, executor)
    assert observed["private_path"] != source
    assert observed["private_path"].name == "verified-input.restart.h5"
    assert observed["private_hash"] == expected_hash
    assert observed["mode"] == 0o600
    assert run_protocol.load_manifest(root)["committed_segments"] == manifest["committed_segments"]


def test_concurrent_protocol_lock_contention_fails_before_executor(tmp_path):
    root = (tmp_path / "run").resolve()
    manifest = run_protocol.new_manifest(
        RunConfig(root=root, total_segments=1, segment_seconds=DAY, snapshot_seconds=DAY)
    )
    run_protocol.write_manifest_atomic(root, manifest)
    executor = Executor()
    with run_protocol.run_lock(root):
        with pytest.raises(ProtocolError, match="already locked"):
            run_protocol.resume_run(root, executor)
    assert executor.calls == []


def test_executor_runtime_contract_rejects_unvalidated_extra_fields(tmp_path):
    root = (tmp_path / "run").resolve()
    executor = Executor(result_extra={"unused": "not validated"})
    with pytest.raises(ProtocolError, match="runtime configuration"):
        run_protocol.run_new(
            RunConfig(root=root, total_segments=1, segment_seconds=DAY, snapshot_seconds=DAY),
            executor,
        )


def test_committed_merged_validation_uses_stable_descriptor(tmp_path):
    root = (tmp_path / "run").resolve()
    run_protocol.run_new(
        RunConfig(root=root, total_segments=1, segment_seconds=DAY, snapshot_seconds=DAY),
        Executor(),
    )
    merged = finalize_run(root, lambda path: {"status": "PASS"})
    expected_hash = hashlib.sha256(merged.read_bytes()).hexdigest()
    observed = {}

    def replacing_validator(stable_path):
        replacement = merged.with_name("replacement.nc")
        replacement.write_bytes(merged.read_bytes())
        merged.unlink()
        merged.symlink_to(replacement)
        observed["path"] = Path(stable_path)
        observed["hash"] = hashlib.sha256(Path(stable_path).read_bytes()).hexdigest()
        return {"status": "PASS"}

    with pytest.raises(ProtocolError, match="changed during|stable|symlink"):
        finalize_run(root, replacing_validator)
    assert str(observed["path"]).startswith("/proc/self/fd/")
    assert observed["hash"] == expected_hash
