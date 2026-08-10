from pathlib import Path

from veros import diagnostics as veros_diagnostics

from veros_bgc.run_protocol import DAY, RunConfig, SegmentSpec, allocate_attempt
from veros_bgc.segment_executor import SegmentedGlobalFourDegreeBGC
from scripts.run_segmented_60_year import build_parser


def test_segment_setup_uses_fresh_absolute_outputs_and_explicit_restart(
    tmp_path, monkeypatch
):
    root = tmp_path.resolve()
    config = RunConfig(
        root=root,
        total_segments=2,
        segment_seconds=DAY,
        snapshot_seconds=DAY,
    )
    previous = root / "previous.restart.h5"
    spec = SegmentSpec(
        index=1,
        start_time=DAY,
        end_time=2 * DAY,
        input_checkpoint=previous,
        paths=allocate_attempt(root, 1),
        config=config,
    )
    monkeypatch.setattr(veros_diagnostics, "read_restart", lambda state: None)

    setup = SegmentedGlobalFourDegreeBGC(spec)
    setup.setup()
    state = setup.state

    assert state.runlen == DAY
    assert state.restart_input_filename == str(previous)
    assert state.restart_output_filename == str(spec.paths.checkpoint)
    assert state.restart_frequency == 0
    assert state.force_overwrite is False
    assert state.identifier == "mobi-parity-segment-001-attempt-001"
    assert state.diagnostics["snapshot"].output_path == str(spec.paths.snapshot)
    assert state.diagnostics["snapshot"].output_frequency == DAY
    for name in ("averages", "energy", "overturning"):
        diagnostic = state.diagnostics[name]
        assert diagnostic.output_path == str(spec.paths.diagnostics[name])
        assert diagnostic.output_frequency == DAY
        assert Path(diagnostic.output_path).is_file()


def test_cli_exposes_new_resume_and_bounded_smoke_controls(tmp_path):
    parser = build_parser()
    root = str(tmp_path.resolve())
    fresh = parser.parse_args(
        [
            "new",
            root,
            "--segments",
            "2",
            "--segment-days",
            "1",
            "--snapshot-days",
            "1",
            "--max-segments",
            "1",
            "--smoke",
        ]
    )
    resumed = parser.parse_args(["resume", root, "--max-segments", "1", "--smoke"])

    assert fresh.command == "new"
    assert fresh.segments == 2
    assert fresh.segment_days == 1
    assert fresh.snapshot_days == 1
    assert fresh.max_segments == 1
    assert fresh.smoke is True
    assert resumed.command == "resume"
    assert resumed.max_segments == 1
    assert resumed.smoke is True
