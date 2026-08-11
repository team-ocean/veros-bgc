#!/usr/bin/env python

import argparse
from pathlib import Path

import h5netcdf

try:
    from scripts.validate_60_year_run import validate_snapshot
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from validate_60_year_run import validate_snapshot

from veros_bgc.run_protocol import (
    DAY,
    RunConfig,
    load_manifest,
    resume_run,
    run_new,
    validate_manifest,
)
from veros_bgc.segment_executor import VerosSegmentExecutor
from veros_bgc.snapshot_merge import finalize_run


def _positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _positive_float(value):
    value = float(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the bundled four-degree MOBI parity setup in committed segments."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fresh = subparsers.add_parser("new", help="start a new immutable segmented run")
    fresh.add_argument("output_root", type=Path)
    fresh.add_argument("--segments", type=_positive_int, default=60)
    fresh.add_argument("--segment-days", type=_positive_float, default=360.0)
    fresh.add_argument("--snapshot-days", type=_positive_float, default=360.0)
    fresh.add_argument("--max-segments", type=_positive_int)
    fresh.add_argument(
        "--smoke",
        action="store_true",
        help="allow a bounded non-60-year run and use structural merge validation",
    )

    resume = subparsers.add_parser("resume", help="resume from the last committed segment")
    resume.add_argument("output_root", type=Path)
    resume.add_argument("--max-segments", type=_positive_int)
    resume.add_argument(
        "--smoke",
        action="store_true",
        help="use structural merge validation for a bounded smoke configuration",
    )
    return parser


def _smoke_validator(path):
    with h5netcdf.File(path, "r") as infile:
        count = len(infile.dimensions["Time"])
    return {"status": "SMOKE_PASS", "snapshot_count": count}


def _is_production_config(config):
    return (
        config["total_segments"] == 60
        and config["segment_seconds"] == 360 * DAY
        and config["snapshot_seconds"] == 360 * DAY
    )


def _config_mapping(config):
    return {
        "total_segments": config.total_segments,
        "segment_seconds": config.segment_seconds,
        "snapshot_seconds": config.snapshot_seconds,
    }


def _check_mode(parser, config, smoke):
    production = _is_production_config(config)
    if production and smoke:
        parser.error("production configuration rejects --smoke and always uses validate_snapshot")
    if not production and not smoke:
        parser.error("non-60-year configurations require --smoke")


def _finish_if_complete(root, manifest, smoke):
    if manifest["status"] != "complete":
        return None
    production = _is_production_config(manifest["config"])
    validator = validate_snapshot if production else _smoke_validator
    return finalize_run(root, validator)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    root = args.output_root
    if not root.is_absolute():
        parser.error("output_root must be an explicit absolute path")

    if args.command == "new":
        config = RunConfig(
            root=root,
            total_segments=args.segments,
            segment_seconds=args.segment_days * DAY,
            snapshot_seconds=args.snapshot_days * DAY,
        )
        _check_mode(parser, _config_mapping(config), args.smoke)
        executor = VerosSegmentExecutor()
        manifest = run_new(config, executor, max_segments=args.max_segments)
    else:
        manifest = load_manifest(root)
        validate_manifest(root, manifest)
        _check_mode(parser, manifest["config"], args.smoke)
        executor = VerosSegmentExecutor()
        manifest = resume_run(root, executor, max_segments=args.max_segments)
    merged = _finish_if_complete(root, manifest, args.smoke)
    if merged is None:
        print(
            "Committed {}/{} segments; resume with the same output root.".format(
                len(manifest["committed_segments"]), manifest["config"]["total_segments"]
            )
        )
    else:
        print("Merged snapshot: {}".format(merged))


if __name__ == "__main__":
    main()
