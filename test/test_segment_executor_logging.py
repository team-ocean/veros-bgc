from types import SimpleNamespace

from loguru import logger

from veros_bgc import segment_executor


def test_executor_installs_attempt_log_after_veros_resets_handlers(tmp_path, monkeypatch):
    class ResettingSetup:
        def __init__(self, spec):
            logger.remove()
            self.state = SimpleNamespace(
                dt_tracer=86400.0,
                default_float_type="float64",
                enable_hdf5_gzip_compression=True,
            )

        def setup(self):
            logger.info("setup reached")

        def run(self, show_progress_bar):
            logger.info("run reached")

    monkeypatch.setattr(segment_executor, "SegmentedGlobalFourDegreeBGC", ResettingSetup)
    monkeypatch.setattr(
        segment_executor,
        "restart_schema_from_state",
        lambda state: {"configured": "contract"},
    )
    spec = SimpleNamespace(paths=SimpleNamespace(log=tmp_path / "run.log"))

    segment_executor.VerosSegmentExecutor()(spec)

    text = spec.paths.log.read_text(encoding="utf-8")
    assert "setup reached" in text
    assert "run reached" in text
