from types import SimpleNamespace

from loguru import logger

from veros_bgc import segment_executor


def test_executor_installs_attempt_log_after_veros_resets_handlers(tmp_path, monkeypatch):
    class ResettingSetup:
        def __init__(self, spec):
            logger.remove()

        def setup(self):
            logger.info("setup reached")

        def run(self, show_progress_bar):
            logger.info("run reached")

    monkeypatch.setattr(segment_executor, "SegmentedGlobalFourDegreeBGC", ResettingSetup)
    spec = SimpleNamespace(paths=SimpleNamespace(log=tmp_path / "run.log"))

    segment_executor.VerosSegmentExecutor()(spec)

    text = spec.paths.log.read_text(encoding="utf-8")
    assert "setup reached" in text
    assert "run reached" in text
