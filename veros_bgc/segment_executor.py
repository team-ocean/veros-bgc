import numpy as np
from loguru import logger

from veros import veros_method

from veros_bgc.setup.bgc_global_4deg.bgc_global_four_degree import GlobalFourDegreeBGC


class SegmentedGlobalFourDegreeBGC(GlobalFourDegreeBGC):
    """The bundled setup with attempt-local I/O and restart configuration."""

    def __init__(self, spec):
        self.segment_spec = spec
        attempt = int(spec.paths.directory.name.rsplit("-", 1)[-1])
        identifier = "mobi-parity-segment-{:03d}-attempt-{:03d}".format(
            spec.index, attempt
        )
        override = {
            "identifier": identifier,
            "runlen": spec.config.segment_seconds,
            "restart_input_filename": (
                str(spec.input_checkpoint) if spec.input_checkpoint is not None else ""
            ),
            "restart_output_filename": str(spec.paths.checkpoint),
            "restart_frequency": 0,
            "force_overwrite": False,
        }
        super().__init__(override=override)

    @veros_method
    def set_diagnostics(self, vs):
        super().set_diagnostics(vs)
        vs.diagnostics["snapshot"].output_path = str(self.segment_spec.paths.snapshot)
        vs.diagnostics["snapshot"].output_frequency = self.segment_spec.config.snapshot_seconds
        for name, path in self.segment_spec.paths.diagnostics.items():
            diagnostic = vs.diagnostics[name]
            diagnostic.output_path = str(path)
            diagnostic.output_frequency = self.segment_spec.config.snapshot_seconds


def _item_schema(value):
    value = np.asarray(value)
    return {"shape": list(value.shape), "dtype": value.dtype.str}


def restart_schema_from_state(state):
    """Derive the restart contract from configured diagnostic metadata in memory."""
    schema = {}
    for diagnostic in state.diagnostics.values():
        captured = []

        def capture(vs, attributes, variable_metadata, variable_data, outfile):
            captured.append(
                {
                    "attributes": {
                        name: _item_schema(value)
                        for name, value in sorted(attributes.items())
                    },
                    "datasets": {
                        name: _item_schema(value)
                        for name, value in sorted(variable_data.items())
                    },
                }
            )

        diagnostic.write_h5_restart = capture
        try:
            diagnostic.write_restart(state, None)
        finally:
            diagnostic.__dict__.pop("write_h5_restart", None)
        if captured:
            if len(captured) != 1:
                raise RuntimeError(
                    "diagnostic {} emitted more than one restart group".format(diagnostic.name)
                )
            schema[diagnostic.name] = captured[0]
    return schema


class VerosSegmentExecutor:
    def __call__(self, spec):
        setup = SegmentedGlobalFourDegreeBGC(spec)
        sink = logger.add(str(spec.paths.log), mode="w", enqueue=False)
        try:
            setup.setup()
            schema = restart_schema_from_state(setup.state)
            runtime_config = {
                "dt_tracer": float(setup.state.dt_tracer),
                "default_float_type": setup.state.default_float_type,
                "enable_hdf5_gzip_compression": bool(
                    setup.state.enable_hdf5_gzip_compression
                ),
            }
            setup.run(show_progress_bar=False)
        finally:
            logger.remove(sink)
        return {"restart_schema": schema, "runtime_config": runtime_config}
