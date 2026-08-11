import numpy as np

from veros_bgc.run_protocol import DAY, RunConfig, SegmentSpec, allocate_attempt
from veros_bgc.segment_executor import (
    SegmentedGlobalFourDegreeBGC,
    restart_schema_from_state,
)


def test_restart_contract_is_derived_from_configured_diagnostics(tmp_path):
    root = tmp_path.resolve()
    config = RunConfig(
        root=root,
        total_segments=1,
        segment_seconds=DAY,
        snapshot_seconds=DAY,
    )
    spec = SegmentSpec(
        index=0,
        start_time=0.0,
        end_time=DAY,
        input_checkpoint=None,
        paths=allocate_attempt(root, 0),
        config=config,
    )
    setup = SegmentedGlobalFourDegreeBGC(spec)
    setup.setup()

    schema = restart_schema_from_state(setup.state)

    assert set(schema) == {
        "averages",
        "energy",
        "overturning",
        "snapshot",
        "tracer_monitor",
    }
    assert set(schema["snapshot"]["attributes"]) == {
        "itt",
        "time",
        "tau",
        "taum1",
        "taup1",
    }
    expected_restart_variables = {
        name for name, metadata in setup.state.variables.items() if metadata.write_to_restart
    }
    assert set(schema["snapshot"]["datasets"]) == expected_restart_variables
    for name in expected_restart_variables:
        expected = np.asarray(getattr(setup.state, name))
        assert schema["snapshot"]["datasets"][name] == {
            "shape": list(expected.shape),
            "dtype": expected.dtype.str,
        }
