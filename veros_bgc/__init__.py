try:
    import veros  # noqa: F401
except ImportError:
    raise RuntimeError(
        'veros-bgc needs Veros to be installed (try `pip install veros`)'
    )

from veros_bgc.variables import VARIABLES, CONDITIONAL_VARIABLES
from veros_bgc.settings import SETTINGS
from veros_bgc.core.npzd import npzd
from veros_bgc.diagnostics.npzd_monitor import NPZDMonitor

def noop(*args, **kwargs): pass


__VEROS_INTERFACE__ = dict(
    name='biogeochemistry',
    setup_entrypoint=noop,
    run_entrypoint=npzd,
    settings=SETTINGS,
    variables=VARIABLES,
    conditional_variables=CONDITIONAL_VARIABLES,
    diagnostics=[NPZDMonitor],
)
