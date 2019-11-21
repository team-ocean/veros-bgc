try:
    import veros  # noqa: F401
except ImportError:
    raise RuntimeError(
        'veros_bgc needs Veros to be installed (try `pip install veros`)'
    )

try:
    from veros.plugins import PluginInterface
except ImportError:
    raise RuntimeError(
        'Could not import Veros plugin interface. '
        'Try updating Veros and veros_bgc.'
    )

from veros_bgc.variables import VARIABLES, CONDITIONAL_VARIABLES
from veros_bgc.settings import SETTINGS
from veros_bgc.npzd import npzd
from veros_bgc.npzd_monitor import NPZDMonitor

__VEROS_INTERFACE__ = PluginInterface(
    name='Biogeochemistry',
    entrypoint=npzd,
    settings=SETTINGS,
    variables=VARIABLES,
    conditional_variables=CONDITIONAL_VARIABLES,
    diagnostics=[NPZDMonitor],
)
