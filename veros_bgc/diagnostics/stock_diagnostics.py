from veros.diagnostics.averages import Averages
from veros.diagnostics.energy import Energy
from veros.diagnostics.overturning import Overturning

from veros_bgc.diagnostics.netcdf_compat import H5NetCDF1OutputMixin


class CompatibleAverages(H5NetCDF1OutputMixin, Averages):
    """Averages with pinned h5netcdf 1.x output initialization."""


class CompatibleEnergy(H5NetCDF1OutputMixin, Energy):
    """Energy with pinned h5netcdf 1.x output initialization."""


class CompatibleOverturning(H5NetCDF1OutputMixin, Overturning):
    """Overturning with pinned h5netcdf 1.x output initialization."""
