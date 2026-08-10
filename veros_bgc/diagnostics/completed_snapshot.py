import copy
import os

from loguru import logger

from veros import time, veros_method
from veros.diagnostics.snapshot import Snapshot

from veros_bgc.core.atmospherefluxes import carbon_flux
from veros_bgc.diagnostics.netcdf_compat import H5NetCDF1OutputMixin


CARBON_DIAGNOSTIC_FIELDS = (
    "wind_speed",
    "hSWS",
    "co2star",
    "dco2star",
    "pCO2",
    "dpCO2",
    "cflux",
)


class CompletedStateSnapshot(H5NetCDF1OutputMixin, Snapshot):
    """Write the completed pre-rotation state under Veros v0.2.3."""

    name = "snapshot"

    def __init__(self, vs):
        super().__init__(vs)
        if "maskT" not in self.output_variables:
            self.output_variables.append("maskT")

    @staticmethod
    def _metadata(vs, names):
        metadata = {name: vs.variables[name] for name in names}
        if "maskT" in metadata:
            mask_metadata = copy.copy(metadata["maskT"])
            mask_metadata.get_mask = lambda state: None
            metadata["maskT"] = mask_metadata
        return metadata

    @veros_method
    def initialize(self, vs):
        var_meta = self._metadata(vs, self.output_variables)
        var_data = {name: getattr(vs, name) for name in self.output_variables}
        self.initialize_output(vs, var_meta, var_data)

    @veros_method
    def output(self, vs):
        logger.info(
            " Writing completed snapshot at {0[0]:.2f} {0[1]}",
            time.format_time(vs.time),
        )

        if not os.path.isfile(self.get_output_file_name(vs)):
            self.initialize(vs)

        original_tau = vs.tau
        carbon_state = {}
        if vs.enable_carbon:
            carbon_state = {
                name: (getattr(vs, name), getattr(vs, name).copy())
                for name in CARBON_DIAGNOSTIC_FIELDS
            }

        try:
            vs.tau = vs.taup1
            if vs.enable_carbon:
                carbon_flux(vs)

            output_names = [
                name
                for name in self.output_variables
                if vs.variables[name].time_dependent
            ]
            var_meta = self._metadata(vs, output_names)
            var_data = {name: getattr(vs, name) for name in output_names}
            self.write_output(vs, var_meta, var_data)
        finally:
            vs.tau = original_tau
            for name, (original_object, original_values) in carbon_state.items():
                original_object[...] = original_values
                setattr(vs, name, original_object)
