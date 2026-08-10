import copy
import os

from loguru import logger

from veros import distributed, runtime_state, time, variables, veros_method
from veros.decorators import do_not_disturb
from veros.diagnostics.io_tools import netcdf as nctools
from veros.diagnostics.snapshot import Snapshot

from veros_bgc.core.atmospherefluxes import carbon_flux


CARBON_DIAGNOSTIC_FIELDS = (
    "wind_speed",
    "hSWS",
    "co2star",
    "dco2star",
    "pCO2",
    "dpCO2",
    "cflux",
)


@veros_method
def initialize_variable(vs, key, var, outfile):
    """Initialize a v0.2.3 variable with h5netcdf 1.x dimension sizes."""
    dims = tuple(dim for dim in var.dims if dim in outfile.dimensions)
    if var.time_dependent and "Time" in outfile.dimensions:
        dims += ("Time",)

    if key in outfile.variables:
        logger.warning("Variable {} already initialized".format(key))
        return

    kwargs = {}
    if vs.enable_hdf5_gzip_compression and runtime_state.proc_num == 1:
        kwargs.update(compression="gzip", compression_opts=1)

    global_shape = [len(outfile.dimensions[dim]) or 1 for dim in dims]
    chunksize = distributed.get_local_size(vs, global_shape, dims, include_overlap=False)
    variable = outfile.create_variable(
        key,
        dims[::-1],
        var.dtype or vs.default_float_type,
        fillvalue=variables.FILL_VALUE,
        chunks=tuple(chunksize[::-1]),
        **kwargs
    )
    variable.missing_value = variables.FILL_VALUE
    variable.attrs.update(
        long_name=var.name,
        units=var.units,
        **var.extra_attributes
    )


@veros_method
def initialize_file(vs, outfile):
    for dim in variables.BASE_DIMENSIONS:
        var = vs.variables[dim]
        dimsize = variables.get_dimensions(
            vs, var.dims[::-1], include_ghosts=False, local=False
        )[0]
        nctools.add_dimension(vs, dim, dimsize, outfile)
        initialize_variable(vs, dim, var, outfile)
        nctools.write_variable(vs, dim, var, getattr(vs, dim), outfile)

    outfile.dimensions["Time"] = None
    time_variable = outfile.create_variable("Time", ("Time",), float)
    time_variable.long_name = "Time"
    time_variable.units = "days"
    time_variable.time_origin = "01-JAN-1900 00:00:00"


class CompletedStateSnapshot(Snapshot):
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

    @do_not_disturb
    @veros_method
    def initialize_output(self, vs, var_meta, var_data):
        if vs.diskless_mode or (not self.output_frequency and not self.sampling_frequency):
            return

        output_path = self.get_output_file_name(vs)
        if os.path.isfile(output_path) and not vs.force_overwrite:
            raise IOError(
                'output file {} for diagnostic "{}" exists '
                '(change output path or enable force_overwrite setting)'.format(
                    output_path, self.name
                )
            )

        distributed.barrier()
        with nctools.threaded_io(vs, output_path, "w") as outfile:
            initialize_file(vs, outfile)
            for key, var in var_meta.items():
                if key not in outfile.variables:
                    initialize_variable(vs, key, var, outfile)
                if not var.time_dependent:
                    if key not in var_data:
                        raise ValueError(
                            "var_data argument must be given for constant variables"
                        )
                    nctools.write_variable(vs, key, var, var_data[key], outfile)

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
