import os

from loguru import logger

from veros import distributed, runtime_state, variables as veros_variables, veros_method
from veros.decorators import do_not_disturb
from veros.diagnostics.io_tools import netcdf as nctools


@veros_method
def initialize_variable(vs, key, variable_metadata, outfile):
    """Initialize a v0.2.3 variable with h5netcdf 1.x dimension sizes."""
    dims = tuple(
        dim for dim in variable_metadata.dims if dim in outfile.dimensions
    )
    if variable_metadata.time_dependent and "Time" in outfile.dimensions:
        dims += ("Time",)

    if key in outfile.variables:
        logger.warning("Variable {} already initialized".format(key))
        return

    kwargs = {}
    if vs.enable_hdf5_gzip_compression and runtime_state.proc_num == 1:
        kwargs.update(compression="gzip", compression_opts=1)

    global_shape = [len(outfile.dimensions[dim]) or 1 for dim in dims]
    chunksize = distributed.get_local_size(
        vs, global_shape, dims, include_overlap=False
    )
    variable = outfile.create_variable(
        key,
        dims[::-1],
        variable_metadata.dtype or vs.default_float_type,
        fillvalue=veros_variables.FILL_VALUE,
        chunks=tuple(chunksize[::-1]),
        **kwargs
    )
    variable.missing_value = veros_variables.FILL_VALUE
    variable.attrs.update(
        long_name=variable_metadata.name,
        units=variable_metadata.units,
        **variable_metadata.extra_attributes
    )


@veros_method
def initialize_file(vs, outfile):
    for dim in veros_variables.BASE_DIMENSIONS:
        variable_metadata = vs.variables[dim]
        dimsize = veros_variables.get_dimensions(
            vs,
            variable_metadata.dims[::-1],
            include_ghosts=False,
            local=False,
        )[0]
        nctools.add_dimension(vs, dim, dimsize, outfile)
        initialize_variable(vs, dim, variable_metadata, outfile)
        nctools.write_variable(
            vs, dim, variable_metadata, getattr(vs, dim), outfile
        )

    outfile.dimensions["Time"] = None
    time_variable = outfile.create_variable("Time", ("Time",), float)
    time_variable.long_name = "Time"
    time_variable.units = "days"
    time_variable.time_origin = "01-JAN-1900 00:00:00"


class H5NetCDF1OutputMixin:
    """Keep v0.2.3 diagnostic behavior while normalizing dimension sizes."""

    @do_not_disturb
    @veros_method
    def initialize_output(
        self, vs, variable_metadata, var_data=None, extra_dimensions=None
    ):
        if vs.diskless_mode or (
            not self.output_frequency and not self.sampling_frequency
        ):
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
            if extra_dimensions:
                for dim_id, size in extra_dimensions.items():
                    nctools.add_dimension(vs, dim_id, size, outfile)
            for key, metadata in variable_metadata.items():
                if key not in outfile.variables:
                    initialize_variable(vs, key, metadata, outfile)
                if not metadata.time_dependent:
                    if var_data is None or key not in var_data:
                        raise ValueError(
                            "var_data argument must be given for constant variables"
                        )
                    nctools.write_variable(
                        vs, key, metadata, var_data[key], outfile
                    )
