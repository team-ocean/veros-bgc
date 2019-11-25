from collections import OrderedDict

from veros.variables import Variable, T_GRID, T_HOR, YT, TIMESTEPS


MAIN_VARIABLES = OrderedDict([])

CONDITIONAL_VARIABLES = OrderedDict([
    ('enable_npzd', OrderedDict([
        ('bottom_mask', Variable(
            'Bottom mask', T_GRID, '', 'Bottom mask', dtype='int8'
        )),
        ('phytoplankton', Variable(
            'Phytoplankton concentration', T_GRID + TIMESTEPS, 'mmol/m^3?',
            'Concentration of phytoplankton in grid box',
            output=True,
            write_to_restart=True
        )),
        ('zooplankton', Variable(
            'Zooplankton concentration', T_GRID + TIMESTEPS, 'mmol/m^3?',
            'Concentration of zooplankton in grid box',
            output=True,
            write_to_restart=True
        )),
        ('detritus', Variable(
            'Detritus concentration', T_GRID + TIMESTEPS, 'mmol/m^3?',
            'Concentration of detritus in grid box',
            output=True,
            write_to_restart=True
        )),
        ('po4', Variable(
            'Phosphate concentration', T_GRID + TIMESTEPS, 'mmol/m^3?',
            'Concentration of phosphate in grid box',
            output=True,
            write_to_restart=True
        )),
        ('swr', Variable(
            'Shortwave radiation', T_HOR, 'W/m^3?',
            'Incomming solar radiation at sea level')),
        ('rctheta', Variable(
            'Effective vertical coordinate for incoming solar radiation', YT, '1',
            'Effective vertical coordinate for incoming solar radiation')),
        ('dayfrac', Variable(
            'Fraction of day with sunlight', YT, '1',
            'Fraction of day with sunlight')),
        ('excretion_total', Variable(
            'Total excretion from zooplankton', T_GRID, 'mmol/m^3 / s',
            'Zooplankton grazing causes excretion. This stores the total excreted amount for all consumed tracers')),
    ])),
    ('enable_carbon', OrderedDict([
        ('dic', Variable(
            'Dissolved Inorganic Carbon', T_GRID + TIMESTEPS, 'mmol/m^3',
            'Concentration of inorganic carbon ions and molecule',
            output=True,
            write_to_restart=True,
        )),
        ('alkalinity', Variable(
            'Alkalinity', T_GRID + TIMESTEPS, 'mmol/m^3',
            'Combined bases and acids',
            output=True,
            write_to_restart=True
        )),
        ('atmospheric_co2', Variable(
            'Atmospheric co2 concentration', T_HOR, 'ppmv',
            'Atmospheric co2 concentration')),
        ('cflux', Variable(
            'DIC Flux', T_HOR, 'mmol/m^3/s',
            'Flux of CO2 over the ocean-atmosphere bounday',
            output=True)),
        ('wind_speed', Variable(
            'Debugging wind speed', T_HOR, 'm/s',
            'Just used for debugging. Please ignore',
            output=True)),
        ('hSWS', Variable('hSWS', T_HOR, '1',
                          '[H] in Sea water sample', output=True)),
        ('pCO2', Variable('pCO2', T_HOR, '?ppmv/atm?',
                          'Partial CO2 pressure', output=True)),
        ('dpCO2', Variable('dpCO2', T_HOR, '?ppmv/atm?',
                           'Difference in ocean CO2 pressure and atmospheric', output=True)),
        ('co2star', Variable('co2star', T_HOR, '?ppmv?',
                             'Adjusted CO2 in ocean', output=True)),
        ('dco2star', Variable('dco2star', T_HOR, '?ppmv?',
                              'Adjusted CO2 difference', output=True)),
        ('rcak', Variable('Calcite redistribution share', T_GRID, '1',
                          'Calcite is redistributed after production by dissolution varying by depth')),
    ])),
])
