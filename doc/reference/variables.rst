.. _variables:

Available variables
===================

.. _flag_legend:

Attributes:
  | :fa:`clock-o`: Time-dependent
  | :fa:`download`: Included in snapshot output by default
  | :fa:`repeat`: Written to restart files by default

.. exec::
  from veros_bgc.variables import MAIN_VARIABLES, CONDITIONAL_VARIABLES
  first_condition = True
  for condition, vardict in [(None, MAIN_VARIABLES)] + list(CONDITIONAL_VARIABLES.items()):
      if not vardict:
          continue
      if condition:
          if first_condition:
              print("Conditional variables")
              print("---------------------")
              first_condition = False
          print(condition)
          print(len(condition) * "#")
      else:
          print("Main variables")
          print("--------------")
      print("")
      for key, var in vardict.items():
          flags = ""
          if var.time_dependent:
              flags += ":fa:`clock-o` "
          if var.output:
              flags += ":fa:`download` "
          if var.write_to_restart:
              flags += ":fa:`repeat` "
          print(".. py:attribute:: VerosState.{}".format(key))
          print("")
          print("  :units: {}".format(var.units))
          print("  :dimensions: {}".format(", ".join(var.dims)))
          print("  :type: :py:class:`{}`".format(var.dtype or "float"))
          print("  :attributes: {}".format(flags))
          print("")
          print("  {}".format(var.long_description))
          print("")
