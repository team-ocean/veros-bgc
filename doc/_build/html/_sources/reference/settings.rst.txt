Available settings
==================

.. exec::
  from veros_bgc.settings import SETTINGS
  for key, sett in SETTINGS.items():
      print(".. _setting-{}:".format(key))
      print("")
      print(".. py:attribute:: VerosState.{} = {}".format(key, sett.default))
      print("")
      print("   {}".format(sett.description))
      print("")
