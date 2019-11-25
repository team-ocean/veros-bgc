Basic usage
===========

First, install Veros-BGC:

::

   $ pip install veros-bgc

To get started with a new setup, you can use :obj:`bgc_global_4deg` as a template:

::

   $ veros copy-setup bgc_global_4deg


To enable Veros-BGC on a new setup, you will have to register it as a Veros plugin.
Add the following to your setup definition:

::

   import veros_bgc

   class MySetup(VerosSetup):
       __veros_plugins__ = (veros_bgc,)

This registers the plugin for use with Veros.
Then, you can use :doc:`the Veros-BGC settings </reference/settings>` to configure Veros-BGC.
The most important settings is :obj:`enable_npzd`, which acts as a master switch for Veros-BGC:

::

   class MySetup(VerosSetup):
       # ...

       def set_parameter(self, vs):
           vs.enable_npzd = True


.. seealso::

   All new :doc:`settings </reference/settings>` and :doc:`variables </reference/variables>` defined by Veros-BGC in their respective sections.
