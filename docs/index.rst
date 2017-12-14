Xentica: The Python CA engine
=============================

Xentica is the framework to build GPU-accelerated models for
multi-dimensional cellular automata. Given pure Python definitions, it
generates kernels in CUDA C and runs them on NVIDIA hardware.

.. warning::
   Current version is a work-in-progress, it works to some
   degree, but please do not expect something beneficial from it. As
   planned, really useful stuff would be available only starting from
   version 0.3.

User Guide
----------

If you brave enough to ignore the warning above, dive right into this
guide. Hopefully, you will manage to install Xentica on your system
and at least run some examples. Otherwise, just read Tutorial and
watch some videos to decide is it worth waiting for future versions.

.. toctree::
   :maxdepth: 2
	      
   manual/installation
   manual/tutorial
   
API Reference
-------------

.. toctree::
   :maxdepth: 2

   api/xentica.core
   api/xentica.seeds
   api/xentica.bridge
   api/xentica.utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
