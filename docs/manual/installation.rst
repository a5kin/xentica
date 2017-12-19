Installation Instructions
=========================

**Xentica** is planned to run with several GPU backends in future,
like *CUDA*, *OpenCL* and *GLSL*. However, right now, only *CUDA* is
supported.

.. warning::
   If your GPU is not CUDA-enabled, this guide is **not** for
   you. Framework will just not run, no matter how hard you try. You
   may check the `list of CUDA-enabled cards`_, if you have any doubts.

.. warning::
   This page *may* give you some insights on how to set up your
   environment and run Xentica examples, but without any
   guarantees. More detailed instructions are coming soon.

Core Prerequisites
------------------

In order to run CA models without any visualization, you have to
correctly install:

- `NVIDIA CUDA Toolkit`_

- `Python 3.5+`_

- `NumPy`_

- `PyCUDA`_

- `cached-property`_

Possible solution for Debian-like systems::

  $ sudo apt-get install python-pycuda nvidia-cuda-toolkit python3
  $ sudo pip3 install numpy
  $ sudo pip3 install cached-property

GUI Prerequisites
-----------------

If you are planning to run visual examples with `Moire`_ GUI, you have to
install some extra things:

- `Kivy framework`_

Possible solution for Debian-like systems::

  $ sudo pip3 install Cython==0.23 Kivy==1.9.1


Run Xentica examples
--------------------

In order to run Game of Life model built with Xentica:

1. Clone `Xentica`_ and `Moire`_ repositories.

2. Put them on Python path.

3. Run ``xentica/examples/game_of_life.py`` with Python 3 interpreter.

Possible solution for Debian-like systems::

  $ mkdir artipixoids
  $ cd artipixoids
  $ git clone https://github.com/a5kin/xentica.git
  $ git clone https://github.com/a5kin/moire.git
  $ PYTHONPATH="$(pwd)/xentica/:$(pwd)/moire/:$PYTHONPATH" python3 ./xentica/examples/game_of_life.py

.. _list of CUDA-enabled cards: https://developer.nvidia.com/cuda-gpus
.. _NVIDIA CUDA Toolkit: http://docs.nvidia.com/cuda/index.html
.. _Python 3.5+: https://www.python.org/downloads/
.. _NumPy: https://docs.scipy.org/doc/
.. _PyCUDA: https://wiki.tiker.net/PyCuda/Installation
.. _cached-property: https://pypi.python.org/pypi/cached-property
.. _Kivy framework: https://kivy.org/docs/installation/installation.html
.. _Moire: https://github.com/a5kin/moire
.. _Xentica: https://github.com/a5kin/xentica
