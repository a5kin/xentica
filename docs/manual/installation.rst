Installation Instructions
=========================

**Xentica** is planned to run with several GPU backends in future,
like *CUDA*, *OpenCL* and *GLSL*. However, right now, only *CUDA* is
supported.

.. warning::
   If your GPU is not CUDA-enabled, this guide is **not** for
   you. Framework will just not run, no matter how hard you try. You
   may check the `list of CUDA-enabled cards`_, if you have any doubts.

.. note::
   This page currently containing instructions only for Debian-like
   systems. If you are on other system, you still can use links to
   pre-requisites in order to install them. If so, please contact us
   by `opening an issue`_ on GitHub. We could help you if you'll meet
   some troubles during installation, and also your experience could
   be used to improve this document.

Core Prerequisites
------------------

In order to run CA models without any visualization, you have to
correctly install following software.

- `NVIDIA CUDA Toolkit`_
  
  Generally, you can install it just from your distrubution's repository::
    
    sudo apt-get install nvidia-cuda-toolkit

  Although, default packages are often out of date, so in case you
  have one of those latest cool GPU, you may want to upgrade to the
  latest CUDA version from `official NVIDIA source`_. We'll hint you
  with a `good article`_ explaining how to do it. But you are really
  on your own with this stuff.

- `Python 3.5+`_
  
  Your distribution should already have all you need::

    sudo apt-get install python3 python3-dev python3-pip wheel

- `NumPy`_
  
  Once Python3 is correctly installed, you can install NumPy by::


    pip3 install numpy
- `PyCUDA`_
  
  If CUDA is correctly installed, you again can simply install PyCUDA
  with ``pip``::

    pip3 install pycuda

Other pre-requisites should be transparently installed with the main
Xentica package.

GUI Prerequisites
-----------------

If you are planning to run visual examples with `Moire`_ GUI, you have
to install `Kivy framework`_.

Its pre-requisites could be installed by::

  sudo apt-get install \
    build-essential \
    git \
    ffmpeg \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libportmidi-dev \
    libswscale-dev \
    libavformat-dev \
    libavcodec-dev \
    zlib1g-dev

Then, to install stable Kivy::

  pip3 install Cython==0.25 Kivy==1.10.0

On latest Debian distributions you can meet conflicts with
``libsdl-mixer``. Then, try to install latest developer version,
like::

  pip3 install Cython==0.27.3 git\+https://github.com/kivy/kivy.git

If you have any other troubles with that, please refer to the official
Kivy installation instructions.

Main package
------------

Xentica package could be installed with::

  pip3 install xentica

Note, it does not depends on pre-requisites described above, but you
still need to install them properly, or Xentica will not run.

Run Xentica examples
--------------------

In order to run Game of Life model built with Xentica::

  pip3 install moire
  wget https://raw.githubusercontent.com/a5kin/xentica/master/examples/game_of_life.py
  python3 game_of_life.py

Or, if you are using ``optirun``::

  optirun python3 game_of_life.py

.. _list of CUDA-enabled cards: https://developer.nvidia.com/cuda-gpus
.. _NVIDIA CUDA Toolkit: http://docs.nvidia.com/cuda/index.html
.. _Python 3.5+: https://www.python.org/downloads/
.. _NumPy: https://docs.scipy.org/doc/
.. _PyCUDA: https://wiki.tiker.net/PyCuda/Installation
.. _cached-property: https://pypi.python.org/pypi/cached-property
.. _Kivy framework: https://kivy.org/docs/installation/installation.html
.. _Moire: https://github.com/a5kin/moire
.. _Xentica: https://github.com/a5kin/xentica
.. _opening an issue: https://github.com/a5kin/xentica/issues/new
.. _good article: http://www.pradeepadiga.me/blog/2017/03/22/installing-cuda-toolkit-8-0-on-ubuntu-16-04/
.. _official NVIDIA source: https://developer.nvidia.com/cuda-downloads
