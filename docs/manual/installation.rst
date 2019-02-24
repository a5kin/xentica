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

Run tests
---------

Our test suite is based on `tox`_. It allows us to run tests in all
supported Python environments with a single ``tox`` command, and also
automates checks for package build, docs, coverage, code style and
performance.

So, in order to run Xentica tests you at minimum have to set up CUDA
environment properly (as described above), and install ``tox``::

  pip3 install tox

If you are planning to run full test suite, you also need to install
all necessary Python interpreters: 3.5-3.7 and pypy3, along with dev
headers to build numpy and pycuda.

On Ubuntu, regular Python interpreters are available with amazing
`deadsnakes`_ repo::

  sudo add-apt-repository ppa:deadsnakes/ppa
  sudo apt-get update
  sudo apt-get install python3.5 python3.5-dev
  sudo apt-get install python3.6 python3.6-dev
  sudo apt-get install python3.7 python3.7-dev

Pypy3 however comes `in binaries`_ (make sure you download latest)::

  wget -q -P /tmp https://bitbucket.org/pypy/pypy/downloads/pypy3.5-v7.0.0-linux64.tar.bz2
  sudo tar -x -C /opt -f /tmp/pypy3.5-v7.0.0-linux64.tar.bz2
  rm /tmp/pypy3.5-v7.0.0-linux64.tar.bz2
  sudo mv /opt/pypy3.5-v7.0.0-linux64 /opt/pypy3
  sudo ln -s /opt/pypy3/bin/pypy3 /usr/local/bin/pypy3

Then, to run full tests::

  git clone https://github.com/a5kin/xentica.git
  cd xentica
  tox

Or, if you are using ``optirun``::

  optirun tox

For the first time, it would take an amount of time to download /
install environtments and all its dependencies. Tox will automatically
set up all necessary stuff for you, including numpy and
pycuda. Subsequent runs should be much quicker, as everything is
already set up. In developer's environment (Ubuntu 18.04) it takes ~42
sec to finish the full test suite.

If you run tests often, it would also be helpful to get less verbose
output. For that, you could execute a strict version of tox::

  tox -q

Or if you'd like to skip all uninstalled interpreters::

  tox -s

Or even quicker, for immediate test purposes, you could run your
default Python3 interpreter tests only with codestyle and coverage::

  tox -q -e flake8,coverage

You could also check the full list of available environments with::

  tox -l -v

If you don't mind, please update us with the metrics under "Benchmark"
section, along with the info about your GPU, environment and Xentica
version. It would help us analyze performance and plan for future
optimizations.

When planning for pull request in core Xentica repo, it is a good practice to run a full test suite with ``tox -q``. Please note, it will be accepted at minimum if everything is green =)

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
.. _tox: https://https://launchpad.net/~deadsnakes/+archive/ubuntu/ppatox.readthedocs.io/en/latest/
.. _deadsnakes: https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa
.. _in binaries: https://pypy.org/download.html
