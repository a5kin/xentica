Testing
=======

Prerequisites
-------------

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

Run tests
---------

To run full tests::

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

.. note::
   When planning for pull request in core Xentica repo, it is a good
   practice to run a full test suite with ``tox -q``. It
   will be accepted at minimum if everything is green =)

.. _tox: https://tox.readthedocs.io/en/latest/
.. _deadsnakes: https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa
.. _in binaries: https://pypy.org/download.html
