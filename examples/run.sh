#!/bin/bash

python3 -m venv venv

activate () {
    . ./venv/bin/activate
    pip3 install wheel
    pip3 install Cython==0.27.3
    pip3 install git\+https://github.com/kivy/kivy.git
    pip3 install numpy
    pip3 install pycuda
    pip3 install git\+https://github.com/a5kin/xentica.git
    pip3 install git\+https://github.com/a5kin/moire.git
    python3 game_of_life.py
}

activate

