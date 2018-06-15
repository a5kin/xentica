#!/bin/bash
cd ..
find . -name "*.py" | grep -v .tox | xargs pylint --rcfile=pylintrc
