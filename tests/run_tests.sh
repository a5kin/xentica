#!/bin/bash
cd ..
coverage run --source hecate -m unittest discover -s tests
echo ""
echo "COVERAGE"
echo "========"
coverage report -m
echo ""
echo "PERFORMANCE"
echo "==========="
CURDIR=$(pwd)
PYTHONPATH="$CURDIR:$PYTHONPATH" python3 tests/performance.py
