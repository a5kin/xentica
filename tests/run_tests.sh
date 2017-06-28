#!/bin/bash
cd ..
coverage run --source hecate -m unittest discover -s tests
echo ""
coverage report -m