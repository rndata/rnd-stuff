#!/bin/bash

export PYTHONPATH=`pwd`

poetry run python -m ipykernel install --user --name=rnd-stuff
poetry run jupyter notebook --port 1337
