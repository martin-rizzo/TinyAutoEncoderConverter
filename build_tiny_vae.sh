#!/usr/bin/env bash
# File    : build_tiny_vae.sh
# Purpose : Wrapper for `build_tiny_vae.py` that automatically handles the python virtual env
# Author  : Martin Rizzo | <martinrizzo@gmail.com>
# Date    : Nov 26, 2024
# Repo    : https://github.com/martin-rizzo/TinyAutoEncoderConverter
# License : MIT
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#                          Tiny AutoEncoder Converter
#   Command-line tool to build VAEs and Transcoders (from Tiny AutoEncoders).
#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
SCRIPT_NAME=$(basename "${BASH_SOURCE[0]}" .sh)         # script name without extension
SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")") # script directory

# Define directory paths
VENV_DIR="$SCRIPT_DIR/venv"
LIB_DIR="$SCRIPT_DIR/bashlib"

# Source the virtual environment library
#shellcheck source=/dev/null
source "$LIB_DIR/lib_venv.sh"

# Compatible version of Python to be used in the virtual environment.
# Usually 'python' (the default installed version) is sufficient,
# but a different version could be used by writing the full path
PYTHON='python'

# Execute the Python script within the virtual environment
require_venv "$VENV_DIR" "$PYTHON" "$SCRIPT_DIR/requirements.txt"
virtual_python "$SCRIPT_DIR/${SCRIPT_NAME%%-*}.py" "$@"
