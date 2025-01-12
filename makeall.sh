#!/usr/bin/env bash
# File    : makeall.sh
# Purpose : Build all models (VAEs and transcoders) from the Tiny AutoEncoder models.
# Author  : Martin Rizzo | <martinrizzo@gmail.com>
# Date    : Nov 26, 2024
# Repo    : https://github.com/martin-rizzo/TinyModelsForLatentConversion
# License : MIT
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#                      Tiny Models for Latent Conversion
#   Build fast VAEs and latent Transcoders models using Tiny AutoEncoders
#
#   Copyright (c) 2024-2025 Martin Rizzo
#
#   Permission is hereby granted, free of charge, to any person obtaining
#   a copy of this software and associated documentation files (the
#   "Software"), to deal in the Software without restriction, including
#   without limitation the rights to use, copy, modify, merge, publish,
#   distribute, sublicense, and/or sell copies of the Software, and to
#   permit persons to whom the Software is furnished to do so, subject to
#   the following conditions:
#
#   The above copyright notice and this permission notice shall be
#   included in all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
#   IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
#   CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#   TORT OR OTHERWISE, ARISING FROM,OUT OF OR IN CONNECTION WITH THE
#   SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
HELP="
Usage: $0 [options]
Options:
    -h, --help        Show this help message and exit
    --float16 --half  Build models in float16
    --float32         Build models in float32
"
MODEL_DIR='./original_taesd_models'
ORIGINAL_MODEL='diffusion_pytorch_model.safetensors'

SD_DIR="$MODEL_DIR/taesd"
SDXL_DIR="$MODEL_DIR/taesdxl"
SD3_DIR="$MODEL_DIR/taesd3"
FLUX_DIR="$MODEL_DIR/taef1"

# extra parameters to pass to the build scripts
# (by default enable color output)
EXTRA_PARAMS=( '--color' )

# loop through the arguments and set the corresponding parameters
SHOW_HELP=false
for arg in "$@"; do
    case $arg in
        --float16|--half)
            EXTRA_PARAMS+=( '--float16' )
            shift
            ;;
        --float32)
            EXTRA_PARAMS+=( '--float32' )
            shift
            ;;
        -h|--help)
            SHOW_HELP=true
            ;;
        *)
            echo "ERROR: Unknown argument $arg"
            exit 1
            ;;
    esac
done

if [[ $SHOW_HELP == true ]]; then
    echo "$HELP"
    exit 0
fi
echo

# BUILD VAEs FOR ALL MODELS
./build_tiny_vae.sh "${EXTRA_PARAMS[@]}" --sd    "$SD_DIR"/*
./build_tiny_vae.sh "${EXTRA_PARAMS[@]}" --sdxl  "$SDXL_DIR"/*
./build_tiny_vae.sh "${EXTRA_PARAMS[@]}" --sd3   "$SD3_DIR"/*
./build_tiny_vae.sh "${EXTRA_PARAMS[@]}" --flux  "$FLUX_DIR"/*

# BUILD TRANSCODERS
./build_tiny_transcoder.sh "${EXTRA_PARAMS[@]}" --blur 0.5 --from-sdxl "$SDXL_DIR/$ORIGINAL_MODEL" --to-sd   "$SD_DIR/$ORIGINAL_MODEL"
./build_tiny_transcoder.sh "${EXTRA_PARAMS[@]}"            --from-sd   "$SD_DIR/$ORIGINAL_MODEL"   --to-sdxl "$SDXL_DIR/$ORIGINAL_MODEL"

