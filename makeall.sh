#!/usr/bin/env bash
# File    : makeall.sh
# Purpose : Generate all models (VAEs and transcoders) from the Tiny AutoEncoder models.
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

MODEL_DIR='./original_taesd_models'
ORIGINAL_MODEL='diffusion_pytorch_model.safetensors'

SD_DIR="$MODEL_DIR/taesd"
SDXL_DIR="$MODEL_DIR/taesdxl"
SD3_DIR="$MODEL_DIR/taesd3"
FLUX_DIR="$MODEL_DIR/taef1"


EXTRA_PARAMS=( '--color' )

# analizar uno por uno los parametros
for arg in "$@"; do
    case $arg in
        --half|--float16)
            EXTRA_PARAMS+=( '--float16' )
            shift
            ;;
        --float32)
            EXTRA_PARAMS+=( '--float32' )
            shift
            ;;
        *)
            echo "ERROR: Unknown argument $arg"
            exit 1
            ;;
    esac
done
echo

# BUILD VAEs FOR ALL MODELS
./build_tiny_vae.sh "${EXTRA_PARAMS[@]}" --sd    "$SD_DIR"/*
./build_tiny_vae.sh "${EXTRA_PARAMS[@]}" --sdxl  "$SDXL_DIR"/*
./build_tiny_vae.sh "${EXTRA_PARAMS[@]}" --sd3   "$SD3_DIR"/*
./build_tiny_vae.sh "${EXTRA_PARAMS[@]}" --flux  "$FLUX_DIR"/*

# BUILD TRANSCODERS
./build_tiny_transcoder.sh "${EXTRA_PARAMS[@]}" --from-sdxl "$SDXL_DIR/$ORIGINAL_MODEL" --to-sd   "$SD_DIR/$ORIGINAL_MODEL"
./build_tiny_transcoder.sh "${EXTRA_PARAMS[@]}" --from-sd   "$SD_DIR/$ORIGINAL_MODEL"   --to-sdxl "$SDXL_DIR/$ORIGINAL_MODEL"

