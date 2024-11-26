#!/usr/bin/env bash
# File    : makeall.sh
# Purpose : Generate all models (VAEs and transcoders) from the Tiny AutoEncoder models.
# Author  : Martin Rizzo | <martinrizzo@gmail.com>
# Date    : Nov 26, 2024
# Repo    : https://github.com/martin-rizzo/TinyAutoEncoderConverter
# License : MIT
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#                          Tiny AutoEncoder Converter
#   Command-line tool to build VAEs and Transcoders (from Tiny AutoEncoders).
#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

MODEL_DIR=./original_taesd_models
ORIGINAL_MODEL=diffusion_pytorch_model.ORIGINAL_MODEL

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


# BUILD VAEs FOR ALL MODELS
./build_tiny_vae.sh "${EXTRA_PARAMS[@]}" --sd    "$SD_DIR"/*
./build_tiny_vae.sh "${EXTRA_PARAMS[@]}" --sdxl  "$SDXL_DIR"/*
./build_tiny_vae.sh "${EXTRA_PARAMS[@]}" --sd3   "$SD3_DIR"/*
./build_tiny_vae.sh "${EXTRA_PARAMS[@]}" --flux  "$FLUX_DIR"/*

# BUILD TRANSCODERS
#./build_tiny_transcoder EXTRA_PARAMS[@] --from-sdxl "$SDXL_DIR/$ORIGINAL_MODEL" --to-sd   "$SD_DIR/$ORIGINAL_MODEL"
#./build_tiny_transcoder EXTRA_PARAMS[@] --from-sd   "$SD_DIR/$ORIGINAL_MODEL"   --to-sdxl "$SDXL_DIR/$ORIGINAL_MODEL"

