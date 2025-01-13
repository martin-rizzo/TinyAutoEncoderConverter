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
    -h, --help         Show this help message and exit
    --float16, --half  Build models in float16
    --float32          Build models in float32
    --clean            Clean output directory
"
SOURCE_DIR='./original_taesd_models'
OUTPUT_DIR='./output'
ORIGINAL_MODEL_NAME='diffusion_pytorch_model.safetensors'

# directories for each model type
declare -A MODEL_DIRS
MODEL_DIRS[sd]="$SOURCE_DIR/taesd"
MODEL_DIRS[sdxl]="$SOURCE_DIR/taesdxl"
MODEL_DIRS[sd3]="$SOURCE_DIR/taesd3"
MODEL_DIRS[flux]="$SOURCE_DIR/taef1"
MODELS=( sd sdxl sd3 flux )

# python script to build the models
BUILD_TINY_VAE="./build_tiny_vae.sh"
BUILD_TINY_TRANSCODER="./build_tiny_transcoder.sh"


# function to generate all combinations of models without repeating the same pair
generate_model_combinations() {
    for i in "${MODELS[@]}"; do
        for j in "${MODELS[@]}"; do
            # skip the pairs where both models are the same
            if [[ $i != "$j" ]]; then
                echo "$i $j"
            fi
        done
    done
}


#===========================================================================#
#////////////////////////////////// MAIN ///////////////////////////////////#
#===========================================================================#

# extra parameters to pass to the build scripts
# (by default enable color and set output directory to $OUTPUT_DIR)
EXTRA_PARAMS=( '--color' '--output-dir' "$OUTPUT_DIR" )

# loop through the arguments and set the corresponding parameters
SHOW_HELP=false
CLEAN=false
for arg in "$@"; do
    case $arg in
        --float16|--half)
            EXTRA_PARAMS+=( '--float16' )
            ;;
        --float32)
            EXTRA_PARAMS+=( '--float32' )
            ;;
        --clean)
            CLEAN=true
            ;;
        -h|--help)
            SHOW_HELP=true
            ;;
        *)
            echo "ERROR: Unknown argument '$arg'"
            echo "use $0 -h for help"
            exit 1
            ;;
    esac
done
if [[ $CLEAN == true ]]; then
    echo "Cleaning output directory..."
    rm -v "$OUTPUT_DIR"/*.safetensors 
    exit 0
fi
if [[ $SHOW_HELP == true ]]; then
    echo "$HELP"
    exit 0
fi
echo


# BUILD VAEs FOR ALL MODELS
for model in "${MODELS[@]}"; do
    echo "Building VAE for $model..."
    model_dir="${MODEL_DIRS[$model]}"
    "$BUILD_TINY_VAE" "${EXTRA_PARAMS[@]}" "--$model" "$model_dir/"*
done

# GENERATE COMBINATIONS OF MODELS
combinations=$(generate_model_combinations)

# BUILD ALL TRANSCODERS
while read -r from to; do
    if [ "$from" != "$to" ]; then
        from_original_model="${MODEL_DIRS[$from]}/$ORIGINAL_MODEL_NAME"
        to_original_model="${MODEL_DIRS[$to]}/$ORIGINAL_MODEL_NAME"
        echo "Building transcoder from $from to $to..."
        "$BUILD_TINY_TRANSCODER" "${EXTRA_PARAMS[@]}" "--from-$from" "$from_original_model" "--to-$to" "$to_original_model"
    fi
done <<< "$combinations"

# BUILD ALL TRANSCODERS WITH BLUR
while read -r from to; do
    if [ "$from" != "$to" ]; then
        from_original_model="${MODEL_DIRS[$from]}/$ORIGINAL_MODEL_NAME"
        to_original_model="${MODEL_DIRS[$to]}/$ORIGINAL_MODEL_NAME"
        echo "Building transcoder with blur layer from $from to $to..."
        "$BUILD_TINY_TRANSCODER" "${EXTRA_PARAMS[@]}" --blur 0.5 "--from-$from" "$from_original_model" "--to-$to" "$to_original_model"
    fi
done <<< "$combinations"

