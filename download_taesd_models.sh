#!/usr/bin/env bash
# File    : download_taesd_models.sh
# Purpose : Download the original TAESD models from the HuggingFace model hub.
# Author  : Martin Rizzo | <martinrizzo@gmail.com>
# Date    : Jan 12, 2024
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
Usage: $0 [-h|--help]

Downloads the original TAESD models from Hugging Face.

Options:
    -h, --help    Show this help message and exit
"

# local directories
OUTPUT_DIR="./original_taesd_models"

# URLs for each TAESD model type
declare -A MODEL_URLS
MODEL_URLS[taesd]="https://huggingface.co/madebyollin/taesd/blob/main/diffusion_pytorch_model.safetensors"
MODEL_URLS[taesdxl]="https://huggingface.co/madebyollin/taesdxl/blob/main/diffusion_pytorch_model.safetensors"
MODEL_URLS[taesd3]="https://huggingface.co/madebyollin/taesd3/blob/main/diffusion_pytorch_model.safetensors"
MODEL_URLS[taef1]="https://huggingface.co/madebyollin/taef1/blob/main/diffusion_pytorch_model.safetensors"
MODELS=( taesd taesdxl taesd3 taef1 )



#===========================================================================#
#////////////////////////////////// MAIN ///////////////////////////////////#
#===========================================================================#

# loop through the arguments and set the corresponding parameters
SHOW_HELP=false
while [[ $# -gt 0 ]]; do
    arg=$1
    case "$arg" in
        -h|--help)
            SHOW_HELP=true
            ;;
        *)
            echo "ERROR: Unknown argument '$arg'"
            echo "use $0 -h for help"
            exit 1
            ;;
    esac
    shift
done
if [[ $SHOW_HELP == true ]]; then
    echo "$HELP"
    exit 0
fi

# validate output directory exists
if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "Error: The output directory '$OUTPUT_DIR' does not exist."
    exit 1
fi

# confirmation prompt
echo "This script will download the TAESD models from Hugging Face hub and"
echo "save them to the '$OUTPUT_DIR' directory."
read -r -p "Do you want to continue? (y/N): " choice
choice=$(echo "$choice" | tr '[:upper:]' '[:lower:]')
if [[ "$choice" != "y" ]]; then
    echo "Download cancelled by user."
    exit 0
fi

# download models
echo "Starting download..."
for model in "${MODELS[@]}"; do
    model_url="${MODEL_URLS[$model]}"
    model_output_dir="$OUTPUT_DIR/$model"
    filename=$(basename "$model_url")
    dest_file="$model_output_dir/$filename"

    # validate model directory exists
    if [[ ! -d "$model_output_dir" ]]; then
        echo "Error: The model directory '$model_output_dir' does not exist."
        echo "Skipping download for model '$model'."
        continue
    fi

    # verify if model file already exists in local directory
    if [[ -f "$dest_file" ]]; then
        echo "File '$filename' already exists in '$model_output_dir'."
        echo "Skipping download for model '$model'."
        continue
    fi
    echo "Downloading '$filename' to '$model_dest_dir'..."
    if ! wget "$model_url" -O "$dest_file"; then
        echo "Error downloading '$filename'. Please check the URL and your internet connection."
        exit 1
    fi
    echo "Download of '$filename' complete."

done
echo "All downloads completed."

