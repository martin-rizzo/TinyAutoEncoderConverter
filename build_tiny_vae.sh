#!/usr/bin/env bash
# File    : build_tiny_vae.sh
# Purpose : Wrapper for `build_tiny_vae.py` that handles the python virtual env
# Author  : Martin Rizzo | <martinrizzo@gmail.com>
# Date    : Jan 12, 2025
# Repo    : https://github.com/martin-rizzo/TinyModelsForLatentConversion
# License : MIT
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#                      Tiny Models for Latent Conversion
#   Build fast VAEs and latent Transcoders models using Tiny AutoEncoders
#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
SCRIPT_NAME=$(basename "${BASH_SOURCE[0]}" .sh)           # script name without extension
SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")   # script directory
PYTHON_SCRIPT="${SCRIPT_DIR}/${SCRIPT_NAME}.py"           # path to python script to run
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"        # path to requirements file

# List of options that do not trigger any action by themselves
NON_ESSENTIAL_OPTIONS=( "-c" "--color" "--color-always" )

# these variables can be set as environment variables
#   VENV_DIR: specifies the directory for python virtual environment; default is `SCRIPT_DIR/venv`
#   PYTHON  : specifies the path to the Python interpreter; default is `python3`
[[ "$VENV_DIR" ]] || VENV_DIR="${SCRIPT_DIR}/venv"
[[ "$PYTHON"   ]] || PYTHON=python3

# ANSI escape codes for colored terminal output
RED='\e[91m'
CYAN='\e[96m'
YELLOW='\e[93m'
RESET='\e[0m'

# Display a warning message
warning() {
    local message=$1
    echo
    echo -e "${CYAN}[${YELLOW}WARNING${CYAN}]${RESET} $message" >&2
}

# Display an error message
error() {
    local message=$1
    echo
    echo -e "${CYAN}[${RED}ERROR${CYAN}]${RESET} $message" >&2
}

# Displays a fatal error message and exits the script with status code 1
fatal_error() {
    local error_message=$1
    error "$error_message"
    shift
    # print informational messages, if any were provided
    while [[ $# -gt 0 ]]; do
        local info_message=$1
        echo -e " ${CYAN}\xF0\x9F\x9B\x88 $info_message${RESET}" >&2
        shift
    done
    echo
    exit 1
}

# Create and activate the python virtual environment
create_venv() {
    if [[ -d "$VENV_DIR" ]]; then
        echo "Virtual environment already exists."
        return
    fi
    echo "Creating virtual environment..."
    if ! "$PYTHON" -m venv "$VENV_DIR"; then
        fatal_error "Virtual environment creation failed." \
                    "Please check if '$PYTHON' and venv are installed on your system."
    fi
    echo "Virtual environment created."
}

# Remove the python virtual environment
remove_venv() {
    if [[ ! -d "$VENV_DIR" ]]; then
        fatal_error "No 'venv' directory found." \
                    "You must create a virtual environment before removing it." \
                    "Use the '$SCRIPT_NAME.sh --create-venv' option to create a new one."
    fi
    rm -rf "$VENV_DIR"
    echo "Virtual environment removed."
}

# Activate the python virtual environment
activate_venv() {
    if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
        fatal_error "The virtual environment does not exist." \
                    "you can use '$SCRIPT_NAME.sh --create-venv' to create it"
    fi
    # shellcheck disable=SC1091
    if ! source "$VENV_DIR/bin/activate"; then
        fatal_error "Error when activating virtual environment, it might be corrupted." \
                    "You can use '$SCRIPT_NAME.sh --recreate-venv' to recreate the virtual environment."
    fi
}

# Install dependencies from requirements.txt file if it exists
install_dependencies() {
    local requirements_file=$1
    if [[ ! -f "$requirements_file" ]]; then
        fatal_error "No '$requirements_file' file found." \
                    "Please check the project instalation instructions."
    fi
    if ! pip install --upgrade pip; then
        # failed to upgrade pip isn´t a fatal error, just a warning
        warning "Error when upgrading pip."
    fi
    if ! pip install -r "$requirements_file"; then
        fatal_error "Error when installing dependencies." \
                    "'pip' failed to install some packages, that might be due to network issues or incompatible packages."
    fi
    echo "Dependencies installed successfully."
}

# Check if a given option is non-essential
# (non-essential options do not trigger any action by themselves)
is_non_essential_option() {
    local option=$1
    [[ -z "$option" ]] && return 0
    for non_essential_option in "${NON_ESSENTIAL_OPTIONS[@]}"; do
        [[ "$option" == "$non_essential_option" ]] && return 0
    done
    return 1
}


#===========================================================================#
#////////////////////////////////// MAIN ///////////////////////////////////#
#===========================================================================#

# verify if any extra options are passed as arguments
CREATE_VENV=false
REMOVE_VENV=false
SHOW_HELP=false

if [[ $# -le 1 ]] && is_non_essential_option "$1"; then
    # if no arguments are passed, the help message will be displayed
    SHOW_HELP=true
else
    # loop through the arguments and set the corresponding
    # variables to true if they match the options
    for arg in "$@"; do
        case $arg in
            -h | --help)
                SHOW_HELP=true
                ;;
            --create-venv)
                CREATE_VENV=true
                ;;
            --remove-venv)
                REMOVE_VENV=true
                ;;
            --recreate-venv)
                CREATE_VENV=true
                REMOVE_VENV=true
                ;;
        esac
    done
fi

# handle the help option
if [[ "$SHOW_HELP" == true ]]; then
    if [[ -d "$VENV_DIR" ]]; then
        activate_venv
        "$PYTHON" "$PYTHON_SCRIPT" --help
    else
        echo "The python virtual environment does not exist."
        echo
        echo "  Please use --create-venv to create a virtual environment"
        echo "  and then run this script again to display the complete help message."
    fi
    echo
    echo "wrapper options:"
    echo "  --create-venv      Create the python virtual environment"
    echo "  --remove-venv      Remove the python virtual environment"
    echo "  --recreate-venv    Recreate the python virtual environment from zero (can solve some issues)"
    echo
    exit 0
fi

# handle the extra options for creating (and recreating) the venv
if [[ "$CREATE_VENV" == true ]]; then
    [[ "$REMOVE_VENV" == true && -d "$VENV_DIR" ]] && remove_venv
    create_venv
    activate_venv
    install_dependencies "$REQUIREMENTS_FILE"
    exit 0
fi

# handle the extra options for removing the venv
if [[ "$REMOVE_VENV" == true ]]; then
    remove_venv
    exit 0
fi

# if no extra options are passed, just run the script normally
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    python_script_name=$(basename "$PYTHON_SCRIPT")
    fatal_error "Python script not found." \
                "Please ensure that the Python script '${python_script_name}' exists in the same directory as this bash wrapper."
fi
activate_venv
"$PYTHON" "$PYTHON_SCRIPT" "$@"
