"""
File    : build_auxiliary.py
Purpose : Command-line tool to build the auxiliary safetensor file used by other project.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Jan 19, 2025
Repo    : https://github.com/martin-rizzo/TinyModelsForLatentConversion
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Tiny Models for Latent Conversion
   Build fast VAEs and latent Transcoders models using Tiny AutoEncoders
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import os
import sys
import argparse
import numpy as np
from safetensors       import safe_open
from safetensors.numpy import save_file

#---------------------------- COLORED MESSAGES -----------------------------#

GRAY   = "\033[90m"
RED    = '\033[91m'
GREEN  = '\033[92m'
YELLOW = '\033[93m'
BLUE   = '\033[94m'
CYAN   = '\033[96m'
DEFAULT_COLOR = '\033[0m'
FATAL_ERROR_CODE = 1

def disable_colors():
    """Disables colored messages."""
    global GRAY, RED, GREEN, YELLOW, BLUE, CYAN, DEFAULT_COLOR
    GRAY, RED, GREEN, YELLOW, BLUE, CYAN, DEFAULT_COLOR = '', '', '', '', '', '', ''

def warning(message: str, *info_messages: str) -> None:
    """Displays and logs a warning message to the standard error stream."""
    print(f"{CYAN}[{YELLOW}WARNING{CYAN}]{YELLOW} {message}{DEFAULT_COLOR}", file=sys.stderr)
    for info_message in info_messages:
        print(f"          {YELLOW}{info_message}{DEFAULT_COLOR}", file=sys.stderr)
    print()

def error(message: str, *info_messages: str) -> None:
    """Displays and logs an error message to the standard error stream."""
    print()
    print(f"{CYAN}[{RED}ERROR{CYAN}]{RED} {DEFAULT_COLOR}{message}", file=sys.stderr)
    for info_message in info_messages:
        print(f"          {RED}{info_message}{DEFAULT_COLOR}", file=sys.stderr)
    print()

def fatal_error(message: str, *info_messages: str) -> None:
    """Displays and logs an fatal error to the standard error stream and exits.
    Args:
        message       : The fatal error message to display.
        *info_messages: Optional informational messages to display after the error.
    """
    error(message)
    for info_message in info_messages:
        print(f" {CYAN}\u24d8  {info_message}{DEFAULT_COLOR}", file=sys.stderr)
    exit(FATAL_ERROR_CODE)


#--------------------------------- HELPERS ---------------------------------#

def is_terminal_output():
    """Return True if the standard output is connected to a terminal."""
    return sys.stdout.isatty()


def get_file_name_tag(object, prefix: str = "") -> str:
    """Returns a string that identifies the provided object/text.
       (the string is used as a tag in file names)."""
    if object == np.float16:
        tag = "fp16"
    elif object == np.float32:
        tag = "fp32"
    elif isinstance(object, float):
        tag = f"{object:.3g}".replace('.','')
    elif object is not None:
        tag = str(object)
    else:
        tag = ""
    return f"{prefix}{tag}" if tag else ""


def find_unique_path(path: str) -> str:
    """Returns the first available path to not overwrite an existing file."""
    if not os.path.exists(path):
        return path
    base_name, extension = os.path.splitext(path)
    for number in range(1, 1000000):
        new_path = f"{base_name}_{number:02d}{extension}"
        if not os.path.exists(new_path) or number == 999999:
            return new_path


#--------------------------------- TENSORS ---------------------------------#

def load_tensors(path         : str,
                 prefix       : str,
                 *,# keyword-only arguments #
                 target_prefix: str = ""
                 ):
    """
    Load tensors with the specified prefix from a safetensors file.

    Args:
        path          (str): The path to the safetensors file.
        prefix        (str): The prefix of the tensors to load.
        target_prefix (str): The prefix used as replacement of the original prefix.
                             If empty, the original prefix is removed and not replaced.
    Returns:
        dict: A dictionary containing the loaded tensors.
    """
    # ensure the prefixes end with a dot
    if prefix and not prefix.endswith('.'):
        prefix += '.'
    if target_prefix and not target_prefix.endswith('.'):
        target_prefix += '.'

    # load the tensors from the file with the specified prefix
    tensors = {}
    prefix_len = len(prefix)
    with safe_open(path, framework="numpy", device='cpu') as f:
        for key in f.keys():
            if key.startswith(prefix):
                target_key = target_prefix + key[prefix_len:]
                tensors[target_key] = f.get_tensor(key)

    return tensors

def load_encoder_decoder(path         : str,
                         prefix       : str,
                         *,# keyword-only arguments #
                         target_prefix: str = ""
                         ):
    """
    Load encoder/decoder tensors with the specified prefix from a safetensors file.

    Args:
        path          (str): The path to the safetensors file.
        prefix        (str): The prefix of the tensors to load.
        target_prefix (str): The prefix used as replacement of the original prefix.
                             If empty, the original prefix is removed and not replaced.
    Returns:
        dict: A dictionary containing the loaded tensors.
    """
    # ensure the prefixes end with a dot
    if prefix and not prefix.endswith('.'):
        prefix += '.'
    if target_prefix and not target_prefix.endswith('.'):
        target_prefix += '.'

    encoder_prefix = f"{target_prefix}encoder."
    decoder_prefix = f"{target_prefix}decoder."

    state_dict = load_tensors(path, prefix)
    output     = { }
    for key, tensor in state_dict.items():

        if   key.startswith("taesd_encoder."):
            output[key.replace("taesd_encoder.", encoder_prefix, 1)] = tensor

        elif key.startswith("taesd_decoder."):
            output[key.replace("taesd_decoder.", decoder_prefix, 1)] = tensor

        elif key == "vae_scale":
            output[f"{encoder_prefix}vae_scale"] = tensor
            output[f"{decoder_prefix}vae_scale"] = tensor

        elif key == "vae_shift":
            output[f"{encoder_prefix}vae_shift"] = tensor
            output[f"{decoder_prefix}vae_shift"] = tensor

    return output


#-------------------------------- BUILDING ---------------------------------#

def build_auxiliary(tiny_vae_sd    : str,
                    tiny_vae_sdxl  : str,
                    tiny_transcoder: str,
                    dtype          : np.dtype,
                    ) -> dict:
    """
    Build the auxiliary safetensor file used by other project.

    Args:
        tiny_vae_sd               : The path to the safetensors file of the TinyVAE SD model.
        tiny_vae_sdxl             : The path to the safetensors file of the TinyVAE SDXL model.
        tiny_transcoder_sdxl_to_sd: The path to the safetensors file of the TinyTranscoder SDXL-to-SD model.
    """
    tiny_vae_sd_tensors     = load_encoder_decoder(tiny_vae_sd, "",
                                                   target_prefix = "first_stage_model.sd"
                                                   )
    tiny_vae_sdxl_tensors   = load_encoder_decoder(tiny_vae_sdxl, "",
                                                   target_prefix = "first_stage_model.xl"
                                                   )
    tiny_transcoder_tensors = load_tensors(tiny_transcoder, "",
                                           target_prefix = "transcoder"
                                           )
    state_dict = { **tiny_vae_sd_tensors, **tiny_vae_sdxl_tensors, **tiny_transcoder_tensors }

    # convert the data type (if required)
    if dtype:
        converted_tensors = {}
        for key, tensor in state_dict.items():
            converted_tensors[key] = tensor.astype(dtype) if isinstance(tensor, np.ndarray) else tensor
        state_dict = converted_tensors

    return state_dict


#===========================================================================#
#////////////////////////////////// MAIN ///////////////////////////////////#
#===========================================================================#

def main(args: list=None, parent_script: str=None):

    # allow this command to be a subcommand of a larger tool (future expansion?)
    prog = None
    if parent_script:
        prog = parent_script + ' ' + os.path.basename(__file__).split('.')[0]

    # start parsing the arguments
    parser = argparse.ArgumentParser(prog=prog,
        description="Build an auxiliary safetensor file used by other project.",
        formatter_class=argparse.RawTextHelpFormatter,
        )
    parser.add_argument("-o", "--output-dir", type=str            , help="the output directory where the model will be saved")
    parser.add_argument("-c", "--color"     , action="store_true" , help="use color output when connected to a terminal")
    parser.add_argument("--color-always"    , action="store_true" , help="always use color output")
    parser.add_argument("-s", "--sd"        , help="a tiny vae model for SD1.5 (generated by build_tiny_vae)")
    parser.add_argument("-x", "--sdxl"      , help="a tiny vae model for SDXL (generated by build_tiny_vae)")
    parser.add_argument("-t", "--transcoder", help="a tiny transcoder model for converting SDXL to SD1.5 (generated by build_tiny_transcoder)")

    _group = parser.add_mutually_exclusive_group()
    _group.add_argument(     "--float16"    , dest="dtype", action="store_const", const=np.float16, help="store the built auxiliary file as float16")
    _group.add_argument(     "--float32"    , dest="dtype", action="store_const", const=np.float32, help="store the built auxiliary file as float32")

    # parse the arguments and check that they are valid
    args = parser.parse_args(args)

    # determine if color should be used
    use_color = args.color_always or (args.color and is_terminal_output())
    if not use_color:
        disable_colors()

    # check that the required models were specified
    if not args.sd:
        fatal_error("A Tiny VAE SD model must be specified (--sd)",
                    "Please provide the path to the safetensors file of a Tiny VAE SD model generated with build_tiny_vae.py")
    if not args.sdxl:
        fatal_error("A Tiny VAE SDXL model must be specified (--sdxl)",
                    "Please provide the path to the safetensors file of a Tiny VAE SDXL model generated with build_tiny_vae.py")
    if not args.transcoder:
        fatal_error("A Tiny Transcoder SDXL-to-SD model must be specified (--transcoder)",
                    "Please provide the path to the safetensors file of a Tiny Transcoder SDXL-to-SD model generated with build_tiny_transcoder.py")

    # build the auxiliary state dictionary
    dtype      = args.dtype or np.float16
    state_dict = build_auxiliary(tiny_vae_sd     = args.sd,
                                 tiny_vae_sdxl   = args.sdxl,
                                 tiny_transcoder = args.transcoder,
                                 dtype           = dtype
                                 )

    # generate a unique name for the output file
    # (the name is based on the model class names and data type)
    dtype_name       = get_file_name_tag(dtype, prefix='_')
    output_file_path = f"auxiliary{dtype_name}.safetensors"
    if args.output_dir:
        if not os.path.exists(args.output_dir):
            fatal_error(f"The specified output directory does not exist. '{args.output_dir}'")
        output_file_path = os.path.join(args.output_dir, output_file_path)
    output_file_path = find_unique_path(output_file_path)

    # save the state dict to a file
    print(f' > Saving "{output_file_path}"\n')
    save_file(state_dict, output_file_path)


if __name__ == "__main__":
    main()
