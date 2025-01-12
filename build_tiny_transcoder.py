"""
File    : build_tiny_transcoder.py
Purpose : Command-line tool to build a `Tiny Transcoder` model.
          (Tiny Transcoders are small models to efficiently convert latent images from one space to another)
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Nov 27, 2024
Repo    : https://github.com/martin-rizzo/TinyModelsForLatentConversion
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Tiny Models for Latent Conversion
   Build fast VAEs and latent Transcoders models using Tiny AutoEncoders
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import os
import sys
import json
import struct
import argparse
import numpy as np
from safetensors       import safe_open
from safetensors.numpy import save_file

# `from_latent_format`/`to_latent_format` parameters
# always must contain any of the following valid formats:
VALID_LATENT_FORMATS = ("sd", "sdxl", "sd3", "f1")

# scale/shift values to apply on latents that are preprocessed by comfyUI/a1111
# (these values ensure compatibility with that software)
SCALE_SHIFT_BY_LATENT_FORMAT = {
    "sd"  : (0.18215,  0.    ),
    "sdxl": (0.13025,  0.    ),
    "sd3" : (1.5305 , -0.0609),
    "f1"  : (0.3611 , -0.1159)
}

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
    """Return True ifthe standard output is connected to a terminal."""
    return sys.stdout.isatty()


def get_file_name_tag(object, prefix: str = "") -> str:
    """Returns a string that identifies the provided object/text.
       (the string is used as a tag in file names)."""
    if object == np.float16:
        tag = "fp16"
    elif object == np.float32:
        tag = "fp32"
    elif object == "f1":
        tag = "flux"
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

def get_safetensors_header(file_path : str,
                           size_limit: int = 67108864
                           ) -> dict:
    """
    Returns a dictionary with the safetensors file header for fast content validation.
    Args:
        file_path  (str): Path to the .safetensors file.
        size_limit (int): Maximum allowed size for the header (a protection against large headers)
    """
    try:
        # verify that the file has at least 8 bytes (the minimum size for a header)
        if os.path.getsize(file_path) < 8:
            return []

        # read the first 8 bytes to get the header length and decode the header data
        with open(file_path, "rb") as f:
            header_length = struct.unpack("<Q", f.read(8))[0]
            if header_length > size_limit:
                return []
            header = json.loads( f.read(header_length) )
            return header

    # handle exceptions that may occur during header reading or decoding
    except (ValueError, json.JSONDecodeError, IOError):
        return []


def get_tensor_prefix(state_dict    : dict,
                      postfix       : str,
                      not_containing: str = None
                      ) -> str:
    """
    Returns the prefix of a key in the state dictionary that matches the given postfix.
    Args:
        state_dict    (dict): The model parameters as a dictionary.
        postfix        (str): The suffix to match at the end of the key.
        not_containing (str): If provided, specifies that the keys returned should not contain this substring.
    """
    # iterate over all keys in the state dictionary
    for key in state_dict.keys():
        if key.endswith(postfix):
            if (not_containing is not None) and (not_containing in key):
                continue
            return key[:-len(postfix)]

    # if no key matches the postfix, return an empty string
    return ""


def load_tensors(path         : str,
                 prefix       : str,
                 target_prefix: str = ""
                 ) -> dict:
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


def shift_layers(tensors: dict,
                 layer_prefix : str,
                 layer_offset : int
                 ) -> dict:
    """
    Shifts the layers of a model by a specified offset.

    Args:
        tensors       (dict): A dictionary containing the tensors to shift.
        layer_prefix   (str): The prefix used to identify the layers to shift.
        layer_offset   (int): The number of layers to shift. Positive values shift
                              the layers forward, negative values shift them backward.
    Returns:
        dict: A dictionary containing the shifted tensors.
    """
    fixed_tensors = {}
    for key, tensor in tensors.items():

        if not key.startswith(layer_prefix):
            fixed_tensors[key] = tensor
            continue

        _parts = key[len(layer_prefix):].split('.',1)
        if not _parts[0].isdecimal():
            fixed_tensors[key] = tensor
            continue

        new_layer_number = int(_parts[0]) + layer_offset
        dot_suffix       = f".{_parts[1]}" if len(_parts)>1 else ""
        fixed_tensors[f"{layer_prefix}{new_layer_number}{dot_suffix}"] = tensor

    return fixed_tensors


#----------------------------- IDENTIFICATION ------------------------------#

def is_taesd(state_dict: dict) -> bool:
    """
    Returns True if the model parameters correspond to a Tiny AutoEncoder (TAESD) model.
    Args:
        state_dict (dict): The model parameters as a dictionary.
    """
    # recognize the following files based on their structure:
    #   - taesd_decoder.safetensors
    #   - taesd_encoder.safetensors
    #   - taesdxl_decoder.safetensors
    #   - taesdxl_encoder.safetensors
    #
    if  "3.conv.4.bias"   in state_dict and \
        "8.conv.0.weight" in state_dict:
        return True

    # recognize the following diffusers files based on their structure:
    #   - diffusion_pytorch_model.safetensors (SD, SDXL, SD3 and FLUX.1 version)
    #
    if  "decoder.layers.3.conv.4.bias"   in state_dict and \
        "decoder.layers.8.conv.0.weight" in state_dict:
        return True
    if  "encoder.layers.4.conv.4.bias"   in state_dict and \
        "encoder.layers.8.conv.0.weight" in state_dict:
        return True

    # recognize any model whose tensor root name starts with some TAESD-related names
    for key in state_dict.keys():
        if key.startswith( ("taesd", "taesdxl", "taesd3", "taef1")  ):
            return True

    # none of the above conditions are met
    # therefore, it does not appear to be a `Tiny AutoEncoder` model
    return False


def is_taesd_with_role(file_path: str, state_dict: dict, role: str) -> bool:
    """
    Returns True if the model parameters correspond to a Tiny AutoEncoder (TAESD) model with a specific role.
    Args:
        file_path  (str) : The path to the model file.
        state_dict (dict): The model parameters or safetensors header.
        role       (str) : The role of the model, either 'encoder' or 'decoder'.
    """
    assert role in ("encoder", "decoder"), "Invalid role. Must be 'encoder' or 'decoder'."

    # names of tensors that betray the role of the model (encoder/decoder)
    ENCODER_TENSOR_SUBNAMES = {
        "encoder" : ("encoder", ),
        "decoder" : ("decoder", )
    }

    # check if state_dict contains any keys related to the specified role
    if not state_dict or not is_taesd(state_dict):
        return False
    subnames = ENCODER_TENSOR_SUBNAMES[role]
    for key in state_dict.keys():
        if any(subname in key for subname in subnames):
            return True

    # how last, check if the filename itself contains the role information
    file_name, _ = os.path.splitext(os.path.basename(file_path))
    return role in file_name.lower()


def find_taesd_with_role(input_files: list[str], role: str) -> tuple[str, str] | None:
    """
    Find the Tiny AutoEncoder (TAESD) model with a specific role from a list of input files.

    Args:
        input_files (list[str]): List of input file paths.
        role            (str)  : The role of the model, either 'encoder' or 'decoder'.
    Returns:
        A tuple containing the taesd model filename and its tensor prefix,
        or None if not found.
    """
    assert role in ("encoder", "decoder"), "Invalid role. Must be 'encoder' or 'decoder'."
    oposite_role = "decoder" if role == "encoder" else "encoder"
    for file in input_files:
        header = get_safetensors_header(file)
        if is_taesd_with_role(file, header, role):
            tensor_prefix = get_tensor_prefix(header, ".3.conv.4.bias", not_containing=oposite_role)
            return (file, tensor_prefix)

    return None


#-------------------------------- BUILDING ---------------------------------#

ENCODER_PREFIX="encoder."
DECODER_PREFIX="decoder."

def build_tiny_transcoder(encoder_path_and_prefix : tuple[str, str],
                          decoder_path_and_prefix : tuple[str, str],
                          use_unnormalized_adapter: bool,
                          input_latent_format     : str,
                          output_latent_format    : str,
                          dtype                   : np.dtype = None
                          ) -> dict:
    assert input_latent_format  in VALID_LATENT_FORMATS, f"Invalid input_latent_format '{input_latent_format}'"
    assert output_latent_format in VALID_LATENT_FORMATS, f"Invalid output_latent_format '{output_latent_format}'"

    encoder_tensors = load_tensors(path   = encoder_path_and_prefix[0],
                                   prefix = encoder_path_and_prefix[1],
                                   target_prefix = ENCODER_PREFIX)

    decoder_tensors = load_tensors(path   = decoder_path_and_prefix[0],
                                   prefix = decoder_path_and_prefix[1],
                                   target_prefix = DECODER_PREFIX)

    # combine the encoder and decoder parameters into a single dictionary
    transcoder_tensors = {**encoder_tensors, **decoder_tensors}

    # apply any necessary fixes to match the expected decoder format:
    #  Layer |     Tensor     |  Module
    # -------+----------------+---------------------------
    #    0   |        -       |  Clamp()
    #    1   | [64, ch, 3, 3] |  conv(latent_channels, 64)
    #    2   |        -       |  ReLU()
    #  ....  |      .....     |  ......
    #
    # if the tensor "decoder.0.weight" exists (which should not exist),
    # then shift all the decoder layers by 1
    if f"{DECODER_PREFIX}0.weight" in transcoder_tensors:
        transcoder_tensors = shift_layers(transcoder_tensors, layer_prefix=DECODER_PREFIX, layer_offset=1)

    # if the transcoder converted from SDXL to SD,
    # then add a gaussian blur process to remove high frequency noise
    if input_latent_format=="sdxl" and output_latent_format=="sd":
        transcoder_tensors["gaussian_blur_sigma"] = np.array( [0.5], dtype=np.float32 )

    # insert a compatible adapter to match the input/output latent formats of comfyui/a1111
    if use_unnormalized_adapter:
        scale_shift = SCALE_SHIFT_BY_LATENT_FORMAT.get(input_latent_format, (1.,0.))
        transcoder_tensors["unnormalized_adapter_in.scale_factor"]  = np.array( [scale_shift[0]], dtype=np.float32 )
        transcoder_tensors["unnormalized_adapter_in.shift_factor"]  = np.array( [scale_shift[1]], dtype=np.float32 )
        scale_shift = SCALE_SHIFT_BY_LATENT_FORMAT.get(output_latent_format, (1.,0.))
        transcoder_tensors["unnormalized_adapter_out.scale_factor"] = np.array( [scale_shift[0]], dtype=np.float32 )
        transcoder_tensors["unnormalized_adapter_out.shift_factor"] = np.array( [scale_shift[1]], dtype=np.float32 )

    # convert the data type (if necessary)
    if dtype is not None:
        converted_tensors = {}
        for key, tensor in transcoder_tensors.items():
            converted_tensors[key] = tensor.astype(dtype) if isinstance(tensor, np.ndarray) else tensor
        transcoder_tensors = converted_tensors

    return transcoder_tensors


#===========================================================================#
#////////////////////////////////// MAIN ///////////////////////////////////#
#===========================================================================#

def main(args: list=None, parent_script: str=None):

    # allow this command to be a subcommand of a larger tool (future expansion?)
    prog = None
    if parent_script:
        prog = parent_script + ' ' + os.path.basename(__file__).split('.')[0]

    # parse the arguments cheking if they are valid
    parser = argparse.ArgumentParser(prog=prog,
        description="Build a Tiny Transcoder model for using in ComfyUI and convert latent images.",
        formatter_class=argparse.RawTextHelpFormatter,
        )
    parser.add_argument("-c", "--color"     , action="store_true" , help="use color output when connected to a terminal")
    parser.add_argument("--color-always"    , action="store_true" , help="always use color output")
    _group = parser.add_mutually_exclusive_group(required=True)
    _group.add_argument(     "--from-sd"    , help="the Tiny AutoEncoder model with a SD1.5 decoder")
    _group.add_argument(     "--from-sdxl"  , help="the Tiny AutoEncoder model with a SDXL decoder")
    _group.add_argument(     "--from-sd3"   , help="the Tiny AutoEncoder model with a SD3 decoder")
    _group.add_argument(     "--from-flux"  , help="the Tiny AutoEncoder model with a Flux decoder")
    _group = parser.add_mutually_exclusive_group(required=True)
    _group.add_argument(     "--to-sd"      , help="the Tiny AutoEncoder model with a SD1.5 encoder")
    _group.add_argument(     "--to-sdxl"    , help="the Tiny AutoEncoder model with a SDXL encoder")
    _group.add_argument(     "--to-sd3"     , help="the Tiny AutoEncoder model with a SD3 encoder")
    _group.add_argument(     "--to-flux"    , help="the Tiny AutoEncoder model with a Flux encoder")
    _group = parser.add_mutually_exclusive_group()
    _group.add_argument(     "--float16"    , dest="dtype", action="store_const", const=np.float16, help="store the built transcoder as float16")
    _group.add_argument(     "--float32"    , dest="dtype", action="store_const", const=np.float32, help="store the built transcoder as float32") 
    args = parser.parse_args(args)

    # determine if color should be used
    use_color = args.color_always or (args.color and is_terminal_output())
    if not use_color:
        disable_colors()

    # determine which file the decoder will be loaded from
    # and the latent format to be used (sd, sdxl,...)
    from_latent_format = ""
    decoder_path       = ""
    if args.from_sd:
        from_latent_format = "sd"
        decoder_path       = args.from_sd
    elif args.from_sdxl:
        from_latent_format = "sdxl"
        decoder_path       = args.from_sdxl
    elif args.from_sd3:
        from_latent_format = "sd3"
        decoder_path       = args.from_sd3
    elif args.from_flux:
        from_latent_format = "f1"
        decoder_path       = args.from_flux

    # determine which file the encoder will be loaded from
    # and the latent format to be used (sd, sdxl,...)
    to_latent_format = ""
    encoder_path     = ""
    if args.to_sd:
        to_latent_format = "sd"
        encoder_path   = args.to_sd
    elif args.to_sdxl:
        to_latent_format = "sdxl"
        encoder_path   = args.to_sdxl
    elif args.to_sd3:
        to_latent_format = "sd3"
        encoder_path   = args.to_sd3
    elif args.to_flux:
        to_latent_format = "f1"
        encoder_path   = args.to_flux

    # check that source/destination models are specified
    if not from_latent_format:
        fatal_error("A model must be specified as the source for transcoding (--from_sd, --from_sdxl, etc.)")
    if not to_latent_format:
        fatal_error("A model must be specified as the destination for transcoding (--to_sd, --to_sdxl, etc.)")

    # find the encoder/decoder file path and tensor prefix
    encoder_path_and_prefix = find_taesd_with_role([encoder_path], role="encoder")
    decoder_path_and_prefix = find_taesd_with_role([decoder_path], role="decoder")

    if not encoder_path_and_prefix:
        fatal_error("No TAESD encoder model found.")
    if not decoder_path_and_prefix:
        fatal_error("No TAESD decoder model found.")

    print(f' - Encoder {'['+from_latent_format+']':<6} | File: "{os.path.basename(encoder_path_and_prefix[0])}" | Prefix: `{encoder_path_and_prefix[1]}`')
    print(f' - Decoder {'['+to_latent_format+']'  :<6} | File: "{os.path.basename(decoder_path_and_prefix[0])}" | Prefix: `{decoder_path_and_prefix[1]}`')
    state_dict = build_tiny_transcoder(encoder_path_and_prefix  = encoder_path_and_prefix,
                                       decoder_path_and_prefix  = decoder_path_and_prefix,
                                       use_unnormalized_adapter = True,
                                       input_latent_format      = from_latent_format,
                                       output_latent_format     = to_latent_format,
                                       dtype                    = args.dtype,
                                       )

    # generate a unique name for the output file
    # (the name is based on the model class names and data type)
    from_latent_format = get_file_name_tag(from_latent_format, prefix='_')
    to_latent_format   = get_file_name_tag(to_latent_format  , prefix='_')
    dtype_name         = get_file_name_tag(args.dtype        , prefix='_')
    output_file_path   = f"transcoder_from{from_latent_format}_to{to_latent_format}{dtype_name}.safetensors"
    output_file_path   = find_unique_path(output_file_path)

    # save the state dict to a file
    print(f' > Saving "{output_file_path}"\n')
    save_file(state_dict, output_file_path)


if __name__ == "__main__":
    main()
