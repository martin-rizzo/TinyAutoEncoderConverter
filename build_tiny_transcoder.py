"""
File    : build_tiny_transcoder.py
Purpose : Command-line tool to build a `Tiny Transcoder` model.
          (Tiny Transcoders are small models to efficiently convert latent images from one space to another)
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Nov 27, 2024
Repo    : https://github.com/martin-rizzo/TinyAutoEncoderConverter
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                          Tiny AutoEncoder Converter
   Command-line tool to build VAEs and Transcoders (from Tiny AutoEncoders)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import os
import sys
import argparse
import numpy as np
VALID_MODEL_CLASSES = ("sd", "sdxl", "sd3", "f1")


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

    # parse the arguments and check that they are valid
    args = parser.parse_args(args)

   # determine if color should be used
    use_color = args.color_always or (args.color and is_terminal_output())
    if not use_color:
        disable_colors()

    # check what model to use as decoder
    from_model_class = ""
    decoder_path     = ""
    if args.from_sd:
        from_model_class = "sd"
        decoder_path     = args.from_sd
    elif args.from_sdxl:
        from_model_class = "sdxl"
        decoder_path     = args.from_sdxl
    elif args.from_sd3:
        from_model_class = "sd3"
        decoder_path     = args.from_sd3
    elif args.from_flux:
        from_model_class = "f1"
        decoder_path     = args.from_flux

    # check what model to use as encoder
    to_model_class = ""
    encoder_path   = ""
    if args.to_sd:
        to_model_class = "sd"
        encoder_path   = args.to_sd
    elif args.to_sdxl:
        to_model_class = "sdxl"
        encoder_path   = args.to_sdxl
    elif args.to_sd3:
        to_model_class = "sd3"
        encoder_path   = args.to_sd3
    elif args.to_flux:
        to_model_class = "f1"
        encoder_path   = args.to_flux

    # check that source/destination models are specified
    if not from_model_class:
        fatal_error("A model must be specified as the source for transcoding (--from_sd, --from_sdxl, etc.)")
    if not to_model_class:
        fatal_error("A model must be specified as the destination for transcoding (--to_sd, --to_sdxl, etc.)")

    # find the encoder and decoder files and their tensor prefixes
    encoder_path, encoder_source_prefix = find_taesd_with_role(encoder_path, role="encoder")
    decoder_path, decoder_source_prefix = find_taesd_with_role(encoder_path, role="decoder")

    if not encoder_path:
        fatal_error("No TAESD encoder model found.")
    if not decoder_path:
        fatal_error("No TAESD decoder model found.")

    print(f"Encoder file: {encoder_path}, Tensor prefix: {encoder_source_prefix}")
    print(f"Decoder file: {decoder_path}, Tensor prefix: {decoder_source_prefix}")

    # state_dict = build_tiny_transcoder(encoder_path_and_prefix = (encoder_path, encoder_source_prefix),
    #                                    decoder_path_and_prefix = (decoder_path, decoder_source_prefix),
    #                                    from_model_class        = from_model_class,
    #                                    to_model_class          = to_model_class,
    #                                    dtype                   = args.dtype,
    #                                    )

    # # find a unique path for the output file
    # output_file_path = f"tiny_vae_{args.model_class}{get_dtype_name(args.dtype,'_')}.safetensors"
    # if args.output_dir:
    #     if not os.path.exists(args.output_dir):
    #         os.makedirs(args.output_dir)
    #     output_file_path = os.path.join(args.output_dir, output_file_path)
    # output_file_path = find_unique_path(output_file_path)

    # # save the state dict to a file
    # print(f"Saving VAE to {output_file_path}")
    # save_file(state_dict, output_file_path)


if __name__ == "__main__":
    main()
