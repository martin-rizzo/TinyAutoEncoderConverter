# Tiny AutoEncoder Converter

**Tiny AutoEncoder Converter** is a command-line utility designed to convert Tiny AutoEncoders into Variational AutoEncoders (VAEs) and Transcoders compatible with ComfyUI.

## What are Tiny AutoEncoders?

Tiny AutoEncoders are highly optimized autoencoders that share the same latent space as Stable Diffusion and Flux VAEs. This characteristic provides a significant advantage in terms of speed and resource efficiency during the encoding and decoding processes.

## Functionality

The tools perform the following conversions:
- **VAE Conversion:** Generates a VAE model compatible with ComfyUI.
- **Transcoder Conversion:** Creates a Transcoder to convert latents from one space to another, such as converting from SDXL to SD.

## Installation and Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/martin-rizzo/TinyAutoEncoderConverter.git
   ```

2. **Run any of the bash scripts:** (For first-time use, this script will install all necessary dependencies, including libraries like Numpy.)
   ```bash
   ./makeall.sh
   ```

## Output

When running the general `makeall.sh` script, it will generate several files with the following naming format:
- `<model_name>_vae.safetensors`: VAE models compatible with ComfyUI.
- `transcoder_from_*_to_*.safetensors`: Transcoder model files.

## License

**Copyright (c) 2024 Martin Rizzo**  
This project is licensed under the MIT license.  
Details can be found in the ["LICENSE"](LICENSE) file.

## Disclaimer

This tool is provided "as is" without any warranty. Use at your own risk.
