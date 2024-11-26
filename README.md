# Tiny AutoEncoder Converter

**Tiny AutoEncoder Converter** is a command-line utility designed to convert Tiny AutoEncoders into Variational AutoEncoders (VAEs) and Transcoders compatible with ComfyUI.


## What are Tiny AutoEncoders?

Tiny AutoEncoders are highly optimized autoencoders that share the same latent space as Stable Diffusion and Flux VAEs. This characteristic provides a significant advantage in terms of speed and resource efficiency during the encoding and decoding processes.

## Functionality

The tools perform the following conversions:
- **VAE Conversion:** Generates a VAE model compatible with ComfyUI.
- **Transcoder Conversion:** Creates a Transcoder to convert latents from one space to another, such as converting from SDXL to SD.


## Installation and Usage

1. **Clone the Repository:**  
   First, you need to clone this repository to your local machine.
   ```bash
   git clone https://github.com/martin-rizzo/TinyAutoEncoderConverter.git
   ```

2. **Download Original Models:**  
   Navigate into the cloned directory and download the necessary models from Hugging Face.
   ```bash
   cd TinyAutoEncoderConverter
   wget https://huggingface.co/madebyollin/taef1/blob/main/diffusion_pytorch_model.safetensors -P original_taesd_models/taef1
   wget https://huggingface.co/madebyollin/taesd/blob/main/diffusion_pytorch_model.safetensors -P original_taesd_models/taesd
   wget https://huggingface.co/madebyollin/taesd3/blob/main/diffusion_pytorch_model.safetensors -P original_taesd_models/taesd3
   wget https://huggingface.co/madebyollin/taesdx/blob/main/diffusion_pytorch_model.safetensors -P original_taesd_models/taesdx   
   ```

4. **Run the Bash Script:**  
   Execute the `makeall.sh` script to generate.
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
