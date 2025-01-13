<div align="center">

# Tiny Models for Latent Conversion

<p>
<img alt="Platform"  src="https://img.shields.io/badge/platform-Linux-blue">
<img alt="License"   src="https://img.shields.io/github/license/martin-rizzo/TinyModelsForLatentConversion?color=blue">
<img alt="Version"   src="https://img.shields.io/github/v/tag/martin-rizzo/TinyModelsForLatentConversion?label=version">
<img alt="Last"      src="https://img.shields.io/github/last-commit/martin-rizzo/TinyModelsForLatentConversion">
<!--
|
<a href="https://civitai.com/models/420163/abominable-workflows">
   <img alt="CivitAI"      src="https://img.shields.io/badge/page-CivitAI-00F"></a>
<a href="https://huggingface.co/martin-rizzo/AbominableWorkflows">
   <img alt="Hugging Face" src="https://img.shields.io/badge/models-HuggingFace-yellow"></a>
-->
</p>
</div>


**Tiny Models for Latent Conversion** offers a set of command-line utilities designed to generate three distinct types of models: ultra-fast ComfyUI-compatible Variational Autoencoders (VAEs); latent space Transcoders for seamless conversion between different latent spaces (e.g., SDXL to SD); and a custom `.safetensors` file specifically tailored for a project currently under development.

All generated models are based on the architecture and weights of the "Tiny AutoEncoder" models.


## What are Tiny AutoEncoders?

Tiny AutoEncoders (TAEs) are highly optimized autoencoders that share the same latent space as Stable Diffusion and Flux VAEs.  This enables significantly faster and more resource-efficient image encoding and decoding.  These models were developed and trained by Ollin Boer Bohan, to whom I extend my sincere gratitude. You can find Ollin's original implementation and pre-trained models in the [Tiny AutoEncoder Repository](https://github.com/madebyollin/taesd).


## Command-Line Tools

This project provides the following command-line conversion tools:
- `build_tiny_vae.py`: Generates a comfyui-compatible VAE model from a Tiny AutoEncoder.
- `build_tiny_transcoder.py`: Creates a transcoder enabling latent space conversion between different models (e.g., SDXL to SD).
- `build_auxiliary_model.py`: Generates a custom `.safetensors` model for an ongoing project.


## Installation and Usage

1. **Clone the Repository:**  
   First, you need to clone this repository to your local machine.
   ```bash
   git clone https://github.com/martin-rizzo/TinyModelsForLatentConversion.git
   cd TinyModelsForLatentConversion
   ```

2. **Download Original Models:**  
   Navigate into the cloned repository and download the necessary models from Hugging Face to the `original_taesd_models` directory:
   ```bash
   wget https://huggingface.co/madebyollin/taef1/blob/main/diffusion_pytorch_model.safetensors -P original_taesd_models/taef1
   wget https://huggingface.co/madebyollin/taesd/blob/main/diffusion_pytorch_model.safetensors -P original_taesd_models/taesd
   wget https://huggingface.co/madebyollin/taesd3/blob/main/diffusion_pytorch_model.safetensors -P original_taesd_models/taesd3
   wget https://huggingface.co/madebyollin/taesdx/blob/main/diffusion_pytorch_model.safetensors -P original_taesd_models/taesdx   
   ```

3. **Create the Virtual Environment and Install Dependencies:**  
   The `build_tiny_vae.sh` script came with a virtual environment setup argument, so you don't need to create a virtual environment manually.
   ```bash
   ./build_tiny_vae.sh --create-venv
   ```

4. **Run the Bash Script:**  
   Execute the `makeall.sh` script to generate all VAEs and transcoders automatically:
   ```bash
   ./makeall.sh
   ```


## Output

When running the general `makeall.sh` script, it will generate several files in the `output` directory, including VAE models and transcoders for different latent spaces. The generated files will have names like:
- `tiny_vae_*.safetensors`: VAE models compatible with ComfyUI.
- `transcoder_from_*_to_*.safetensors`: Transcoder model files.
- `auxiliary_model.safetensors`: Custom auxiliary model for a specific project.


## License

**Copyright (c) 2024-2025 Martin Rizzo**  
This project is licensed under the MIT license.  
Details can be found in the ["LICENSE"](LICENSE) file.


## Disclaimer

This tool is provided "as is" without any warranty. Use at your own risk.
