# TinyAutoEncoderConverter

This Python command-line tool converts Tiny AutoEncoders (TAEs) into ready-to-use Variational AutoEncoders (VAEs) and Transcoders for ComfyUI.  This allows you to leverage the speed and efficiency of TAEs within the ComfyUI workflow.

**What are Tiny AutoEncoders?**

Tiny AutoEncoders are extremely lightweight autoencoders that share the same "latent API" as Stable Diffusion's VAE.  They offer a significant advantage in speed and resource consumption when decoding Stable Diffusion latents into full-size images.

**What is ComfyUI?**

ComfyUI is a powerful and flexible node-based workflow editor for image generation and manipulation. It supports various models, including Stable Diffusion, SDXL, and Stable Video Diffusion.

**Functionality:**

This tool takes a Tiny AutoEncoder model as input and performs the following conversions:

* **VAE Conversion:** Creates a VAE compatible with ComfyUI.
* **Transcoder Conversion:** Creates a Transcoder compatible with ComfyUI.  This allows for seamless integration with other nodes in your ComfyUI workflows.

**Installation:**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your_github_username>/TinyAutoEncoderConverter.git
   ```

2. **Install dependencies:**  (You'll need to adjust this based on your specific dependencies.  Include all necessary libraries here, e.g., PyTorch, etc.)
   ```bash
   pip install -r requirements.txt
   ```

**Usage:**

```bash
python TinyAutoEncoderConverter.py --input <path_to_tae_model> --output <output_directory>
```

* `<path_to_tae_model>`: Path to your Tiny AutoEncoder model file (e.g., `.ckpt`, `.pth`).
* `<output_directory>`: Directory where the converted VAE and Transcoder files will be saved.

**Example:**

```bash
python TinyAutoEncoderConverter.py --input my_tae_model.ckpt --output ./comfyui_models
```

**Output:**

The script will generate two files in the specified output directory:

* `<model_name>_vae.safetensors`: The VAE model file for ComfyUI.
* `<model_name>_transcoder.safetensors`: The Transcoder model file for ComfyUI.


## License

Copyright (c) 2024 Martin Rizzo  
This project is licensed under the MIT license.  
See the ["LICENSE"](LICENSE) file for details.


**Disclaimer:**

This tool is provided "as is" without any warranty.  Use at your own risk.

