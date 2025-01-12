# ./original_taesd_models

In this folder, you need to download the different versions of the original Tiny Autoencoder (TAESD) models.  
Each model version should be stored in its corresponding subdirectory.


## Directory Structure and Files

- **/taef1:** place the `diffusion_pytorch_model.safetensors` for Flux.1 here.
- **/taesd:** place the `diffusion_pytorch_model.safetensors` for SD1.5 here.
- **/taesd3:** place the `diffusion_pytorch_model.safetensors` for SD3 here.
- **/taesdxl:** place the `diffusion_pytorch_model.safetensors` for SDXL here.


## Where can I find the models?

The original TAESD models can be found on the Hugging Face model hub:
- [Flux.1](https://huggingface.co/madebyollin/taef1/tree/main)
- [SD1.5](https://huggingface.co/madebyollin/taesd/tree/main)
- [SD3](https://huggingface.co/madebyollin/taesd3/tree/main)
- [SDXL](https://huggingface.co/madebyollin/taesdxl/tree/main)


## How to download automatically the models?

You can use the `download_taesd_models.sh` script provided with the project to automatically download the TAESD models.

To run the script you must be in the root directory of the project and execute:
```bash
./download_taesd_models.sh
```

This will download the models from the Hugging Face model hub of @madebyollin and place them in the corresponding subdirectories.
