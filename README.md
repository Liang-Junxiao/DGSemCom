# DGSemCom

This repository contains the preview code for **DGSemCom: Digital Generative Semantic Communications via Discrete Denoising Diffusion Model for Latent Error Correction**.

DGSemCom is a digital generative semantic communication framework for image transmission. It combines product quantization (PQ) for discrete semantic representation with a discrete denoising diffusion prior for latent error correction under noisy channels.

> **Notice**
>
> The paper is currently under review. This repository is provided as a preliminary code release. Pretrained checkpoints, full reproduction instructions, and model hosting links will be released after the paper is accepted.

## Project Structure

```text
main.py                 Entry for codec training, prior training, and communication tests
config.py               Hyperparameters, data paths, checkpoint paths, and channel settings
PQ_net.py               PQ image representation model wrapper
quantize.py             Product quantization modules
DGSemCom.py             Discrete prior model and posterior sampling logic
VQGAN_modules/          Encoder, decoder, and attention blocks for the codec
GAN/                    PatchGAN discriminator used when adversarial training is enabled
loss/                   Distortion and perceptual losses
channel/                BSC and lightweight uncoded 4QAM AWGN channel simulation
dataloader/             Image-folder dataloaders
checkpoints/            Placeholder directory for future pretrained weights
examples/               Optional local image folders, not included in this preview release
```

## Installation

Python 3.9+ is recommended. Install PyTorch according to your CUDA version first if needed, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies:

- `torch`, `torchvision`
- `numpy`, `Pillow`, `scipy`
- `lpips`

The public channel implementation does not require TensorFlow or Sionna.

## Code Status

The current codebase includes the core components of DGSemCom:

- PQ-based image semantic codec
- discrete diffusion prior model
- posterior sampling for latent error correction
- BSC and lightweight 4QAM AWGN channel simulation
- image-folder dataloaders and basic train/test entry points

Pretrained checkpoints and example datasets are not included in this preliminary release. After acceptance, the repository will be updated with pretrained models, detailed reproduction commands, citation information, and links to externally hosted checkpoints.

## Citation

Citation information will be added after the paper is accepted.
