from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import special


def makedirs(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)


def logger_configuration(filename, phase, save_log=True, root="runs"):
    workdir = Path(root) / filename
    if phase == "test":
        workdir = Path(str(workdir) + "_test")
    makedirs(workdir / "samples")
    makedirs(workdir / "models")

    logger = logging.getLogger(f"{filename}_{phase}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s", "%Y-%m-%d %H:%M:%S")

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if save_log:
        file_handler = logging.FileHandler(workdir / f"Log_{filename}.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return str(workdir), logger


class AverageMeter:
    def __init__(self):
        self.clear()

    def update(self, val, n=1):
        if torch.is_tensor(val):
            val = float(val.detach().mean().cpu())
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)

    def clear(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0


def save_model(model, save_path):
    makedirs(Path(save_path).parent)
    torch.save(model.state_dict(), save_path)


def load_weights(net, model_path, strict=False, map_location=None):
    state = torch.load(model_path, map_location=map_location)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    filtered = {}
    for key, value in state.items():
        key = key.replace("module.", "")
        filtered[key] = value
    return net.load_state_dict(filtered, strict=strict)


def CalcuPSNR_int(img1, img2, max_val=255.0):
    img1 = np.round(torch.clamp(img1, 0, 1).detach().cpu().numpy() * 255.0)
    img2 = np.round(torch.clamp(img2, 0, 1).detach().cpu().numpy() * 255.0)
    mse = np.mean(np.square(img1.astype(np.float64) - img2.astype(np.float64)), axis=(1, 2, 3))
    mse = np.maximum(mse, 1e-12)
    return 20 * np.log10(max_val) - 10 * np.log10(mse)


def tensor2img(data):
    data = torch.clamp(data.detach(), 0, 1)
    img = (data[0] * 255.0).cpu().numpy().transpose(1, 2, 0)
    return img.astype(np.uint8)


def save_tensor_image(tensor, path):
    makedirs(Path(path).parent)
    Image.fromarray(tensor2img(tensor)).save(path)


def LPIPS_vgg(x, y, normalize=True):
    try:
        import lpips
    except ImportError as exc:
        raise ImportError("LPIPS loss requires `pip install lpips`.") from exc
    if not hasattr(LPIPS_vgg, "_model"):
        LPIPS_vgg._model = lpips.LPIPS(net="vgg").to(x.device).eval()
    return LPIPS_vgg._model(x, y, normalize=normalize)


def LPIPS(x, y, normalize=True):
    try:
        import lpips
    except ImportError as exc:
        raise ImportError("LPIPS metric requires `pip install lpips`.") from exc
    if not hasattr(LPIPS, "_model"):
        LPIPS._model = lpips.LPIPS(net="alex").to(x.device).eval()
    return LPIPS._model(x, y, normalize=normalize)


def MSSSIM(x, y):
    try:
        from loss.distortion import MS_SSIM
    except ImportError as exc:
        raise ImportError("MS-SSIM metric needs the bundled loss.distortion module.") from exc
    if not hasattr(MSSSIM, "_model"):
        MSSSIM._model = MS_SSIM(data_range=1.0, levels=4, channel=3).to(x.device).eval()
    return MSSSIM._model(x, y)


def q_function(x):
    return 0.5 * special.erfc(x / np.sqrt(2))


def qam_ber_dB(snr_dB, M):
    snr_linear = 10 ** (snr_dB / 10)
    k = math.log2(M)
    return (4 / k) * (1 - 1 / math.sqrt(M)) * q_function(math.sqrt(3 * snr_linear / (M - 1)))


def gray_encode(input_tensor):
    input_tensor = input_tensor.to(torch.int64)
    return input_tensor ^ (input_tensor >> 1)


def gray_decode(gray_value):
    binary = gray_value.clone()
    shift = 1
    while shift < 64:
        binary = binary ^ (binary >> shift)
        shift <<= 1
    return binary
