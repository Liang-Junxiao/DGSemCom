from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import torch


ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    # Run mode: train_codec, train_prior, or test.
    phase: str = "test"
    name: str = "dgsemcom_4x256"
    output_dir: str = str(ROOT / "runs")

    # Data. Put images in these folders or override from CLI.
    train_data_dir: List[str] = field(default_factory=lambda: [str(ROOT / "examples/train")])
    test_data_dir: List[str] = field(default_factory=lambda: [str(ROOT / "examples/test")])
    visual_data_dir: List[str] = field(default_factory=lambda: [str(ROOT / "examples/visual")])

    # Default released model setting used in the paper experiments.
    embed_n: int = 256
    chunk_num: int = 4
    embed_dim: int = 256
    share_codebook: bool = False
    share_emb: bool = False
    norm_codebook: bool = True
    pretrained: bool = False

    # Checkpoints. Copy released weights here, or override from CLI.
    codec_checkpoint: str = str(ROOT / "checkpoints" / "pqgan_codec_4x256_norm_noshare.model")
    prior_checkpoint: str = str(ROOT / "checkpoints" / "dgsemcom_prior_4x256_norm_noshare.model")

    # Training.
    batch_size: int = 4
    num_workers: int = 0
    epochs: int = 1
    lr_codec: float = 1e-4
    lr_prior: float = 1e-5
    print_step: int = 50
    save_every: int = 1
    seed: int = 0
    gpu_id: str = "0"
    cuda: bool = True
    save_log: bool = True
    save_images: bool = True
    holdon: bool = False

    # Codec losses.
    distortion_metric: str = "MSE"
    latent_loss_weight: float = 0.25
    use_gan: bool = False
    disc_factor: float = 0.0

    # Image and latent shape.
    image_dims: Tuple[int, int, int] = (3, 256, 256)
    ga_kwargs: dict = field(default_factory=lambda: {"embed_dims": [256, 256, 256, 256]})

    # Prior model.
    num_timesteps: int = 100
    num_inference: int = 100
    truncation_rate: float = 0.7
    delta_t: int = 1
    num_attention_heads: int = 16
    attention_head_dim: int = 88
    num_layers: int = 10
    dropout: float = 0.0
    norm_num_groups: int = 32
    cross_attention_dim: Optional[int] = None
    attention_bias: bool = True
    activation_fn: str = "geglu"
    num_embeds_ada_norm: int = 100
    auxiliary_loss_weight: float = 0.001
    adaptive_auxiliary_loss: bool = True

    # Communication test.
    channel_type: str = "bsc"  # bsc or awgn
    posterior_mode: str = "both"  # hard, soft, or both. BSC always uses hard.
    ber: float = 0.05
    snr: float = 4.0
    snr_offset: float = 0.0
    channel_kwargs: dict = field(
        default_factory=lambda: {"modulation": "4QAM", "SNR": 6.0}
    )

    # Runtime fields.
    device: torch.device | str = "cuda"
    logger: object = None

    def finalize(self):
        self.bit_num = int(math.log2(self.embed_n))
        self.latent_size = (self.image_dims[1] // 16, self.image_dims[2] // 16, self.chunk_num)
        self.sample_size = self.latent_size[0]
        self.num_codebook = self.chunk_num
        self.num_vector_embeds = self.embed_n + 1
        self.checkpoint = self.codec_checkpoint
        self.checkpoint_diff = self.prior_checkpoint
        self.channel_kwargs["SNR"] = self.snr
        return self


def get_config() -> Config:
    return Config().finalize()
