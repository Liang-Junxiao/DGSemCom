from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from config import get_config
from dataloader.datasets import get_loader, get_test_loader
from DGSemCom import DGSemCom
from PQ_net import PQNet
from utils import (
    AverageMeter,
    CalcuPSNR_int,
    LPIPS,
    load_weights,
    logger_configuration,
    qam_ber_dB,
    save_model,
    save_tensor_image,
)


def parse_args():
    parser = argparse.ArgumentParser("DGSemCom open-source entry")
    parser.add_argument("--phase", choices=["train_codec", "train_prior", "test"], default=None)
    parser.add_argument("--train-data", nargs="*", default=None)
    parser.add_argument("--test-data", nargs="*", default=None)
    parser.add_argument("--codec-checkpoint", default=None)
    parser.add_argument("--prior-checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr-codec", type=float, default=None)
    parser.add_argument("--lr-prior", type=float, default=None)
    parser.add_argument("--ber", type=float, default=None)
    parser.add_argument("--snr", type=float, default=None)
    parser.add_argument("--channel-type", choices=["bsc", "awgn"], default=None)
    parser.add_argument("--posterior-mode", choices=["hard", "soft", "both"], default=None)
    parser.add_argument("--num-inference", type=int, default=None)
    parser.add_argument("--truncation-rate", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-test-images", type=int, default=None)
    parser.add_argument("--save-images", dest="save_images", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def apply_overrides(config, args):
    for attr, value in {
        "phase": args.phase,
        "codec_checkpoint": args.codec_checkpoint,
        "prior_checkpoint": args.prior_checkpoint,
        "output_dir": args.output_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr_codec": args.lr_codec,
        "lr_prior": args.lr_prior,
        "save_images": args.save_images,
        "ber": args.ber,
        "snr": args.snr,
        "channel_type": args.channel_type,
        "posterior_mode": args.posterior_mode,
        "num_inference": args.num_inference,
        "truncation_rate": args.truncation_rate,
        "seed": args.seed,
    }.items():
        if value is not None:
            setattr(config, attr, value)
    if args.train_data is not None and len(args.train_data) > 0:
        config.train_data_dir = args.train_data
    if args.test_data is not None and len(args.test_data) > 0:
        config.test_data_dir = args.test_data
        config.visual_data_dir = args.test_data
    if args.device is not None:
        config.device = args.device
        config.cuda = str(args.device).startswith("cuda")
    return config.finalize()


def setup(config):
    if config.seed is not None:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    if isinstance(config.device, str):
        if config.device == "cuda" and not torch.cuda.is_available():
            config.device = "cpu"
    config.device = torch.device(config.device)
    workdir, logger = logger_configuration(config.name, config.phase, config.save_log, root=config.output_dir)
    config.logger = logger
    logger.info(config)
    return Path(workdir), logger


def train_codec(config, workdir, logger):
    train_loader, test_loader = get_loader(config)
    net = PQNet(config).to(config.device)
    opt_g, opt_d = net.configure_optimizers(config)

    best_loss = float("inf")
    global_step = 0
    for epoch in range(config.epochs):
        net.train()
        losses, psnrs = AverageMeter(), AverageMeter()
        for image in train_loader:
            image = image.to(config.device)
            opt_g.zero_grad()
            opt_d.zero_grad()

            out = net(image, epoch=epoch)
            out["loss"].backward(retain_graph=config.use_gan)
            if config.use_gan:
                out["gan_loss"].backward()
                opt_d.step()
            opt_g.step()

            global_step += 1
            losses.update(out["loss"])
            psnrs.update(np.mean(CalcuPSNR_int(out["dec"], image)))
            if global_step % config.print_step == 0:
                logger.info(f"codec epoch={epoch} step={global_step} loss={losses.avg:.4f} psnr={psnrs.avg:.4f}")
                losses.clear()
                psnrs.clear()

        val_loss, val_psnr = evaluate_codec(config, net, test_loader)
        logger.info(f"codec validate epoch={epoch} loss={val_loss:.4f} psnr={val_psnr:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            save_model(net, workdir / "models" / "codec_best.model")
        if (epoch + 1) % config.save_every == 0:
            save_model(net, workdir / "models" / f"codec_ep{epoch + 1}.model")


@torch.no_grad()
def evaluate_codec(config, net, loader):
    net.eval()
    losses, psnrs = AverageMeter(), AverageMeter()
    for image in loader:
        image = image.to(config.device)
        out = net(image)
        losses.update(out["loss"])
        psnrs.update(np.mean(CalcuPSNR_int(out["dec"], image)))
    return losses.avg, psnrs.avg


def train_prior(config, workdir, logger):
    if not Path(config.codec_checkpoint).exists():
        raise FileNotFoundError(f"Codec checkpoint not found: {config.codec_checkpoint}")
    train_loader, _ = get_loader(config)
    net = DGSemCom(config).to(config.device)
    net.load_vq_model(config.codec_checkpoint)
    optimizer = optim.Adam(net.parameters(), lr=config.lr_prior)

    global_step = 0
    for epoch in range(config.epochs):
        net.train()
        total_losses, lb_losses, aux_losses = AverageMeter(), AverageMeter(), AverageMeter()
        for image in train_loader:
            image = image.to(config.device)
            optimizer.zero_grad()
            out = net(image)
            out["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

            global_step += 1
            total_losses.update(out["total_loss"])
            lb_losses.update(out["lb_loss"])
            aux_losses.update(out["aux_loss"])
            if global_step % config.print_step == 0:
                logger.info(
                    f"prior epoch={epoch} step={global_step} total={total_losses.avg:.4f} "
                    f"lb={lb_losses.avg:.4f} aux={aux_losses.avg:.4f}"
                )
                total_losses.clear()
                lb_losses.clear()
                aux_losses.clear()
        if (epoch + 1) % config.save_every == 0:
            save_model(net, workdir / "models" / f"prior_ep{epoch + 1}.model")


@torch.no_grad()
def test(config, workdir, logger, max_images=None, save_images=None):
    if save_images is None:
        save_images = config.save_images
    if not Path(config.prior_checkpoint).exists():
        raise FileNotFoundError(f"Prior checkpoint not found: {config.prior_checkpoint}")
    loader = get_test_loader(config)
    net = DGSemCom(config).to(config.device)
    load_weights(net, config.prior_checkpoint, strict=False, map_location=config.device)
    net.eval()

    direct_psnr, hard_psnr, soft_psnr = AverageMeter(), AverageMeter(), AverageMeter()
    direct_lpips, hard_lpips, soft_lpips = AverageMeter(), AverageMeter(), AverageMeter()
    bers = AverageMeter()

    for idx, image in enumerate(loader):
        if max_images is not None and idx >= max_images:
            break
        image = image.to(config.device)
        _, _, latent, _, _ = net.PQNet.encode(image)
        bsz, h, w, c = latent.shape
        latent_flat = latent.reshape(bsz, -1)

        if config.channel_type == "bsc":
            received = net.channel_trans.BSC_bit_flip(latent_flat, config.ber)
            net.channel_trans.calculate_transition_matrix(config.ber)
            soft_prob = None
            ber = config.ber
            hard_ber = config.ber
            run_hard = True
            run_soft = False
        else:
            channel = net.convention_channel.integer_trans_llr(
                latent_flat, snr=config.snr, chan_type="AWGN", snr_offset=config.snr_offset
            )
            received = channel["hard_out"]
            soft_prob = channel["prob"]
            ber = channel["ber"]
            hard_ber = qam_ber_dB(config.snr, M=4)
            net.channel_trans.calculate_transition_matrix(hard_ber)
            run_hard = config.posterior_mode in {"hard", "both"}
            run_soft = config.posterior_mode in {"soft", "both"}

        direct = net.PQNet.decode_code(received.reshape(bsz, h, w, c))
        hard_corrected = None
        soft_corrected = None
        if run_hard:
            hard_corrected = net.posterior_sample_star(
                num_inference_steps=config.num_inference,
                received_signal=received,
                truncation_rate=config.truncation_rate,
                prob=None,
                delta_t=config.delta_t,
                ori_signal=latent_flat,
            )
        if run_soft:
            soft_corrected = net.posterior_sample_star(
                num_inference_steps=config.num_inference,
                received_signal=received,
                truncation_rate=config.truncation_rate,
                prob=soft_prob,
                delta_t=config.delta_t,
                ori_signal=latent_flat,
            )

        direct_psnr.update(np.mean(CalcuPSNR_int(direct, image)))
        if hard_corrected is not None:
            hard_psnr.update(np.mean(CalcuPSNR_int(hard_corrected, image)))
        if soft_corrected is not None:
            soft_psnr.update(np.mean(CalcuPSNR_int(soft_corrected, image)))
        try:
            direct_lpips.update(LPIPS(direct, image, normalize=True).mean())
            if hard_corrected is not None:
                hard_lpips.update(LPIPS(hard_corrected, image, normalize=True).mean())
            if soft_corrected is not None:
                soft_lpips.update(LPIPS(soft_corrected, image, normalize=True).mean())
        except ImportError:
            pass
        bers.update(ber)

        if save_images:
            save_tensor_image(image, workdir / "samples" / f"{idx:04d}_origin.png")
            save_tensor_image(direct, workdir / "samples" / f"{idx:04d}_direct_hard_decision.png")
            if hard_corrected is not None:
                save_tensor_image(hard_corrected, workdir / "samples" / f"{idx:04d}_posterior_hard.png")
            if soft_corrected is not None:
                save_tensor_image(soft_corrected, workdir / "samples" / f"{idx:04d}_posterior_soft.png")

        logger.info(
            f"test idx={idx} measured_ber={ber:.6f} hard_ber={hard_ber:.6f} "
            f"direct_psnr={direct_psnr.val:.4f}/{direct_psnr.avg:.4f} "
            f"posterior_hard_psnr={hard_psnr.val:.4f}/{hard_psnr.avg:.4f} "
            f"posterior_soft_psnr={soft_psnr.val:.4f}/{soft_psnr.avg:.4f}"
        )

    logger.info(
        f"finish test avg_measured_ber={bers.avg:.6f} direct_psnr={direct_psnr.avg:.4f} "
        f"posterior_hard_psnr={hard_psnr.avg:.4f} posterior_soft_psnr={soft_psnr.avg:.4f} "
        f"direct_lpips={direct_lpips.avg:.4f} posterior_hard_lpips={hard_lpips.avg:.4f} "
        f"posterior_soft_lpips={soft_lpips.avg:.4f}"
    )


def main():
    args = parse_args()
    config = apply_overrides(get_config(), args)
    workdir, logger = setup(config)
    if config.phase == "train_codec":
        train_codec(config, workdir, logger)
    elif config.phase == "train_prior":
        train_prior(config, workdir, logger)
    elif config.phase == "test":
        test(config, workdir, logger, max_images=args.max_test_images, save_images=config.save_images)
    else:
        raise ValueError(f"Unknown phase: {config.phase}")


if __name__ == "__main__":
    main()
