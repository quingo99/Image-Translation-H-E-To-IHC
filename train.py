"""Phase 2 training script: Pix2Pix + Gaussian pyramid, with optional L_expr.

Usage:
    python train.py --config configs/base.yaml
    python train.py --config configs/expr.yaml
"""

import argparse
import math
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.bci_dataset import BCIDataset
from src.models.pix2pix import Generator, PatchDiscriminator
from src.losses.pyramid import PyramidL1Loss
from src.losses.expression import ExpressionLoss
from src.utils.repro import seed_everything, save_run_metadata, get_next_run_dir
from src.metrics.metrics import (
    compute_psnr,
    compute_ssim,
    compute_lpips,
    compute_dab_metrics,
    load_lpips_model,
)

import cv2

def debug_pairs(dataset, k=4, out="pair_debug.png"):
    fig, axes = plt.subplots(k, 2, figsize=(6, 3*k))
    for i in range(k):
        fname = dataset.filenames[i]
        he_path = dataset.he_dir / fname
        ihc_path = dataset.ihc_dir / fname

        he = cv2.cvtColor(cv2.imread(str(he_path)), cv2.COLOR_BGR2RGB)
        ihc = cv2.cvtColor(cv2.imread(str(ihc_path)), cv2.COLOR_BGR2RGB)

        axes[i,0].imshow(he); axes[i,0].set_title(f"HE {fname}"); axes[i,0].axis("off")
        axes[i,1].imshow(ihc); axes[i,1].set_title(f"IHC {fname}"); axes[i,1].axis("off")

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print("saved", out)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def save_sample_images(he, real_ihc, fake_ihc, path, n=4):
    """Save side-by-side comparison grid."""
    n = min(n, he.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[None, :]
    for i in range(n):
        for j, (img, title) in enumerate([
            (he[i], "H&E Input"),
            (real_ihc[i], "Real IHC"),
            (fake_ihc[i], "Generated IHC"),
        ]):
            arr = ((img.cpu().permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
            axes[i, j].imshow(arr)
            axes[i, j].set_title(title)
            axes[i, j].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _to_finite_float(value):
    """Convert value to a finite float, otherwise return None."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _is_better(metric_name, current, best):
    """Compare metric values with metric-specific direction."""
    if metric_name in {"psnr", "ssim", "dab_pearson_r", "expr"}:
        return current > best
    if metric_name in {"lpips", "iod_rel_err", "miod_rel_err", "pyr_loss"}:
        return current < best
    raise ValueError(f"Unsupported metric for best-checkpoint selection: {metric_name}")


def _compute_expr_score(dab_r, miod_rel_err, eps=1e-8):
    """Higher is better: favors high DAB correlation and low mIOD relative error."""
    return float(dab_r) / max(float(miod_rel_err), eps)


def _tile_batch_2x2(x, y, tile_size):
    """Create deterministic 2x2 corner tiles for each image in a batch."""
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape for tiled validation.")
    if x.dim() != 4:
        raise ValueError("Expected x and y to be 4D tensors: (B, C, H, W).")

    _, _, h, w = x.shape
    if h < tile_size or w < tile_size:
        raise ValueError(
            f"val_tile_size={tile_size} exceeds validation image size ({h}, {w})."
        )

    y_starts = [0, h - tile_size]
    x_starts = [0, w - tile_size]
    x_tiles, y_tiles = [], []
    for y0 in y_starts:
        for x0 in x_starts:
            x_tiles.append(x[:, :, y0 : y0 + tile_size, x0 : x0 + tile_size])
            y_tiles.append(y[:, :, y0 : y0 + tile_size, x0 : x0 + tile_size])

    return torch.cat(x_tiles, dim=0), torch.cat(y_tiles, dim=0)


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg["seed"])

    # ── Run directory ──
    run_dir = get_next_run_dir(cfg["training"]["checkpoint_dir"])
    sample_dir = os.path.join(run_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    save_run_metadata(run_dir, cfg)
    print(f"Run directory: {run_dir}")

    # ── Data ──
    # Use dedicated train/val splits (val has paired ground truth)
    data_cfg = cfg["data"]
    train_crops_per_image = int(data_cfg.get("train_crops_per_image", 1))
    train_crop_mode = str(data_cfg.get("train_crop_mode", "random")).lower()
    val_full_resolution = bool(data_cfg.get("val_full_resolution", False))
    val_tiling_mode = str(cfg["training"].get("val_tiling_mode", "none")).lower()
    val_tile_size = int(cfg["training"].get("val_tile_size", 512))
    val_sample_n = int(cfg["training"].get("val_sample_n", 4))
    val_batch_size_default = 1 if val_full_resolution else cfg["training"]["batch_size"]
    val_batch_size = int(cfg["training"].get("val_batch_size", val_batch_size_default))
    if val_batch_size < 1:
        raise ValueError("training.val_batch_size must be >= 1")
    if val_tiling_mode not in {"none", "2x2"}:
        raise ValueError("training.val_tiling_mode must be one of {'none', '2x2'}.")
    if val_tile_size < 1:
        raise ValueError("training.val_tile_size must be >= 1")
    if val_sample_n < 1:
        raise ValueError("training.val_sample_n must be >= 1")
    if val_tiling_mode != "none" and not val_full_resolution:
        raise ValueError(
            "Set data.val_full_resolution=true when using tiled validation."
        )

    train_ds = BCIDataset(
        data_cfg["root_dir"],
        split="train",
        image_size=data_cfg["image_size"],
        train_crops_per_image=train_crops_per_image,
        train_crop_mode=train_crop_mode,
    )
    #debug_pairs(train_ds, k=8)
    #return
    val_ds = BCIDataset(
        data_cfg["root_dir"],
        split="val",
        image_size=data_cfg["image_size"],
        use_full_resolution=val_full_resolution,
    )
    # debug_pairs(train_ds, k=8)
    # return

    print(
        f"Train crops/image: {train_crops_per_image} "
        f"(epoch samples={len(train_ds)})"
    )
    print(f"Train crop mode: {train_crop_mode}")
    if val_full_resolution:
        print(f"Validation transform: full resolution (batch_size={val_batch_size})")
    else:
        print(
            "Validation transform: center crop "
            f"{data_cfg['image_size']}x{data_cfg['image_size']} "
            f"(batch_size={val_batch_size})"
        )
    if val_tiling_mode == "2x2":
        print(f"Validation tiling: 2x2 deterministic tiles ({val_tile_size}x{val_tile_size})")
    else:
        print("Validation tiling: none")
    print(f"Validation sample rows: {val_sample_n}")


    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )

    # ── Models ──
    model_norm = cfg["model"].get("norm", "instance")
    G = Generator(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        num_res_blocks=cfg["model"]["num_res_blocks"],
        norm_type=model_norm,
    ).to(device)
    D = PatchDiscriminator(
        in_channels=cfg["model"]["in_channels"] + cfg["model"]["out_channels"],
        norm_type=model_norm,
    ).to(device)

    # ── Losses ──
    criterion_GAN = nn.MSELoss()
    criterion_pyr = PyramidL1Loss(levels=cfg["training"]["pyramid_levels"]).to(device)
    expr_enabled = cfg["expression"]["enabled"]
    if expr_enabled:
        criterion_expr = ExpressionLoss(
            od_threshold=cfg["expression"]["od_threshold"],
            dilate_radius=cfg["expression"]["dilate_radius"],
            min_iod_signal=float(cfg["expression"].get("min_iod_signal", 0.0)),
            min_miod_signal=float(cfg["expression"].get("min_miod_signal", 0.0)),
            max_rel_term=float(cfg["expression"].get("max_rel_term", 10.0)),
            dab_nonnegative=cfg["expression"].get("dab_nonnegative", "clamp"),
            softplus_beta=float(cfg["expression"].get("softplus_beta", 10.0)),
            stain_reference_mode=cfg["expression"].get("stain_reference_mode", "none"),
            stain_ref_blend=float(cfg["expression"].get("stain_ref_blend", 0.0)),
            stain_min_ref_images=int(cfg["expression"].get("stain_min_ref_images", 4)),
        ).to(device)
    lpips_model = load_lpips_model(device)
    if lpips_model is None:
        print("WARNING: `lpips` is not installed. Validation LPIPS will be NaN.")

    # ── Optimizers ──
    lr_g = cfg["training"].get("lr_G", cfg["training"].get("lr"))
    lr_d = cfg["training"].get("lr_D", cfg["training"].get("lr"))
    if lr_g is None or lr_d is None:
        raise ValueError("Set training.lr_G and training.lr_D (or legacy training.lr).")

    opt_G = torch.optim.Adam(
        G.parameters(),
        lr=lr_g,
        betas=(cfg["training"]["beta1"], cfg["training"]["beta2"]),
    )
    opt_D = torch.optim.Adam(
        D.parameters(),
        lr=lr_d,
        betas=(cfg["training"]["beta1"], cfg["training"]["beta2"]),
    )
    sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_G, T_max=cfg["training"]["epochs"]
    )
    sched_D = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_D, T_max=cfg["training"]["epochs"]
    )

    # ── Resume ──
    start_epoch = 0
    if cfg["training"]["resume"]:
        ckpt = torch.load(cfg["training"]["resume"], map_location=device)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])
        if "sched_G" in ckpt:
            sched_G.load_state_dict(ckpt["sched_G"])
        if "sched_D" in ckpt:
            sched_D.load_state_dict(ckpt["sched_D"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    # ── Training loop ──
    epochs = cfg["training"]["epochs"]
    lambda_adv = float(cfg["training"].get("lambda_adv", 1.0))
    lambda_pyr = cfg["training"]["lambda_pyr"]
    lambda_expr = float(cfg["expression"]["lambda_expr"])
    epoch_add_expr = int(cfg["expression"]["epoch_add_expr"])
    expr_warmup_epochs = int(cfg["expression"].get("warmup_epochs", 0))
    if expr_warmup_epochs < 0:
        raise ValueError("expression.warmup_epochs must be >= 0")
    d_update_every = int(cfg["training"].get("d_update_every", 1))
    real_label = float(cfg["training"].get("gan_label_real", 1.0))
    fake_label = float(cfg["training"].get("gan_label_fake", 0.0))
    gen_label = float(cfg["training"].get("gan_label_gen", 1.0))
    if d_update_every < 1:
        raise ValueError("training.d_update_every must be >= 1")
    val_schedule = str(cfg["training"].get("val_schedule", "fixed")).lower()
    val_start_epoch = int(cfg["training"].get("val_start_epoch", 1))
    val_every = int(cfg["training"].get("val_every", 5))
    val_cooldown_epochs = int(cfg["training"].get("val_cooldown_epochs", val_every))
    val_loss_min_delta = float(cfg["training"].get("val_loss_min_delta", 0.0))
    val_reset_global_best_after_epochs = int(
        cfg["training"].get("val_reset_global_best_after_epochs", 15)
    )
    if val_schedule not in {"fixed", "on_train_loss"}:
        raise ValueError(
            "training.val_schedule must be one of {'fixed', 'on_train_loss'}."
        )
    if val_start_epoch < 1:
        raise ValueError("training.val_start_epoch must be >= 1")
    if val_every < 1:
        raise ValueError("training.val_every must be >= 1")
    if val_cooldown_epochs < 1:
        raise ValueError("training.val_cooldown_epochs must be >= 1")
    if val_loss_min_delta < 0:
        raise ValueError("training.val_loss_min_delta must be >= 0")
    if val_reset_global_best_after_epochs < 1:
        raise ValueError(
            "training.val_reset_global_best_after_epochs must be >= 1"
        )
    valid_best_metrics = {
        "psnr",
        "ssim",
        "lpips",
        "dab_pearson_r",
        "iod_rel_err",
        "miod_rel_err",
        "pyr_loss",
        "expr",
    }
    best_metric = str(cfg["training"].get("best_metric", "psnr")).lower()
    if best_metric not in valid_best_metrics:
        raise ValueError(
            f"training.best_metric must be one of {sorted(valid_best_metrics)}. "
            f"Got: {best_metric}"
        )
    save_extra_best = bool(cfg["training"].get("save_extra_best_checkpoints", True))

    if best_metric in {"psnr", "ssim", "dab_pearson_r", "expr"}:
        best_primary_value = -float("inf")
    else:
        best_primary_value = float("inf")
    best_primary_epoch = 0
    last_val_epoch = 0
    best_train_loss_seen = float("inf")
    best_extra_values = {
        "psnr": -float("inf"),
        "lpips": float("inf"),
        "expr": -float("inf"),
    }
    best_extra_epochs = {k: 0 for k in best_extra_values}

    log_file = open(os.path.join(run_dir, "train_log.csv"), "w")
    log_file.write(
        "epoch,loss_D,loss_adv,loss_pyr,loss_expr,loss_G,"
        "val_psnr,val_ssim,val_lpips,val_dab_pearson_r,val_iod_rel_err,val_miod_rel_err,val_pyr_loss,"
        "iod_real,iod_gen,miod_real,miod_gen\n"
    )

    if val_schedule == "on_train_loss":
        print(
            "Validation schedule: on_train_loss "
            f"(start_epoch={val_start_epoch}, cooldown={val_cooldown_epochs}, "
            f"min_delta={val_loss_min_delta}, "
            f"reset_after={val_reset_global_best_after_epochs})"
        )
    else:
        print(
            "Validation schedule: fixed "
            f"(start_epoch={val_start_epoch}, every={val_every})"
        )

    for epoch in range(start_epoch, epochs):
        G.train()
        D.train()

        epoch_stats = {
            "loss_D": 0, "loss_adv": 0, "loss_pyr": 0,
            "loss_expr": 0, "loss_G": 0, "n_batches": 0, "n_d_steps": 0,
            "iod_real": 0, "iod_gen": 0, "miod_real": 0, "miod_gen": 0,
        }

        use_expr = expr_enabled and epoch >= epoch_add_expr

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        last_loss_d = 0.0
       
        for step, batch in enumerate(pbar):
            x = batch["he"].to(device)
            y = batch["ihc"].to(device)
            y_hat = G(x) 
            if step % d_update_every == 0:
                for p in D.parameters():
                    p.requires_grad_(True)
                real_pred = D(x, y)
                fake_pred = D(x, y_hat.detach())            
                loss_D_real = criterion_GAN(real_pred, torch.full_like(real_pred, real_label))
                loss_D_fake = criterion_GAN(fake_pred, torch.full_like(fake_pred, fake_label))
                loss_D = (loss_D_real + loss_D_fake) * 0.5

                opt_D.zero_grad(set_to_none=True)
                loss_D.backward()
                opt_D.step()

                last_loss_d = loss_D.item()
                epoch_stats["loss_D"] += last_loss_d
                epoch_stats["n_d_steps"] += 1


            for p in D.parameters():
                p.requires_grad_(False)
            
            fake_pred_G = D(x, y_hat)

            loss_adv = criterion_GAN(fake_pred_G, torch.full_like(fake_pred_G, gen_label))
            
            loss_pyr = criterion_pyr(y, y_hat)
            
            loss_G = lambda_adv * loss_adv + lambda_pyr * loss_pyr
            

            loss_expr_val = torch.tensor(0.0, device=device)
            expr_weight = 0.0
            expr_info = {"iod_real": 0, "iod_gen": 0, "miod_real": 0, "miod_gen": 0}
            # print("test y:")
            # print(y)
            # print("test y^")
            # print(y_hat)
            if use_expr:
                loss_expr_val, expr_info = criterion_expr(y, y_hat)
                if expr_warmup_epochs > 0:
                    warmup_progress = min(
                        1.0, (epoch - epoch_add_expr + 1) / expr_warmup_epochs
                    )
                    expr_weight = lambda_expr * warmup_progress
                else:
                    expr_weight = lambda_expr
                #total_loss with expression
                loss_G = loss_G + expr_weight * loss_expr_val
            
            opt_G.zero_grad(set_to_none=True)
            loss_G.backward()
            opt_G.step()

            # Accumulate
            epoch_stats["loss_adv"] += loss_adv.item()
            epoch_stats["loss_pyr"] += loss_pyr.item()
            epoch_stats["loss_expr"] += loss_expr_val.item()
            epoch_stats["loss_G"] += loss_G.item()
            epoch_stats["iod_real"] += expr_info["iod_real"]
            epoch_stats["iod_gen"] += expr_info["iod_gen"]
            epoch_stats["miod_real"] += expr_info["miod_real"]
            epoch_stats["miod_gen"] += expr_info["miod_gen"]
            epoch_stats["n_batches"] += 1

            pbar.set_postfix({
                "D": f"{last_loss_d:.4f}",
                "G": f"{loss_G.item():.4f}",
                "pyr": f"{loss_pyr.item():.4f}",
                "expr": f"{loss_expr_val.item():.4f}" if use_expr else "off",
                "expr_w": f"{expr_weight:.3f}" if use_expr else "0.000",
            })
        sched_G.step()
        sched_D.step()

        nb = max(epoch_stats["n_batches"], 1)
        nd = max(epoch_stats["n_d_steps"], 1)
        avg = {}
        for k, v in epoch_stats.items():
            if k in {"n_batches", "n_d_steps"}:
                continue
            avg[k] = v / (nd if k == "loss_D" else nb)

        epoch_num = epoch + 1
        if val_schedule == "on_train_loss":
            if last_val_epoch > 0:
                epochs_since_last_val = epoch_num - last_val_epoch
                if epochs_since_last_val > val_reset_global_best_after_epochs:
                    print(
                        "  Resetting train-loss global_best "
                        f"(no validation for {epochs_since_last_val} epochs)."
                    )
                    best_train_loss_seen = float("inf")
            train_loss_epoch = float(avg["loss_G"])
            loss_improved = train_loss_epoch + val_loss_min_delta < best_train_loss_seen
            if loss_improved:
                best_train_loss_seen = train_loss_epoch
            cooldown_ready = (
                last_val_epoch == 0
                or (epoch_num - last_val_epoch) >= val_cooldown_epochs
            )
            should_validate = (
                epoch_num >= val_start_epoch
                and loss_improved
                and cooldown_ready
            )
        else:
            should_validate = (
                epoch_num >= val_start_epoch
                and ((epoch_num - val_start_epoch) % val_every == 0)
            )

        # ── Validation ──
        val_psnr, val_ssim = 0.0, 0.0
        val_lpips = float("nan")
        val_dab_pearson_r = 0.0
        val_iod_rel_err = 0.0
        val_miod_rel_err = 0.0
        val_pyr_loss = 0.0
        if should_validate:
            G.eval()
            psnr_vals, ssim_vals, lpips_vals = [], [], []
            dab_pearson_vals, iod_rel_vals, miod_rel_vals = [], [], []
            pyr_vals = []
            sample_saved = False
            sample_count = 0
            sample_x_parts, sample_y_parts, sample_yhat_parts = [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["he"].to(device)
                    y = batch["ihc"].to(device)
                    if val_tiling_mode == "2x2":
                        x_val, y_val = _tile_batch_2x2(x, y, val_tile_size)
                    else:
                        x_val, y_val = x, y
                    y_hat = G(x_val)
                    psnr_vals.extend(compute_psnr(y_val, y_hat))
                    ssim_vals.extend(compute_ssim(y_val, y_hat))
                    lpips_vals.extend(compute_lpips(y_val, y_hat, lpips_model))
                    pyr_vals.append(criterion_pyr(y_val, y_hat).item())
                    dab_batch = compute_dab_metrics(y_val, y_hat)
                    for m in dab_batch:
                        dab_pearson_vals.append(m["dab_pearson_r"])
                        iod_rel_vals.append(m["iod_rel_err"])
                        miod_rel_vals.append(m["miod_rel_err"])
                    if not sample_saved:
                        remaining = val_sample_n - sample_count
                        if remaining > 0:
                            take = min(remaining, x_val.shape[0])
                            sample_x_parts.append(x_val[:take].detach())
                            sample_y_parts.append(y_val[:take].detach())
                            sample_yhat_parts.append(y_hat[:take].detach())
                            sample_count += take
                        if sample_count >= val_sample_n:
                            save_sample_images(
                                torch.cat(sample_x_parts, dim=0),
                                torch.cat(sample_y_parts, dim=0),
                                torch.cat(sample_yhat_parts, dim=0),
                                os.path.join(sample_dir, f"epoch_{epoch+1:03d}.png"),
                                n=val_sample_n,
                            )
                            sample_saved = True

            if not sample_saved and sample_count > 0:
                save_sample_images(
                    torch.cat(sample_x_parts, dim=0),
                    torch.cat(sample_y_parts, dim=0),
                    torch.cat(sample_yhat_parts, dim=0),
                    os.path.join(sample_dir, f"epoch_{epoch+1:03d}.png"),
                    n=val_sample_n,
                )

            val_psnr = sum(psnr_vals) / len(psnr_vals)
            val_ssim = sum(ssim_vals) / len(ssim_vals)
            val_lpips = sum(lpips_vals) / len(lpips_vals) if lpips_vals else float("nan")
            val_dab_pearson_r = (
                sum(dab_pearson_vals) / len(dab_pearson_vals) if dab_pearson_vals else 0.0
            )
            val_iod_rel_err = sum(iod_rel_vals) / len(iod_rel_vals) if iod_rel_vals else 0.0
            val_miod_rel_err = (
                sum(miod_rel_vals) / len(miod_rel_vals) if miod_rel_vals else 0.0
            )
            val_pyr_loss = sum(pyr_vals) / len(pyr_vals) if pyr_vals else 0.0
            print(
                f"  Val PSNR: {val_psnr:.2f}  SSIM: {val_ssim:.4f}  LPIPS: {val_lpips:.4f}"
            )
            print(
                f"  Val DAB-r: {val_dab_pearson_r:.4f}  IOD-rel: {val_iod_rel_err:.4f}"
                f"  mIOD-rel: {val_miod_rel_err:.4f}  Pyr: {val_pyr_loss:.4f}"
            )

            expr_score = _compute_expr_score(val_dab_pearson_r, val_miod_rel_err)
            metric_values = {
                "psnr": val_psnr,
                "ssim": val_ssim,
                "lpips": val_lpips,
                "dab_pearson_r": val_dab_pearson_r,
                "iod_rel_err": val_iod_rel_err,
                "miod_rel_err": val_miod_rel_err,
                "pyr_loss": val_pyr_loss,
                "expr": expr_score,
            }

            current_primary = _to_finite_float(metric_values[best_metric])
            if current_primary is not None and _is_better(
                best_metric, current_primary, best_primary_value
            ):
                best_primary_value = current_primary
                best_primary_epoch = epoch + 1
                torch.save(G.state_dict(), os.path.join(run_dir, "generator_best.pth"))
                print(
                    f"  Saved generator_best.pth "
                    f"(best {best_metric}={current_primary:.6f} at epoch {epoch+1})"
                )

            if save_extra_best:
                extra_paths = {
                    "psnr": "generator_best_psnr.pth",
                    "lpips": "generator_best_lpips.pth",
                    "expr": "generator_best_expr.pth",
                }
                for metric_name, filename in extra_paths.items():
                    if metric_name == best_metric:
                        continue
                    current_value = _to_finite_float(metric_values[metric_name])
                    if current_value is None:
                        continue
                    if _is_better(metric_name, current_value, best_extra_values[metric_name]):
                        best_extra_values[metric_name] = current_value
                        best_extra_epochs[metric_name] = epoch + 1
                        torch.save(G.state_dict(), os.path.join(run_dir, filename))

            last_val_epoch = epoch_num
        # Log
        log_file.write(
            f"{epoch+1},{avg['loss_D']:.6f},{avg['loss_adv']:.6f},"
            f"{avg['loss_pyr']:.6f},{avg['loss_expr']:.6f},{avg['loss_G']:.6f},"
            f"{val_psnr:.4f},{val_ssim:.6f},{val_lpips:.6f},{val_dab_pearson_r:.6f},"
            f"{val_iod_rel_err:.6f},{val_miod_rel_err:.6f},{val_pyr_loss:.6f},"
            f"{avg['iod_real']:.4f},{avg['iod_gen']:.4f},"
            f"{avg['miod_real']:.6f},{avg['miod_gen']:.6f}\n"
        )
        log_file.flush()

        # ── Checkpoints ──
        if (epoch + 1) % cfg["training"]["save_every"] == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "G": G.state_dict(),
                    "D": D.state_dict(),
                    "opt_G": opt_G.state_dict(),
                    "opt_D": opt_D.state_dict(),
                    "sched_G": sched_G.state_dict(),
                    "sched_D": sched_D.state_dict(),
                },
                os.path.join(run_dir, f"checkpoint_epoch_{epoch+1:03d}.pth"),
            )

    # Save final
    torch.save(G.state_dict(), os.path.join(run_dir, "generator_final.pth"))
    log_file.close()
    if best_primary_epoch > 0:
        print(
            f"Training complete. Best {best_metric}: "
            f"{best_primary_value:.6f} at epoch {best_primary_epoch}"
        )
    else:
        print(
            f"Training complete. No valid value observed for "
            f"training.best_metric='{best_metric}'."
        )
    if save_extra_best:
        for metric_name in ("psnr", "lpips", "expr"):
            if metric_name == best_metric:
                continue
            if best_extra_epochs[metric_name] > 0:
                print(
                    f"  Best {metric_name}: {best_extra_values[metric_name]:.6f} "
                    f"at epoch {best_extra_epochs[metric_name]}"
                )
    print(f"Outputs saved to: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)

