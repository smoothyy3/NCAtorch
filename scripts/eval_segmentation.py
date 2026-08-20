"""
Dice evaluation for segmentation NCAs.

Works with any binary segmentation dataset. Datasets exposing volume_groups()
pool all slices of a volume into one score; others are scored per sample.
Both the single-backbone and the two-backbone Med-NCA rollout are supported.

The metric follows Med-NCA's reporting convention (BaseAgent.test in
MECLabTUDA/Med-NCA, src/agents/Agent.py), so numbers are comparable to Table 1
of the Med-NCA paper:

  - runs on the test split, whole image
  - sigmoid on the logit channel, no threshold
  - smooth = 0 on both numerator and denominator (Med-NCA passes smooth=0 explicitly at Agent.py:272, overriding the loss default of 1)
  - groups with no foreground in the target are skipped entirely
  - a NaN Dice (empty prediction AND empty target) counts as 0.0
  - final score is the unweighted mean over groups
"""

import os
import sys
import glob
import math
import argparse

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from nca.utils.config import load_config
from nca.core.models.model_factory import create_model
from nca.data.dataset_factory import create_dataset
from nca.training.evolve_factory import create_evolver
from nca.training.trainers.med_nca_rollout import med_nca_rollout

LOGIT_CHANNEL = 1
FREEZE_CHANNELS = 1


def _find_checkpoint(log_dir: str, name: str | None, prefix: str = "ca") -> str:
    """
    Resolve a checkpoint path, preferring <prefix>_final.pt then the latest step.
    """
    if name is not None:
        path = os.path.join(log_dir, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    final = os.path.join(log_dir, f"{prefix}_final.pt")
    if os.path.isfile(final):
        return final

    candidates = sorted(glob.glob(os.path.join(log_dir, f"{prefix}_*.pt")))
    if not candidates:
        raise FileNotFoundError(f"No {prefix}_*.pt checkpoints found in {log_dir}")

    def _step(p):
        base = os.path.splitext(os.path.basename(p))[0]
        try:
            return int(base.split("_")[-1])
        except ValueError:
            return -1

    return max(candidates, key=_step)


def _load_weights(model, ckpt_path, device):
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    if any("_orig_mod." in k for k in state_dict):
        state_dict = {
            k.replace("._orig_mod.", ".").removeprefix("_orig_mod."): v
            for k, v in state_dict.items()
        }
    model.load_state_dict(state_dict)
    return model


@torch.no_grad()
def predict_volume(rollout, dataset, indices, device, chunk_size):
    """
    Run the CA over every slice of one volume.

    Slices are processed in chunks purely to bound memory. The CA is
    per pixel anyway so chunking cannot change the result.

    Args:
        rollout: Callable taking a batch of seeds (B, C, H, W) and returning
            the final state. Single-backbone and Med-NCA differ here.

    Returns:
        (probs, targets), each (S, 1, H, W) on CPU, S = number of slices.
    """
    probs, targets = [], []

    for start in range(0, len(indices), chunk_size):
        chunk = indices[start:start + chunk_size]

        items  = [dataset[i] for i in chunk]
        seeds  = torch.stack([it[0] for it in items]).to(device)
        target = torch.stack([it[2] for it in items])

        state = rollout(seeds)

        logits = state[:, LOGIT_CHANNEL:LOGIT_CHANNEL + 1]
        probs.append(torch.sigmoid(logits).float().cpu())
        targets.append(target.float())

    return torch.cat(probs), torch.cat(targets)


def volume_dice(probs, target, smooth=0.0):
    """
    Med-NCAs Dice over a whole volume: soft, unthresholded, unsmoothed.

    Returns:
        float Dice, or 0.0 when the expression is NaN.
    """
    p = probs.flatten()
    t = target.flatten()
    dice = (2.0 * (p * t).sum() + smooth) / (p.sum() + t.sum() + smooth)
    value = dice.item()
    return 0.0 if math.isnan(value) else value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-volume Dice on the test split (Med-NCA comparable)."
    )
    parser.add_argument("--log_dir", required=True,
                        help="Training log directory holding config.yaml and checkpoints.")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint filename inside log_dir. Default: ca_final.pt, else latest. "
                             "Ignored with --dual, which resolves b1_*/b2_* itself.")
    parser.add_argument("--dual", action="store_true",
                        help="Evaluate a two-backbone Med-NCA run: load b1_*.pt and b2_*.pt "
                             "and run the full-image two-scale rollout (no patching).")
    parser.add_argument("--device", default=None, help="Override config.DEVICE.")
    parser.add_argument("--iter_n", type=int, default=64,
                        help="CA steps at inference. Default 64, matching Med-NCA.")
    parser.add_argument("--chunk_size", type=int, default=32,
                        help="Slices per forward pass. Memory only; does not affect the score.")
    parser.add_argument("--train_split", action="store_true",
                        help="Evaluate the training split instead of the test split.")
    parser.add_argument("--csv", default=None, help="Optional path to write per-patient scores.")
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = os.path.join(args.log_dir, "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.yaml not found in {args.log_dir}")
    config = load_config(config_path)

    if args.device:
        config.DEVICE = args.device
    device = config.DEVICE

    dataloader, cond_dim, im_height, im_width = create_dataset(config, train=args.train_split)
    dataset = dataloader.get_dataset()

    if cond_dim != 0:
        raise ValueError(
            f"Segmentation eval assumes an unconditioned model, got cond_dim={cond_dim}."
        )

    evolver = create_evolver(config)

    if args.dual:
        b1 = create_model(config, cond_dim, im_height, im_width)
        b2 = create_model(config, cond_dim, im_height, im_width)
        b1_path = _find_checkpoint(args.log_dir, None, prefix="b1")
        b2_path = _find_checkpoint(args.log_dir, None, prefix="b2")
        _load_weights(b1, b1_path, device)
        _load_weights(b2, b2_path, device)
        b1.to(device).eval()
        b2.to(device).eval()
        ckpt_path = f"{b1_path} + {os.path.basename(b2_path)}"

        def rollout(seeds):
            state, _ = med_nca_rollout(
                b1=b1, b2=b2, evolver=evolver, seed=seeds,
                iter_n=args.iter_n, target=None, patch_size=None,
                freeze_channels=FREEZE_CHANNELS,
            )
            return state
    else:
        model = create_model(config, cond_dim, im_height, im_width)
        ckpt_path = _find_checkpoint(args.log_dir, args.checkpoint)
        _load_weights(model, ckpt_path, device)
        model.to(device).eval()

        def rollout(seeds):
            return evolver(
                ca_model=model, state_in=seeds, conds=None,
                iter_n=args.iter_n, freeze_channels=FREEZE_CHANNELS,
            )

    # Volumetric datasets pool all slices of a patient into one score. Datasets
    # without that structure fall back to one score per sample.
    if hasattr(dataset, "volume_groups"):
        groups, unit = dataset.volume_groups(), "volume"
    else:
        groups, unit = [(str(i), [i]) for i in range(len(dataset))], "sample"

    print(f"Config: {config_path}")
    print(f"Mode: {'Med-NCA (two backbones)' if args.dual else 'single backbone'}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Device: {device}")
    print(f"Split: {'train' if args.train_split else 'test'}")
    print(f"Iterations: {args.iter_n}")
    print(f"Grouping: per {unit}")
    print(f"{unit.capitalize()}s: {len(groups)} ({len(dataset)} samples)")
    print()

    scores = {}
    skipped = []

    for fname, indices in groups:
        probs, target = predict_volume(
            rollout, dataset, indices,
            device=device, chunk_size=args.chunk_size,
        )

        if target.sum() == 0:
            skipped.append(fname)
            print(f"{fname}: skipped (no foreground in target)")
            continue

        scores[fname] = volume_dice(probs, target, smooth=0.0)
        print(f"{fname}: {scores[fname]:.4f}")

    if not scores:
        raise RuntimeError(f"No {unit} had foreground in its target; nothing to average.")

    values = torch.tensor(list(scores.values()))
    mean = values.mean().item()
    std = values.std(unbiased=False).item()

    print()
    print(f"{unit.capitalize()}s scored: {len(scores)} (skipped {len(skipped)})")
    print(f"Mean Dice: {mean:.4f}")
    print(f"Std: {std:.4f}")

    if args.csv:
        with open(args.csv, "w", encoding="utf-8") as f:
            f.write(f"{unit},dice\n")
            for fname, score in scores.items():
                f.write(f"{fname},{score:.6f}\n")
        print(f"Per-{unit} scores written to {args.csv}")


if __name__ == "__main__":
    main()
