import torch
import torch.nn.functional as F

SCALE_FACTOR = 4


def derive_patch_size(config, scale_factor: int = SCALE_FACTOR) -> int:
    """
    Side length of the training crop
    """
    size = config.DATASET.TARGET_SIZE
    if size % scale_factor:
        raise ValueError(
            f"DATASET.TARGET_SIZE={size} is not divisible by scale_factor="
            f"{scale_factor}; the downscaled grid would not be an integer size."
        )
    return size // scale_factor


def _random_crops(tensors, patch_size, generator=None):
    """
    Crop every tensor at the same random window, chosen per batch element.
    """
    reference = tensors[0]
    batch, _, height, width = reference.shape

    ys = torch.randint(0, height - patch_size + 1, (batch,), generator=generator)
    xs = torch.randint(0, width - patch_size + 1, (batch,), generator=generator)

    cropped = []
    for tensor in tensors:
        cropped.append(
            torch.stack([
                tensor[b, :, ys[b]:ys[b] + patch_size, xs[b]:xs[b] + patch_size]
                for b in range(batch)
            ])
        )
    return cropped


def med_nca_rollout(b1, b2, evolver, seed, iter_n, target=None, scale_factor: int = SCALE_FACTOR, patch_size=None, freeze_channels: int = 1, generator=None,):
    """
    One Med-NCA forward pass: ``b1`` low-res, resample-and-reinject, ``b2``.

    Args:
        b1: Backbone evolved on the downscaled grid.
        b2: Backbone evolved on the full-resolution grid (or a crop of it).
        evolver: Shared Evolver; just two calls.
        seed: (B, C, H, W) seed state, channel 0 holding the input image.
        iter_n: CA steps per backbone. Med-NCA uses same count for both.
        target: Optional (B, 1, H, W) target
        scale_factor: Downscale/upscale factor between the two grids.
        patch_size: Side length of the training crop, or ``None`` for a full-image pass (at inference).
        freeze_channels: Passed through to the CA; 1 freezes channel 0 (input image).
        generator: Optional ``torch.Generator`` for reproducible crops.

    Returns:
        Tuple (final_state, target). target is the cropped target when both target and patch_size were given,
        otherwise it is returned unchanged (None if none was passed).
    """
    _, _, height, width = seed.shape

    seed_lo = F.interpolate(
        seed,
        size=(height // scale_factor, width // scale_factor),
        mode="bilinear",
        align_corners=False,
    )

    state_lo = evolver(
        ca_model=b1,
        state_in=seed_lo,
        conds=None,
        iter_n=iter_n,
        freeze_channels=freeze_channels,
    )

    upsampled = F.interpolate(state_lo, size=(height, width), mode="nearest")

    # Reinject: channel 0 comes from the full resolution seed, the hidden channels from b1 upsampled state
    state = torch.cat([seed[:, :1], upsampled[:, 1:]], dim=1)

    if patch_size is not None:
        if target is None:
            state, = _random_crops([state], patch_size, generator)
        else:
            state, target = _random_crops([state, target], patch_size, generator)

    state = evolver(
        ca_model=b2,
        state_in=state,
        conds=None,
        iter_n=iter_n,
        freeze_channels=freeze_channels,
    )

    return state, target