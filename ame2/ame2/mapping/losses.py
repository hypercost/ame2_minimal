"""Losses for the per-frame mapping model (paper Sec. V-B3)."""

from __future__ import annotations

import torch


def beta_nll_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    beta: float = 0.5,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-sample beta-NLL (Seitzer et al. 2022), paper Eq. 9 with beta = 0.5.

    ``L = sg[var^beta] * (0.5 * log var + 0.5 * (y - mu)^2 / var)``

    Args:
        mu, logvar, target: (B, 1, L, W) or (B, L, W).
        mask: same shape, 1 where the label is valid (e.g. not a ray miss).

    Returns:
        (B,) masked per-sample mean loss.
    """
    var = torch.exp(logvar)
    nll = 0.5 * logvar + 0.5 * (target - mu).square() / var
    if beta > 0:
        nll = nll * var.detach().pow(beta)
    B = nll.shape[0]
    nll = nll.reshape(B, -1)
    if mask is None:
        return nll.mean(dim=1)
    m = mask.reshape(B, -1).to(nll.dtype)
    return (nll * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


def total_variation(y: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Anisotropic TV per sample, paper Eq. 10: (|grad_x y|_1 + |grad_y y|_1) / (H W).

    Args:
        y: (B, 1, H, W) or (B, H, W) ground-truth elevations.
        mask: optional validity mask; differences touching an invalid cell are ignored.
    """
    if y.dim() == 4:
        y = y[:, 0]
        mask = mask[:, 0] if mask is not None else None
    B, H, W = y.shape
    dx = (y[:, 1:, :] - y[:, :-1, :]).abs()
    dy = (y[:, :, 1:] - y[:, :, :-1]).abs()
    if mask is not None:
        m = mask.to(y.dtype)
        dx = dx * (m[:, 1:, :] * m[:, :-1, :])
        dy = dy * (m[:, :, 1:] * m[:, :, :-1])
    return (dx.reshape(B, -1).sum(1) + dy.reshape(B, -1).sum(1)) / float(H * W)


def tv_batch_weights(tv: torch.Tensor, eps: float = 1e-6, floor: float = 0.0) -> torch.Tensor:
    """Batch weights ``w_b = (TV_b + floor) / (sum_b' (TV_b' + floor) + eps)`` (sum to ~1).

    ``floor = 0`` is the paper's formula. A small positive floor keeps
    perfectly flat samples from getting exactly zero weight (our choice; the
    paper does not specify how all-flat batches are handled).
    """
    t = tv + floor
    return t / (t.sum() + eps)


def mapping_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    beta: float = 0.5,
    tv_reweight: bool = True,
    tv_floor: float = 1e-3,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """TV-reweighted beta-NLL over a batch. Returns (loss, logging dict)."""
    per_sample = beta_nll_loss(mu, logvar, target, beta=beta, mask=mask)
    if tv_reweight:
        with torch.no_grad():
            w = tv_batch_weights(total_variation(target, mask), floor=tv_floor)
        loss = (w * per_sample).sum()
    else:
        loss = per_sample.mean()
    with torch.no_grad():
        err = (mu - target).abs()
        if mask is not None:
            mae = (err * mask).sum() / mask.sum().clamp_min(1.0)
        else:
            mae = err.mean()
        info = {"beta_nll": per_sample.mean(), "mae": mae, "mean_std": torch.exp(0.5 * logvar).mean()}
    return loss, info
