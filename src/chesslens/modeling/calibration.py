"""Post-hoc temperature scaling for frozen multiclass logits."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional


def _validate_logits(logits: torch.Tensor) -> None:
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] < 2:
        raise ValueError("logits must have shape [N, C] with N > 0 and C >= 2")
    if not torch.is_floating_point(logits):
        raise ValueError("logits must use a floating-point dtype")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must contain only finite values")


def temperature_scale(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Return softmax(logits / temperature), preserving shape, dtype, and device."""
    _validate_logits(logits)
    if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    scaled_logits = logits / temperature
    if not bool(torch.isfinite(scaled_logits).all()):
        raise ValueError("scaled logits are not finite; check temperature and input magnitude")
    return torch.softmax(scaled_logits, dim=1)


def fit_temperature(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    min_temperature: float = 0.05,
    max_temperature: float = 20.0,
    max_iter: int = 100,
) -> float:
    """Fit one positive temperature by minimizing calibration-set cross-entropy.

    Use a held-out calibration set, not training or final test data. Inputs are
    detached and copied to CPU float64; model parameters and input gradients are
    untouched. A smooth bounded log-temperature parameterization prevents invalid
    temperatures. Return the best finite candidate observed, including T=1, so
    calibration-set NLL cannot worsen. Held-out ECE improvement is not guaranteed.
    """
    _validate_logits(logits)
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError("targets must have shape [N] matching logits")
    if targets.dtype != torch.int64:
        raise ValueError("targets must use torch.int64 class indices")
    if bool(((targets < 0) | (targets >= logits.shape[1])).any()):
        raise ValueError("targets contain an out-of-range class index")
    if (
        isinstance(min_temperature, bool)
        or isinstance(max_temperature, bool)
        or not math.isfinite(min_temperature)
        or not math.isfinite(max_temperature)
        or not 0.0 < min_temperature < 1.0 < max_temperature
    ):
        raise ValueError(
            "temperature bounds must satisfy 0 < min_temperature < 1 < max_temperature"
        )
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")

    frozen_logits = logits.detach().to(device="cpu", dtype=torch.float64).clone()
    frozen_targets = targets.detach().to(device="cpu").clone()
    best_nll = float(functional.cross_entropy(frozen_logits, frozen_targets).item())
    if not math.isfinite(best_nll):
        raise RuntimeError("Uncalibrated cross-entropy is not finite")
    best_temperature = 1.0

    log_min = math.log(min_temperature)
    log_span = math.log(max_temperature) - log_min
    initial_fraction = -log_min / log_span
    initial_raw = math.log(initial_fraction / (1.0 - initial_fraction))
    raw_temperature = torch.tensor(initial_raw, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [raw_temperature],
        lr=1.0,
        max_iter=max_iter,
        history_size=10,
        tolerance_grad=1.0e-8,
        tolerance_change=1.0e-12,
        line_search_fn="strong_wolfe",
    )

    def objective() -> torch.Tensor:
        nonlocal best_nll, best_temperature
        log_temperature = log_min + log_span * torch.sigmoid(raw_temperature)
        temperature = torch.exp(log_temperature)
        loss = functional.cross_entropy(frozen_logits / temperature, frozen_targets)
        nll = float(loss.detach().item())
        if not math.isfinite(nll):
            raise RuntimeError("Temperature fitting produced non-finite cross-entropy")
        if nll < best_nll:
            best_nll = nll
            best_temperature = float(temperature.detach().item())
        return loss

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        loss.backward()
        gradient = raw_temperature.grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise RuntimeError("Temperature fitting produced a missing or non-finite gradient")
        return loss

    # Collecting logits may use no_grad(); fitting this scalar still needs autograd.
    with torch.enable_grad():
        optimizer.step(closure)
    with torch.no_grad():
        objective()
    return best_temperature
