from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as functional

from chesslens.modeling.calibration import fit_temperature, temperature_scale


def _calibration_sample(strength: float = 4.0) -> tuple[torch.Tensor, torch.Tensor]:
    # All predictions are identical; observed class frequencies are 60%/20%/20%.
    logits = torch.tensor([[strength, 0.0, 0.0]] * 10, dtype=torch.float64)
    targets = torch.tensor([0] * 6 + [1] * 2 + [2] * 2, dtype=torch.int64)
    return logits, targets


@pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0, 2.0, 10.0])
def test_temperature_scale_preserves_probability_and_prediction_contract(
    temperature: float,
) -> None:
    logits = torch.tensor([[1.0, 0.0, -1.0], [-0.5, 1.5, 0.5]])
    original = logits.clone()
    probs = temperature_scale(logits, temperature)

    assert probs.shape == logits.shape
    assert probs.dtype == logits.dtype
    assert probs.device == logits.device
    assert bool(torch.isfinite(probs).all())
    assert bool((probs >= 0.0).all())
    torch.testing.assert_close(probs.sum(dim=1), torch.ones(2))
    assert torch.equal(probs.argmax(dim=1), logits.argmax(dim=1))
    assert torch.equal(logits, original)
    if temperature == 1.0:
        torch.testing.assert_close(probs, torch.softmax(logits, dim=1))


def test_temperature_changes_confidence() -> None:
    logits = torch.tensor([[2.0, 0.0, -1.0]])
    original = temperature_scale(logits, 1.0)[0, 0]
    assert temperature_scale(logits, 2.0)[0, 0] < original
    assert temperature_scale(logits, 0.5)[0, 0] > original


@pytest.mark.parametrize("strength", [0.5, 4.0], ids=["underconfident", "overconfident"])
def test_fit_temperature_matches_known_optimum_and_improves_nll(strength: float) -> None:
    logits, targets = _calibration_sample(strength)
    temperature = fit_temperature(logits, targets)

    # At the optimum, p0/p1 = 0.6/0.2 = 3 = exp(strength/T).
    assert temperature == pytest.approx(strength / math.log(3.0), rel=1.0e-3)
    assert functional.cross_entropy(logits / temperature, targets) < functional.cross_entropy(
        logits, targets
    )
    torch.testing.assert_close(
        temperature_scale(logits, temperature)[0],
        torch.tensor([0.6, 0.2, 0.2], dtype=torch.float64),
        atol=1.0e-4,
        rtol=1.0e-4,
    )


def test_fit_temperature_does_not_modify_inputs_or_accumulate_model_gradients() -> None:
    logits, targets = _calibration_sample()
    logits.requires_grad_(True)
    original_logits = logits.detach().clone()
    original_targets = targets.clone()

    with torch.no_grad():
        temperature = fit_temperature(logits, targets)

    assert temperature > 1.0
    assert logits.grad is None
    assert torch.equal(logits.detach(), original_logits)
    assert torch.equal(targets, original_targets)


def test_fit_temperature_keeps_uninformative_logits_at_one() -> None:
    logits = torch.zeros((6, 3), dtype=torch.float64)
    targets = torch.tensor([0, 1, 2, 0, 1, 2])
    assert fit_temperature(logits, targets) == 1.0


def test_fit_temperature_retains_baseline_with_small_iteration_budget() -> None:
    logits, targets = _calibration_sample()
    temperature = fit_temperature(logits, targets, max_iter=1)
    assert 0.05 <= temperature <= 20.0
    assert functional.cross_entropy(logits / temperature, targets) <= functional.cross_entropy(
        logits, targets
    )


@pytest.mark.parametrize(
    ("strength", "minimum", "maximum", "boundary"),
    [(4.0, 0.5, 2.0, 2.0), (0.5, 0.7, 2.0, 0.7)],
)
def test_fit_temperature_respects_custom_bounds(
    strength: float, minimum: float, maximum: float, boundary: float
) -> None:
    logits, targets = _calibration_sample(strength)
    temperature = fit_temperature(logits, targets, min_temperature=minimum, max_temperature=maximum)
    assert minimum <= temperature <= maximum
    assert temperature == pytest.approx(boundary, abs=0.01)


@pytest.mark.parametrize("temperature", [0.0, -1.0, math.nan, math.inf, True])
def test_temperature_scale_rejects_invalid_temperature(temperature: float) -> None:
    with pytest.raises(ValueError, match="temperature"):
        temperature_scale(torch.zeros((2, 3)), temperature)


@pytest.mark.parametrize(
    "logits",
    [
        torch.zeros(3),
        torch.zeros((0, 3)),
        torch.zeros((2, 1)),
        torch.zeros((2, 3), dtype=torch.int64),
        torch.tensor([[math.nan, 0.0, 1.0]]),
        torch.tensor([[math.inf, 0.0, 1.0]]),
    ],
)
def test_both_functions_reject_invalid_logits(logits: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="logits"):
        temperature_scale(logits, 1.0)
    with pytest.raises(ValueError, match="logits"):
        fit_temperature(logits, torch.zeros(2, dtype=torch.int64))


@pytest.mark.parametrize(
    "targets",
    [
        torch.tensor([0]),
        torch.tensor([[0], [1]]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([-1, 0]),
        torch.tensor([0, 3]),
    ],
)
def test_fit_temperature_rejects_invalid_targets(targets: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="targets"):
        fit_temperature(torch.zeros((2, 3)), targets)


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [(0.0, 20.0), (1.0, 20.0), (0.05, 1.0), (2.0, 0.1), (math.nan, 20.0)],
)
def test_fit_temperature_rejects_invalid_bounds(minimum: float, maximum: float) -> None:
    logits, targets = _calibration_sample()
    with pytest.raises(ValueError, match="bounds"):
        fit_temperature(logits, targets, min_temperature=minimum, max_temperature=maximum)


@pytest.mark.parametrize("max_iter", [0, -1, True])
def test_fit_temperature_rejects_invalid_iteration_budget(max_iter: int) -> None:
    logits, targets = _calibration_sample()
    with pytest.raises(ValueError, match="max_iter"):
        fit_temperature(logits, targets, max_iter=max_iter)


def test_temperature_scale_rejects_numeric_overflow() -> None:
    with pytest.raises(ValueError, match="scaled logits"):
        temperature_scale(torch.tensor([[3.0e38, 0.0, 1.0]]), 0.01)
