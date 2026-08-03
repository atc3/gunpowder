import logging

logger = logging.getLogger(__name__)


class WeightedMSELoss:
    """Mean squared error loss with a per-element weight, e.g. to exclude
    masked-out voxels (weight 0) from training.

    Unlike multiplying an unweighted MSE loss by a mask, this divides by the
    sum of the weights rather than the total element count, so masking out
    part of a batch does not artificially shrink the loss -- masked-out
    elements are excluded from the average, not just zeroed out within it.

    Meant to be used as the ``loss`` of :class:`gunpowder.torch.Train`, with
    the weight array passed through ``loss_inputs`` as a named argument,
    e.g. ``loss_inputs={0: prediction, 1: target, "weight": mask}``.

    This is a plain callable rather than a ``torch.nn.Module`` subclass (it
    holds no parameters, so there is nothing to move to a device) -- this
    also keeps ``gunpowder.torch`` importable when torch is not installed.

    Args:

        reduction (``string``, optional):

            ``'mean'``, ``'sum'``, or ``'none'``, matching
            ``torch.nn.MSELoss``. Defaults to ``'mean'``.
    """

    def __init__(self, reduction: str = "mean"):
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.reduction = reduction

    def __call__(self, input, target, weight):
        squared_error = (input - target) ** 2 * weight

        if self.reduction == "none":
            return squared_error
        if self.reduction == "sum":
            return squared_error.sum()

        # clamp to avoid dividing by zero if a batch happens to be fully
        # masked out, rather than silently producing a NaN loss
        total_weight = weight.sum().clamp_min(1e-8)
        return squared_error.sum() / total_weight
