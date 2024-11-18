"""
Adapted from hyena-dna, src/tasks/metrics.py : https://github.com/HazyResearch/hyena-dna
"""

import torch
import torch.nn.functional as F


def get_output_mask(outs, sample_len):
    mask = torch.zeros_like(outs, dtype=torch.bool)
    for i, length in enumerate(sample_len):
        mask[i, :length] = 1
    return mask


def get_stacked_batch_loss(loss, sample_len):
    return torch.stack([loss[i, :length].mean() for i, length in enumerate(sample_len)])


def get_nonzero_batch_vals(outs, y, sample_len):
    # Computes the loss of the first `lens` items in the batches
    mask = get_output_mask(outs, sample_len)
    outs_masked = torch.masked_select(outs, mask)
    y_masked = torch.masked_select(y, mask)
    return outs_masked, y_masked


def discrete_accuracy(vals, y, sample_len=None, reduction="mean"):
    transform_func = {"mean": torch.mean, "sum": torch.sum, "none": lambda x: x}

    vals_m, y_m = (
        get_nonzero_batch_vals(vals, y, sample_len)
        if sample_len is not None
        else (vals, y)
    )
    vals = torch.eq(torch.round(vals_m), y_m).float()
    return transform_func[reduction](vals)


def accuracy(logits, y):
    logits = logits.view(-1, logits.shape[-1])
    preds = torch.argmax(logits, dim=-1)
    if y.numel() > logits.shape[0]:
        # Mixup leads to this case: use argmax class
        y = y.argmax(dim=-1)
    y = y.view(-1)
    return torch.eq(preds, y).float().mean()


def weighted_mse(outs, y, weights):
    loss = F.mse_loss(outs, y, reduction="none")
    return (loss * weights).mean()


def mse(outs, y, sample_len=None, reduction="mean"):
    # assert outs.shape[:-1] == y.shape and outs.shape[-1] == 1
    # outs = outs.squeeze(-1)
    if len(y.shape) < len(outs.shape):
        assert outs.shape[-1] == 1
        outs = outs.squeeze(-1)
    if sample_len is None:
        return F.mse_loss(outs, y)
    return F.mse_loss(*get_nonzero_batch_vals(outs, y, sample_len), reduction=reduction)


def get_dynamicaly_weighted_loss(loss_tuple, user_weights, epsilon=1e-6):
    include_losses = [
        loss for loss, uw in zip(loss_tuple, user_weights) if uw > epsilon
    ]
    loss_weights = [1 / (loss.detach().item() + epsilon) for loss in include_losses]
    return sum(
        loss_weights[i] / sum(loss_weights) * loss
        for i, loss in enumerate(include_losses)
    )


output_metric_fns = {
    "discrete_accuracy": discrete_accuracy,
    "accuracy": accuracy,
    "mse": mse,
    "weighted_mse": weighted_mse,
}
