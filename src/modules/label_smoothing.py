"""Label smoothing for regularization."""

import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing."""

    def __init__(self, smoothing=0.1, ignore_index=-1):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        smooth_loss = -log_probs.mean(dim=-1)

        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss

        mask = targets != self.ignore_index
        return loss[mask].mean() if mask.any() else loss.mean()

