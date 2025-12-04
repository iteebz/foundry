"""Direct Preference Optimization (DPO) loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DPOLoss(nn.Module):
    """DPO loss for preference-based training."""

    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute DPO loss.
        
        Args:
            policy_chosen_logps: Log probs of chosen responses under policy
            policy_rejected_logps: Log probs of rejected responses under policy
            reference_chosen_logps: Log probs of chosen responses under reference
            reference_rejected_logps: Log probs of rejected responses under reference
        
        Returns:
            DPO loss (scalar tensor)
        """
        policy_logratios = policy_chosen_logps - policy_rejected_logps
        reference_logratios = reference_chosen_logps - reference_rejected_logps

        logits = policy_logratios - reference_logratios

        if self.label_smoothing == 0.0:
            losses = -F.logsigmoid(self.beta * logits)
        else:
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )

        return losses.mean()


def compute_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute log probabilities for given labels.
    
    Args:
        logits: Model logits (batch_size, seq_len, vocab_size)
        labels: Target labels (batch_size, seq_len)
    
    Returns:
        Sum of log probabilities per sequence (batch_size,)
    """
    log_probs = F.log_softmax(logits, dim=-1)
    
    per_token_logps = torch.gather(
        log_probs[:, :-1, :], dim=2, index=labels[:, 1:].unsqueeze(2)
    ).squeeze(2)
    
    return per_token_logps.sum(dim=1)
