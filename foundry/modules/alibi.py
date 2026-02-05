"""ALiBi (Attention with Linear Biases) position encoding."""

import torch
import torch.nn as nn


def get_alibi_slopes(n_heads: int) -> torch.Tensor:
    """Get ALiBi slopes for attention heads."""

    def get_slopes_power_of_2(n: int) -> torch.Tensor:
        return 2 ** (-(2 ** -(torch.arange(n).float() / n)))

    if (n_heads & (n_heads - 1)) == 0:
        return get_slopes_power_of_2(n_heads)
    closest_power_of_2 = 2 ** torch.floor(torch.log2(torch.tensor(n_heads)))
    return torch.cat(
        [
            get_slopes_power_of_2(int(closest_power_of_2)),
            get_alibi_slopes(int(2 * closest_power_of_2))[0::2][
                : n_heads - int(closest_power_of_2)
            ],
        ]
    )


class ALiBi(nn.Module):
    """ALiBi position encoding via attention bias."""

    def __init__(self, n_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        slopes = get_alibi_slopes(n_heads)
        self.register_buffer("slopes", slopes.view(n_heads, 1, 1))

        alibi = torch.arange(max_seq_len).unsqueeze(0).unsqueeze(0)
        alibi = -torch.abs(alibi.unsqueeze(-1) - alibi.unsqueeze(-2))
        self.register_buffer("alibi_bias", alibi)

    def forward(self, seq_len: int) -> torch.Tensor:
        """Returns attention bias of shape (1, n_heads, seq_len, seq_len)."""
        alibi_bias: torch.Tensor = self.alibi_bias  # type: ignore[assignment]
        slopes: torch.Tensor = self.slopes  # type: ignore[assignment]
        return alibi_bias[:, :, :seq_len, :seq_len] * slopes
