import torch
import torch.nn as nn


class GatingCoefficient(nn.Module):
    """Predicts g_t in [0,1] from motion history.

    g_t ≈ 0 → Base dominant (navigation phase)
    g_t ≈ 1 → Joint dominant (manipulation phase)
    """

    def __init__(
        self,
        state_dim: int = 15,
        hidden_size: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        history_window: int = 10,
    ):
        super().__init__()
        self.input_proj = nn.Linear(state_dim, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 2,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, motion_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            motion_history: [B, K, state_dim]
        Returns:
            g_t: [B, 1] in range [0, 1]
        """
        x = self.input_proj(motion_history)  # [B, K, hidden_size]
        x = self.transformer(x)  # [B, K, hidden_size]
        x = x[:, -1, :]  # take last timestep [B, hidden_size]
        g_t = self.head(x)  # [B, 1]
        return g_t


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gt_module = GatingCoefficient(state_dim=15, hidden_size=256).to(device)
    history = torch.randn(4, 10, 15).to(device)
    g_t = gt_module(history)
    print(f"g_t shape: {g_t.shape}, range: [{g_t.min().item():.4f}, {g_t.max().item():.4f}]")
    assert g_t.shape == (4, 1)
    assert g_t.min() >= 0 and g_t.max() <= 1
    print("--- GatingCoefficient test passed! ---")
