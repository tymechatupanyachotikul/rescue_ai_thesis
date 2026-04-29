import torch
import torch.nn as nn
from model.core.vae import EncoderRNN


class SimCLRModel(nn.Module):
    """Encoder-only model for SimCLR pretraining.

    Contains only the GRU encoder and a 2-layer projection head.
    No ODE, no decoder, no MoNODE dependency.

    forward(x, mask)  → projected representation [N, proj_dim]  (used during training)
    encode(x, mask)   → encoder mean s0_mu [N, enc_out_dim]     (used for linear probing)
    """

    def __init__(self, input_dim, enc_out_dim, rnn_hidden, enc_H, proj_dim, device, dtype):
        super().__init__()
        self.encoder = EncoderRNN(
            input_dim=input_dim,
            rnn_hidden=rnn_hidden,
            enc_out_dim=enc_out_dim,
            out_distr='normal',
            H=enc_H,
        ).to(device).to(dtype)
        self.proj_head = nn.Sequential(
            nn.Linear(enc_out_dim, enc_out_dim * 2),
            nn.BatchNorm1d(enc_out_dim * 2),
            nn.ReLU(),
            nn.Linear(enc_out_dim * 2, proj_dim),
        ).to(device).to(dtype)

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """Return encoder mean s0_mu: [N, enc_out_dim]. Used for downstream probing."""
        mu, _ = self.encoder(x, mask=mask)
        return mu

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """Return projected representation: [N, proj_dim]. Used during SimCLR training."""
        return self.proj_head(self.encode(x, mask=mask))
