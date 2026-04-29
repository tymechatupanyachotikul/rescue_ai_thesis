import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.core.vae import EncoderRNN


def _make_projector(enc_out_dim: int, proj_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(enc_out_dim, enc_out_dim * 2),
        nn.BatchNorm1d(enc_out_dim * 2),
        nn.ReLU(),
        nn.Linear(enc_out_dim * 2, proj_dim),
    )


def _make_predictor(proj_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(proj_dim, proj_dim // 2),
        nn.BatchNorm1d(proj_dim // 2),
        nn.ReLU(),
        nn.Linear(proj_dim // 2, proj_dim),
    )


class BYOLModel(nn.Module):
    """BYOL encoder-only model for self-supervised ECG pretraining.

    Online network : encoder → projector → predictor
    Target network : encoder → projector  (EMA-updated, no gradients)

    encode(x, mask)  → online encoder mean [N, enc_out_dim]  ← downstream probing
    forward(x, mask) → (p, z_target) for computing the BYOL loss
    """

    def __init__(self, input_dim, enc_out_dim, rnn_hidden, enc_H, proj_dim, device, dtype):
        super().__init__()

        # ── Online network ────────────────────────────────────────────────────
        self.online_encoder   = EncoderRNN(
            input_dim   = input_dim,
            rnn_hidden  = rnn_hidden,
            enc_out_dim = enc_out_dim,
            out_distr   = 'normal',
            H           = enc_H,
        ).to(device).to(dtype)
        self.online_projector = _make_projector(enc_out_dim, proj_dim).to(device).to(dtype)
        self.predictor        = _make_predictor(proj_dim).to(device).to(dtype)

        # ── Target network (deep copy of online, frozen) ──────────────────────
        self.target_encoder   = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

    @property
    def device(self):
        return next(self.online_encoder.parameters()).device

    # ── Inference ─────────────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """Online encoder mean [N, enc_out_dim]. Used for downstream linear probing."""
        mu, _ = self.online_encoder(x, mask=mask)
        return mu

    def _online_project_predict(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """encoder → projector → predictor → L2-normalised: [N, proj_dim]."""
        z = self.predictor(self.online_projector(self.encode(x, mask=mask)))
        return F.normalize(z, dim=-1)

    def _target_project(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """Target encoder → projector → L2-normalised (no grad): [N, proj_dim]."""
        with torch.no_grad():
            mu, _ = self.target_encoder(x, mask=mask)
            z = self.target_projector(mu)
        return F.normalize(z, dim=-1)

    # ── EMA update ────────────────────────────────────────────────────────────

    def update_target(self, tau: float = 0.996) -> None:
        """EMA update of target network parameters and BatchNorm buffers."""
        for online_mod, target_mod in [
            (self.online_encoder,   self.target_encoder),
            (self.online_projector, self.target_projector),
        ]:
            for op, tp in zip(online_mod.parameters(), target_mod.parameters()):
                tp.data = tau * tp.data + (1.0 - tau) * op.data
            for (name, ob), (_, tb) in zip(
                online_mod.named_buffers(), target_mod.named_buffers()
            ):
                if 'num_batches_tracked' not in name:
                    tb.data = tau * tb.data + (1.0 - tau) * ob.data
