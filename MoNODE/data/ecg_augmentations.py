import random
import torch
from scipy.signal import resample


class GaussianNoise:
    def __init__(self, sigma=0.05):
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, D)
        return x + torch.randn_like(x) * self.sigma


class RandomResizedCrop:
    def __init__(self, min_frac=0.7, max_frac=1.0):
        self.min_frac = min_frac
        self.max_frac = max_frac

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, D)
        T        = x.shape[0]
        lo       = max(1, int(T * self.min_frac))
        hi       = max(lo, int(T * self.max_frac))
        crop_len = random.randint(lo, hi)
        start    = random.randint(0, T - crop_len)
        cropped  = x[start:start + crop_len]
        resampled = torch.from_numpy(resample(cropped.numpy(), T, axis=0))
        return resampled.to(x.dtype)


class TimeOut:
    def __init__(self, max_frac=0.2):
        self.max_frac = max_frac

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, D)
        T        = x.shape[0]
        zero_len = random.randint(0, int(T * self.max_frac))
        if zero_len == 0:
            return x
        start = random.randint(0, T - zero_len)
        out = x.clone()
        out[start:start + zero_len] = 0.0
        return out


class SimCLRAugment:
    """Compose GaussianNoise → RandomResizedCrop → TimeOut for SimCLR views."""

    def __init__(self, noise_sigma=0.05, crop_min_frac=0.7, crop_max_frac=1.0, timeout_max_frac=0.2):
        self.noise   = GaussianNoise(noise_sigma)
        self.crop    = RandomResizedCrop(crop_min_frac, crop_max_frac)
        self.timeout = TimeOut(timeout_max_frac)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, D)
        return self.timeout(self.crop(self.noise(x)))


def augment_batch(X: torch.Tensor, augmenter: SimCLRAugment) -> torch.Tensor:
    """Apply augmenter independently to each sample. X: (N, T, D) → (N, T, D)."""
    return torch.stack([augmenter(X[i].cpu()).to(X.device) for i in range(X.shape[0])])
