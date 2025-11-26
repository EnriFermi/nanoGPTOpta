import torch

from .low_freq_muon import LowFreqMuon


def get_muon_impl():
    """Return torch.optim.Muon if available, otherwise fall back to the vendored implementation."""
    muon = getattr(torch.optim, "Muon", None)
    if muon is not None:
        return muon
    from .muon_fallback import Muon as FallbackMuon

    return FallbackMuon

