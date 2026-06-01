"""
chaining.py — Thin wrapper around ProfileChain.detect for use by the CLI.

The actual grouping logic lives in model.ProfileChain. This module exists
to provide a stable public import surface for the `detect_chains` function.
"""
from __future__ import annotations

from typing import List, Optional

from .model import ProfileChain, SegyProfile


def detect_chains(
    profiles: List[SegyProfile],
    gap_km: Optional[float] = None,
) -> List[ProfileChain]:
    """
    Group a list of SegyProfile objects into contiguous ProfileChain groups.

    Delegates entirely to ProfileChain.detect — see model.ProfileChain.detect
    for full documentation on the grouping algorithm and criteria.

    Parameters
    ----------
    profiles : list of SegyProfile (error-bearing profiles are silently skipped)
    gap_km   : maximum inter-profile gap in km; defaults to ProfileChain.GAP_KM_MAX

    Returns
    -------
    List of ProfileChain objects, one per detected contiguous group.
    """
    return ProfileChain.detect(profiles, gap_km=gap_km)
