"""
chaining.py — Chain grouping/import helpers (CLI + GUI public surface).

The actual grouping logic lives in model.ProfileChain. This module provides a
stable public import surface for ``detect_chains`` (auto-detection) and
``import_chains_from_directory`` (manual directory-structure import).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

from .io_segy import load_profile
from .model import ProfileChain, SegyProfile
from .tasks import CancelToken, ProgressCallback, _noop_progress

# Extensions recognised as SEG-Y when scanning a campaign directory.
SEGY_EXTS = (".sgy", ".segy", ".seg")


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


def import_chains_from_directory(
    root: str,
    progress: ProgressCallback = _noop_progress,
    cancel: Optional[CancelToken] = None,
    existing_names: Iterable[str] = (),
) -> List[ProfileChain]:
    """
    Build one :class:`ProfileChain` per immediate subdirectory of ``root`` that
    contains SEG-Y files — a manual alternative to auto-detection for campaigns
    pre-organised into one folder per line/chain.

    Memory-flat by construction: every profile is loaded as a HEADER-ONLY stub
    (``load_traces=False``) and ``ProfileChain.__init__`` is lazy (it never
    concatenates trace matrices), so scanning a whole campaign tree reads only
    headers. The stitched matrix is assembled later, on demand, when a chain is
    actually viewed (``ProfileChain.load_chain_traces``).

    Each chain is named after its subdirectory. Subdirectories that are empty,
    contain no readable SEG-Y, or whose name is already in ``existing_names``
    (and intra-batch duplicates) are skipped.

    Parameters
    ----------
    root           : campaign directory whose immediate subdirectories are scanned
    progress       : (fraction, message) sink — reports per-subdirectory progress
    cancel         : cooperative cancellation token (polled per subdir/file)
    existing_names : chain names already present (skipped to avoid duplicates)

    Returns
    -------
    List of lazily-initialised ProfileChain objects (data not loaded).
    """
    if cancel is None:
        cancel = CancelToken.never()
    seen = set(existing_names)

    root_path = Path(root)
    subdirs = sorted((p for p in root_path.iterdir() if p.is_dir()),
                     key=lambda p: p.name.lower())
    chains: List[ProfileChain] = []
    total = len(subdirs)
    for i, sub in enumerate(subdirs):
        cancel.check()
        progress(i / total if total else 1.0, sub.name)
        if sub.name in seen:                       # duplicate name → skip
            continue
        files = sorted((f for f in sub.iterdir()
                        if f.is_file() and f.suffix.lower() in SEGY_EXTS),
                       key=lambda f: f.name.lower())
        if not files:                              # empty / no SEG-Y → skip
            continue
        profiles: List[SegyProfile] = []
        for f in files:
            cancel.check()
            prof = load_profile(str(f), load_traces=False)   # stub: headers only
            if not prof.error:
                profiles.append(prof)
        if not profiles:                           # all files unreadable → skip
            continue
        chain = ProfileChain(profiles)
        chain.name = chain.label = sub.name        # name the chain after the folder
        chains.append(chain)
        seen.add(sub.name)                         # guard against intra-batch dupes
    progress(1.0, "")
    return chains
