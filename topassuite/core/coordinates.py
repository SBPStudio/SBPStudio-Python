"""
coordinates.py — CRS resolution and validation helpers.

Public API
----------
resolve_crs(crs_str)  → str           normalise to "EPSG:XXXX" or WKT
validate_crs(crs_str) → pyproj.CRS   raises CRSError if invalid
"""
from __future__ import annotations

from pyproj import CRS

from .tasks import CRSError


def resolve_crs(crs_str: str) -> str:
    """
    Normalise a CRS string (bare EPSG integer, "EPSG:XXXX", or WKT) to a
    canonical form that pyproj can accept.

    Returns the canonical EPSG authority string (e.g. "EPSG:4326") when
    the CRS can be parsed by pyproj, otherwise raises CRSError.
    """
    if not crs_str or not str(crs_str).strip():
        raise CRSError("Empty CRS string.")
    raw = str(crs_str).strip()
    # Bare integer → prepend EPSG:
    if raw.isdigit():
        raw = f"EPSG:{raw}"
    try:
        crs = CRS.from_user_input(raw)
        # Use the canonical authority string when available
        auth = crs.to_authority()
        if auth:
            return f"{auth[0]}:{auth[1]}"
        return raw
    except Exception as exc:
        raise CRSError(f"Cannot parse CRS '{crs_str}': {exc}") from exc


def validate_crs(crs_str: str) -> CRS:
    """
    Parse and validate a CRS string.

    Returns the pyproj.CRS object on success.
    Raises CRSError if the string is empty, malformed, or unrecognised.
    """
    if not crs_str or not str(crs_str).strip():
        raise CRSError("Empty CRS string.")
    try:
        return CRS.from_user_input(str(crs_str).strip())
    except Exception as exc:
        raise CRSError(f"Invalid CRS '{crs_str}': {exc}") from exc
