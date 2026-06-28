"""
node_i18n.py — Localisation map for DSP node names and parameter labels.

The node ``DISPLAY`` strings and ``ParamSpec.label`` strings live on the (Qt-free)
node classes, so the panel can only reach them through a *variable* —
``self.tr(node.DISPLAY)`` — which ``pylupdate6`` CANNOT extract. This module
restates each label as an explicit ``QCoreApplication.translate("DSPNodes", "…")``
literal so the extractor sees it and Qt Linguist can localise it (e.g. to
Spanish). The panel calls :func:`tr_node` / :func:`tr_param` instead of
``self.tr`` on the raw attribute.

Preset CHOICE labels are intentionally NOT here — they are core domain data
(``FILTER_PRESETS``), shown verbatim like the colormap names. The category
HEADERS that group those choices (:func:`tr_preset_category`) ARE UI chrome,
not domain data, so they ARE translated here — same split ProcessingControls'
static preset combo already uses for its own (separately-built) headers.
"""
from __future__ import annotations

from PyQt6.QtCore import QCoreApplication

# IMPORTANT: every translate() call MUST pass BOTH a literal context AND a
# literal source string. pylupdate6 only extracts literals — wrapping the call
# in a helper (translate(ctx_var, text_var)) makes the strings invisible to the
# extractor. So the calls are written out in full below.


def tr_node(key: str, fallback: str = "") -> str:
    """Localised display name for a node KEY (literals → pylupdate6-extractable)."""
    table = {
        "decon":      QCoreApplication.translate("DSPNodes", "Predictive Deconvolution"),
        "bandpass":   QCoreApplication.translate("DSPNodes", "Bandpass Filter"),
        "preset":     QCoreApplication.translate("DSPNodes", "Filter Preset / Attribute"),
        "tvg":        QCoreApplication.translate("DSPNodes", "TVG (Time-Variant Gain)"),
        "agc":        QCoreApplication.translate("DSPNodes", "AGC (Automatic Gain Control)"),
        "align":      QCoreApplication.translate("DSPNodes", "Delay Alignment (compensate groups)"),
        "water_mute": QCoreApplication.translate("DSPNodes", "Water Column Mute"),
        "swell":      QCoreApplication.translate("DSPNodes", "Swell Filter / Heave Correction"),
        "whiten":     QCoreApplication.translate("DSPNodes", "Spectral Whitening"),
        "fk":         QCoreApplication.translate("DSPNodes", "F-K Dip Filter"),
        "demultiple": QCoreApplication.translate("DSPNodes", "Seabed Multiple Suppression"),
        "notch":      QCoreApplication.translate("DSPNodes", "Notch Filter"),
        "log_compress": QCoreApplication.translate("DSPNodes", "Log Compression (Seismic HDR)"),
        "clahe":      QCoreApplication.translate("DSPNodes", "CLAHE (Adaptive Local Contrast)"),
        "despike":    QCoreApplication.translate("DSPNodes", "Despike (Impulsive Noise Removal)"),
        "trace_eq":   QCoreApplication.translate("DSPNodes", "Trace Equalization (RMS Balance)"),
        "trace_mix":  QCoreApplication.translate("DSPNodes", "Trace Mixing (Horizontal Smoothing)"),
        "median_filter": QCoreApplication.translate("DSPNodes", "Median Filter (Edge-Preserving)"),
        "svd_filter": QCoreApplication.translate("DSPNodes", "SVD Filter (Eigenvalues)"),
        "bilateral_filter": QCoreApplication.translate("DSPNodes", "Bilateral Filter (Smart Smoothing)"),
        "spherical_divergence": QCoreApplication.translate("DSPNodes", "Spherical Divergence (True Amplitude)"),
    }
    return table.get(key, fallback or key)


def tr_tooltip(key: str, fallback: str = "") -> str:
    """Localised 1-2 sentence geophysical explanation for a node KEY — shown
    on hover in the "Add module" dropdown (literals → pylupdate6-extractable).
    """
    table = {
        "decon": QCoreApplication.translate(
            "DSPNodes",
            "Estimates and removes the predictable (repetitive) part of each trace's "
            "waveform, such as reverberation, to sharpen the source pulse and "
            "improve vertical resolution."),
        "bandpass": QCoreApplication.translate(
            "DSPNodes",
            "Passes only frequencies between F-low and F-high, rejecting low-frequency "
            "swell/heave noise and high-frequency electrical/thermal noise outside "
            "the source's useful bandwidth."),
        "preset": QCoreApplication.translate(
            "DSPNodes",
            "Applies a named seismic attribute transform (e.g. envelope, instantaneous "
            "phase/frequency via the Hilbert transform) used to highlight specific "
            "geological features instead of the raw amplitude."),
        "tvg": QCoreApplication.translate(
            "DSPNodes",
            "Exponentially boosts amplitude with time-since-seabed to compensate "
            "for signal attenuation with depth, so weak deep reflectors become as "
            "visible as the strong shallow seabed return."),
        "agc": QCoreApplication.translate(
            "DSPNodes",
            "Statistically equalises amplitude using a sliding RMS window per trace, "
            "boosting weak zones and damping strong ones, regardless of where they "
            "occur — a data-driven gain, unlike TVG's fixed geometric curve."),
        "water_mute": QCoreApplication.translate(
            "DSPNodes",
            "Zeroes everything above each trace's own picked seafloor, removing the "
            "water column entirely so reverberation and direct-wave energy don't "
            "interfere with sub-bottom interpretation."),
        "swell": QCoreApplication.translate(
            "DSPNodes",
            "Removes vessel heave from sea-surface swell by aligning each trace to a "
            "smooth spatial reference via cross-correlation, flattening the wavy "
            "seafloor distortion that heave introduces into the section."),
        "whiten": QCoreApplication.translate(
            "DSPNodes",
            "Flattens the amplitude spectrum within a chosen band while preserving "
            "phase exactly, sharpening reflectors and improving vertical resolution "
            "by recovering frequencies the source/medium attenuated unevenly."),
        "fk": QCoreApplication.translate(
            "DSPNodes",
            "Rejects coherently DIPPING events (side-echoes, diffractions, "
            "cable/towfish noise) by their apparent slope in the "
            "frequency-wavenumber domain, leaving flat reflectors untouched."),
        "demultiple": QCoreApplication.translate(
            "DSPNodes",
            "Predicts and adaptively subtracts the seabed (water-bottom) multiple "
            "reflection at roughly twice the seafloor's two-way time, which "
            "otherwise masks weaker, genuine sub-bottom reflectors beneath it."),
        "notch": QCoreApplication.translate(
            "DSPNodes",
            "Surgically removes a single narrow interference frequency (electrical "
            "resonance, tow-cable strum) with a zero-phase band-stop, leaving the "
            "rest of the spectrum untouched."),
        "log_compress": QCoreApplication.translate(
            "DSPNodes",
            "Phase-preserving logarithmic rescale that compresses dynamic range, "
            "making weak reflectors visible alongside strong ones in the same "
            "display without clipping — like HDR tone-mapping for a photograph."),
        "clahe": QCoreApplication.translate(
            "DSPNodes",
            "Equalises contrast independently within small local tiles instead of "
            "globally, revealing subtle structure in both low- and high-amplitude "
            "regions of the same section simultaneously."),
        "despike": QCoreApplication.translate(
            "DSPNodes",
            "Replaces samples that exceed a robust local amplitude threshold with "
            "the local median, removing 1-2 sample impulsive spikes (electrical "
            "transients, bad bits) while leaving genuine wavelet peaks untouched."),
        "trace_eq": QCoreApplication.translate(
            "DSPNodes",
            "Divides each trace by its own RMS amplitude so every trace carries "
            "comparable energy along the line, correcting for source/receiver "
            "coupling variation before a downstream gain stage amplifies it."),
        "trace_mix": QCoreApplication.translate(
            "DSPNodes",
            "Averages each sample with its horizontal neighbour traces. "
            "Coherent reflectors survive via constructive interference; "
            "incoherent random noise is attenuated via destructive interference."),
        "median_filter": QCoreApplication.translate(
            "DSPNodes",
            "Replaces each sample with the median of its horizontal neighbours, "
            "rejecting isolated noise spikes outright instead of blending them — "
            "preserves sharp fault/reflector edges that a mean filter would blur."),
        "svd_filter": QCoreApplication.translate(
            "DSPNodes",
            "Rebuilds the section from only its top singular components. Coherent "
            "reflectors concentrate their energy into a few components; dense "
            "random noise spreads thinly across all of them and is discarded."),
        "bilateral_filter": QCoreApplication.translate(
            "DSPNodes",
            "Averages horizontal neighbour traces only where their amplitude is "
            "similar — smooths random noise in flat zones while excluding "
            "neighbours across a fault or steep edge, so the edge stays sharp."),
        "spherical_divergence": QCoreApplication.translate(
            "DSPNodes",
            "Deterministic, physics-based gain (t^exponent) referenced to each "
            "trace's own picked seafloor, compensating for wavefront spreading "
            "loss without the artificial water-column boost of a global t=0 curve."),
    }
    return table.get(key, fallback or key)


def tr_preset_category(name: str) -> str:
    """Localised category header grouping PresetNode's "Type" choices (and
    ProcessingControls' static preset combo, which shares the same
    PRESET_CATEGORIES grouping) — literals so pylupdate6 can extract them."""
    table = {
        "Complex Trace Attributes": QCoreApplication.translate(
            "DSPNodes", "Complex Trace Attributes"),
        "Structural Attributes": QCoreApplication.translate(
            "DSPNodes", "Structural Attributes"),
        "2D Image Filters": QCoreApplication.translate(
            "DSPNodes", "2D Image Filters"),
        "Frequency & Smoothing Filters": QCoreApplication.translate(
            "DSPNodes", "Frequency & Smoothing Filters"),
        "Other": QCoreApplication.translate(
            "DSPNodes", "Other"),
    }
    return table.get(name, name)


def tr_param(label: str) -> str:
    """Localised parameter label (matches the English ``ParamSpec.label`` keys)."""
    table = {
        "Operator length":         QCoreApplication.translate("DSPNodes", "Operator length"),
        "Prediction gap":          QCoreApplication.translate("DSPNodes", "Prediction gap"),
        "Pre-whitening":           QCoreApplication.translate("DSPNodes", "Pre-whitening"),
        "F low":                   QCoreApplication.translate("DSPNodes", "F low"),
        "F high":                  QCoreApplication.translate("DSPNodes", "F high"),
        "Type":                    QCoreApplication.translate("DSPNodes", "Type"),
        "Attenuation coef. alpha": QCoreApplication.translate("DSPNodes", "Attenuation coef. alpha"),
        "Window":                  QCoreApplication.translate("DSPNodes", "Window"),
        "Threshold":               QCoreApplication.translate("DSPNodes", "Threshold"),
        "Margin":                  QCoreApplication.translate("DSPNodes", "Margin"),
        "Trace window":            QCoreApplication.translate("DSPNodes", "Trace window"),
        "Max shift":               QCoreApplication.translate("DSPNodes", "Max shift"),
        "Smooth window":           QCoreApplication.translate("DSPNodes", "Smooth window"),
        "Reject dip":              QCoreApplication.translate("DSPNodes", "Reject dip"),
        "Fan half-width":          QCoreApplication.translate("DSPNodes", "Fan half-width"),
        "Mode":                    QCoreApplication.translate("DSPNodes", "Mode"),
        "Seabed threshold":        QCoreApplication.translate("DSPNodes", "Seabed threshold"),
        "Period (0=auto)":         QCoreApplication.translate("DSPNodes", "Period (0=auto)"),
        "Max gain":                QCoreApplication.translate("DSPNodes", "Max gain"),
        "Notch frequency":         QCoreApplication.translate("DSPNodes", "Notch frequency"),
        "Q factor":                QCoreApplication.translate("DSPNodes", "Q factor"),
        "Strength (k)":            QCoreApplication.translate("DSPNodes", "Strength (k)"),
        "Clip Limit":              QCoreApplication.translate("DSPNodes", "Clip Limit"),
        "Tile Grid Size":          QCoreApplication.translate("DSPNodes", "Tile Grid Size"),
        "Window Size":             QCoreApplication.translate("DSPNodes", "Window Size"),
        "Traces to mix":           QCoreApplication.translate("DSPNodes", "Traces to mix"),
        "Traces to evaluate":      QCoreApplication.translate("DSPNodes", "Traces to evaluate"),
        "Principal Components":    QCoreApplication.translate("DSPNodes", "Principal Components"),
        "Amplitude Tolerance":     QCoreApplication.translate("DSPNodes", "Amplitude Tolerance"),
        "Falloff Exponent":        QCoreApplication.translate("DSPNodes", "Falloff Exponent"),
        "Reference Seabed":        QCoreApplication.translate("DSPNodes", "Reference Seabed"),
    }
    return table.get(label, label)
