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
(``FILTER_PRESETS``), shown verbatim like the colormap names.
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
    }
    return table.get(key, fallback or key)


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
    }
    return table.get(label, label)
