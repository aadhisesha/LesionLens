"""UI helper utilities for clinical dermoscopy visualization and formatting."""

from __future__ import annotations

from typing import Any, Dict


def format_clinical_summary_text(result: Dict[str, Any]) -> str:
    """Format analysis dictionary into a structured clinical dermoscopy text report."""
    meta = result.get("metadata", {})
    morph = result.get("morphology", {})
    asym = morph.get("asymmetry", {})
    border = morph.get("border", {})
    color = morph.get("color", {})
    diam = morph.get("diameter", {})
    uncert = morph.get("uncertainty", {})
    risk = morph.get("risk_assessment", {})

    lines = [
        "===========================================================",
        "LESIONLENS — CLINICAL DERMOSCOPY MORPHOLOGICAL REPORT",
        "===========================================================",
        f"Model Architecture:       {meta.get('model_name', 'transattunet')} (Phase 3: Efficient-GSA)",
        f"Active Checkpoint:        {meta.get('checkpoint_name', 'best_epoch_26.pth')}",
        f"Input Dimensions:         {meta.get('original_width', 256)} x {meta.get('original_height', 256)} pixels",
        f"Binarization Threshold:   {meta.get('threshold', 0.5):.2f}",
        f"Test-Time Augmentation:   {'Enabled (' + str(meta.get('tta_views', 8)) + ' views)' if meta.get('use_tta') else 'Disabled'}",
        f"Inference Duration:       {meta.get('inference_duration_s', 0.0):.3f}s",
        "-----------------------------------------------------------",
        "QUANTITATIVE ABCDE DERMOSCOPIC MORPHOMETRICS:",
        "-----------------------------------------------------------",
        f"[A] Asymmetry Index:      {asym.get('score', 0.0):.1f}% (H: {asym.get('horizontal', 0.0):.1f}%, V: {asym.get('vertical', 0.0):.1f}%)",
        f"[B] Border Irregularity:  {border.get('score', 0.0):.1f}% ({border.get('regularity', 'N/A')})",
        f"    Compactness (4*pi*A/P^2): {border.get('compactness', 0.0):.3f} (Perimeter: {morph.get('perimeter_pixels', 0):,} px)",
        f"[C] Color Variegation:    {color.get('score', 0.0):.1f}% (RGB Variation: +/-{color.get('mean_std', 0.0):.1f})",
        f"    Detected Tone Zones:  {', '.join(color.get('zones_detected', []))}",
        f"[D] Diameter & Geometry:  {diam.get('est_diam_mm', 0.0):.2f} mm (Equiv: {diam.get('equiv_diam_px', 0.0):.1f} px)",
        f"    Major x Minor Axis:   {diam.get('major_axis_px', 0)} x {diam.get('minor_axis_px', 0)} px",
        f"    Lesion Area Coverage: {morph.get('coverage_pct', 0.0):.2f}% ({morph.get('area_pixels', 0):,} px)",
        f"[E] TTA Disagreement:     {uncert.get('mean_disagreement', 0.0):.1f}% (High-Disagreement Pixels: {uncert.get('high_disagreement_pct', 0.0):.1f}%)",
        "-----------------------------------------------------------",
        f"HEURISTIC ATTENTION LEVEL: {risk.get('level', 'Evaluated').upper()}",
        f"Composite Atypical Score:  {risk.get('atypical_score', 0.0):.1f} / 100",
        f"Advisory Description:     {risk.get('description', '')}",
        "===========================================================",
        "DISCLAIMER: For investigational and research purposes only.",
        "Not a certified diagnostic medical device.",
        "===========================================================",
    ]
    return "\n".join(lines)
