"""LesionLens premium Gradio UI: TransAttUNet + Efficient GSA + TTA uncertainty.

Drop this file into the existing repository at app/gradio_demo.py and keep
app/tta_utils.py, models/model_factory.py, and models/transattunet.py alongside it.
Run from the repository root:
    python -m app.gradio_demo

Environment variables:
    LESIONLENS_CHECKPOINT=/path/to/best_epoch_26.pth
    GRADIO_SHARE=true          # optional; exposes a temporary public URL
    GRADIO_SERVER_NAME=127.0.0.1
    PORT=7860

TTA disagreement is a heuristic uncertainty proxy, not a calibrated probability
of error or a medical diagnosis.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.model_factory import build_model  # noqa: E402
from app.tta_utils import TTA_TRANSFORMS, predict_with_tta  # noqa: E402

IMAGE_SIZE = 256
THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_checkpoint() -> Path:
    """Find the trained Phase 3 checkpoint, preferring an explicit path."""
    candidates: list[Path] = []
    configured = os.environ.get("LESIONLENS_CHECKPOINT", "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            REPO_ROOT / "checkpoints/phase3_efficient_gsa/transattunet_efficient_gsa_best.pth",
            REPO_ROOT / "checkpoints/phase3_efficient_gsa/best_epoch_26.pth",
            REPO_ROOT / "checkpoints/phase3_efficient_gsa/transattunet_baseline_best.pth",
            Path(
                "/content/drive_lesionlens/MyDrive/LesionLens/phase3/runs/"
                "efficient_gsa_20261017_145306/best/best_epoch_26.pth"
            ),
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    checked = "\n".join(f"  • {path}" for path in candidates)
    raise FileNotFoundError(
        "Could not find the trained Phase 3 checkpoint. Set LESIONLENS_CHECKPOINT "
        "to the .pth path or copy the checkpoint into the repository. Checked:\n"
        + checked
    )


def load_model() -> tuple[torch.nn.Module, Path, int | str]:
    checkpoint_path = find_checkpoint()
    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    config = checkpoint.get("config", {})
    model_config = config.get("model", {}) if isinstance(config, dict) else {}
    if not model_config.get("use_efficient_gsa", False):
        raise ValueError(
            "This app expects the Phase 3 Efficient GSA checkpoint with "
            "model.use_efficient_gsa=true. Check LESIONLENS_CHECKPOINT."
        )
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(DEVICE).eval()
    best_epoch = checkpoint.get("best_epoch", checkpoint.get("epoch", "unknown"))
    print(f"[LesionLens] Checkpoint: {checkpoint_path}")
    print(f"[LesionLens] Device: {DEVICE}; best epoch: {best_epoch}")
    print(f"[LesionLens] TTA views per image: {len(TTA_TRANSFORMS)}")
    return model, checkpoint_path, best_epoch


MODEL, CHECKPOINT_PATH, BEST_EPOCH = load_model()


def color_heatmap(values: np.ndarray, cmap_name: str = "magma") -> Image.Image:
    """Convert a [0, 1] scalar map into a colored RGB heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgba = plt.get_cmap(cmap_name)(np.clip(values, 0.0, 1.0))
    rgb = (rgba[..., :3] * 255).round().astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def resize_float_map(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a continuous map to (width, height) without hard thresholding."""
    map_image = Image.fromarray(values.astype(np.float32), mode="F")
    return np.asarray(
        map_image.resize(size, resample=Image.Resampling.BILINEAR), dtype=np.float32
    ).clip(0.0, 1.0)


def segment_image(image: Image.Image | None):
    """Run Phase 3 TTA and return five carousel images, mask download, and status."""
    if image is None:
        return (
            None, None, None, None, None, None,
            f"### No image selected\nUpload a dermoscopic image, then select **Begin analysis**.\n\n_Run: {time.strftime('%H:%M:%S')}_",
        )

    try:
        original = image.convert("RGB")
        resized = original.resize(
            (IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.BILINEAR
        )
        image_array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).unsqueeze(0).float()
        tensor = tensor.to(DEVICE)

        try:
            mean_probability_256, uncertainty_256 = predict_with_tta(MODEL, tensor)
        except RuntimeError as exc:
            if DEVICE.type == "cuda" and "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                return (
                    None, None, None, None, None, None,
                    "### GPU memory exhausted\nTry again after closing other GPU workloads, or run this app on CPU.",
                )
            raise

        binary_256 = (mean_probability_256 >= THRESHOLD).astype(np.uint8)
        original_size = original.size
        mean_probability = resize_float_map(mean_probability_256, original_size)
        uncertainty = resize_float_map(uncertainty_256, original_size)

        binary_256_image = Image.fromarray(binary_256 * 255, mode="L")
        mask_output = binary_256_image.resize(
            original_size, resample=Image.Resampling.NEAREST
        )
        mask_binary_original = np.asarray(mask_output) >= 128

        original_array = np.asarray(original, dtype=np.float32)
        overlay_array = original_array.copy()
        hot_pink = np.array([255.0, 45.0, 158.0], dtype=np.float32)
        overlay_array[mask_binary_original] = (
            0.52 * overlay_array[mask_binary_original]
            + 0.48 * hot_pink
        )
        overlay = Image.fromarray(
            np.clip(overlay_array, 0, 255).astype(np.uint8), mode="RGB"
        )
        probability_heatmap = color_heatmap(mean_probability, "magma")
        uncertainty_heatmap = color_heatmap(uncertainty, "inferno")

        fd, output_path = tempfile.mkstemp(
            prefix="lesionlens_tta_mask_", suffix=".png"
        )
        os.close(fd)
        mask_output.save(output_path, format="PNG")

        foreground_percent = float(mask_binary_original.mean() * 100.0)
        mean_disagreement_percent = float(uncertainty.mean() * 100.0)
        high_disagreement_percent = float((uncertainty >= 0.5).mean() * 100.0)
        status = (
            "### Analysis complete\n"
            f"**Model:** TransAttUNet + Efficient GSA · **Best epoch:** {BEST_EPOCH}  \n"
            f"**Input:** {original.width} × {original.height} px · "
            f"**Model resolution:** {IMAGE_SIZE} × {IMAGE_SIZE} · "
            f"**TTA views:** {len(TTA_TRANSFORMS)} · **Threshold:** {THRESHOLD:.2f}\n\n"
            f"**Predicted foreground coverage:** {foreground_percent:.2f}%  \n"
            f"**Mean scaled TTA disagreement:** {mean_disagreement_percent:.2f}%  \n"
            f"**Pixels with scaled disagreement ≥ 0.5:** {high_disagreement_percent:.2f}%\n\n"
            "> The uncertainty view highlights pixels whose predictions differ across "
            "transformed images. This is a TTA disagreement proxy, not a calibrated "
            "probability of error or a diagnostic confidence score. LesionLens is a "
            "research prototype and does not provide a medical diagnosis."
        )
        return (
            original,
            mask_output,
            overlay,
            probability_heatmap,
            uncertainty_heatmap,
            output_path,
            status,
        )
    except Exception as exc:  # Keep the loader UI able to exit on a visible error.
        print(f"[LesionLens] Inference error: {type(exc).__name__}: {exc}")
        return (
            None, None, None, None, None, None,
            f"### Analysis failed\n`{type(exc).__name__}: {str(exc)[:500]}`\n\n"
            f"Check the terminal for details, confirm the checkpoint path, and try again.\n\n_Run: {time.strftime('%H:%M:%S')}_",
        )


INTRO_HTML = r"""
<div id="ll-intro" aria-hidden="true">
  <div class="intro-grain"></div>
  <div class="intro-orbit intro-orbit-one"></div><div class="intro-orbit intro-orbit-two"></div>
  <div class="feather-stage">
    <svg class="royal-feather" viewBox="0 0 260 310" role="presentation">
      <defs>
        <linearGradient id="featherFill" x1="0" y1="1" x2="1" y2="0">
          <stop offset="0" stop-color="#ff3caa"/><stop offset="0.52" stop-color="#c5ff46"/><stop offset="1" stop-color="#f9e9c9"/>
        </linearGradient>
        <linearGradient id="quillFill" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stop-color="#f6e8cb"/><stop offset="1" stop-color="#ff3caa"/>
        </linearGradient>
        <filter id="featherGlow"><feGaussianBlur stdDeviation="3" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
      </defs>
      <g class="feather-drawing" filter="url(#featherGlow)" fill="none" stroke-linecap="round" stroke-linejoin="round">
        <path class="feather-outline" d="M35 254 C38 205 45 143 84 83 C111 43 160 24 222 27 C219 86 196 143 160 185 C127 224 81 246 35 254 Z" fill="url(#featherFill)" fill-opacity=".09" stroke="url(#featherFill)" stroke-width="1.6"/>
        <path class="feather-quill" d="M35 254 C92 197 143 128 218 31" stroke="url(#quillFill)" stroke-width="2.2"/>
        <g class="feather-barbs" stroke="#d8ff78" stroke-opacity=".72" stroke-width=".9">
          <path d="M49 237 L42 213 M56 228 L35 225 M65 218 L46 193 M74 207 L45 205 M82 197 L55 176 M92 185 L55 184 M101 172 L70 154 M111 159 L73 165 M121 145 L89 136 M130 132 L91 146 M141 117 L110 110 M150 104 L112 123 M161 90 L135 85 M171 78 L134 101 M183 64 L160 60 M193 52 L158 79 M204 40 L184 39"/>
          <path d="M45 246 L72 250 M53 236 L82 239 M61 225 L93 229 M70 213 L104 218 M80 201 L115 205 M90 188 L126 193 M101 175 L137 179 M112 161 L148 164 M123 147 L159 149 M135 132 L170 135 M148 116 L183 118 M160 100 L194 101 M172 84 L205 85 M186 67 L214 69 M199 49 L222 51"/>
        </g>
      </g>
    </svg>
    <div class="feather-spark spark-a"></div><div class="feather-spark spark-b"></div><div class="feather-spark spark-c"></div>
  </div>
  <div class="intro-wordmark"><span class="intro-kicker">INTELLIGENT DERMOSCOPY</span><div>LESION<span>LENS</span></div><p>Precision, with perspective.</p></div>
  <div class="intro-baseline"><span></span> INITIALIZING ANALYSIS ENVIRONMENT <span></span></div>
</div>
"""

CURSOR_HTML = '<div id="plush-cursor-layer" aria-hidden="true"></div>'

HEADER_HTML = r"""
<div class="brandbar">
  <div class="brand-lockup"><span class="brand-glyph"><i></i><i></i><i></i></span><span>LESION<span class="brand-hot">LENS</span></span></div>
  <div class="brand-meta"><span class="status-pip"></span> RESEARCH INTERFACE <span class="meta-divider">/</span> PHASE 03</div>
</div>
<div class="hero-heading">
  <div class="eyebrow"><span class="eyebrow-line"></span> SKIN LESION SEGMENTATION <span class="eyebrow-line"></span></div>
  <h1>See the <em>contour.</em><br><span>Understand the unknown.</span></h1>
  <p>One image. A clearer boundary. A view into prediction disagreement.</p>
</div>
"""

UPLOAD_TOP_HTML = r"""
<div class="upload-overline"><span class="mini-star">✳</span><span>BEGIN A NEW ANALYSIS</span><span class="upload-step">01 <b>/</b> INPUT</span></div>
<div class="upload-title">Bring your image into focus.</div>
<div class="upload-subtitle">Drop a dermoscopic image below, or browse your device. JPG and PNG supported.</div>
"""

LOADER_HTML = r"""
<div id="prismcoil-loader" role="status" aria-live="polite">
  <div class="coil-scene"><span class="coil-ring ring-one"></span><span class="coil-ring ring-two"></span><span class="coil-ring ring-three"></span><span class="coil-core"></span><span class="coil-satellite"></span></div>
  <div class="loader-copy"><span class="loader-kicker">COMPUTING EIGHT VIEWS</span><div class="loader-title">Reading the contours<span class="loader-ellipsis">...</span></div><p>Aligning predictions · Measuring pixel disagreement</p></div>
  <div class="loader-scanline"></div>
</div>
"""

RESULTS_HEADING_HTML = r"""
<div class="results-heading"><div><div class="eyebrow results-eyebrow"><span class="eyebrow-line"></span> ANALYSIS OUTPUT</div><h2>Every view, <em>in sequence.</em></h2></div><div class="results-badge"><span class="status-pip"></span> INFERENCE COMPLETE</div></div>
"""

FOOTER_HTML = r"""
<div class="luxury-footer"><span>LESIONLENS <b>·</b> COMPUTER-AIDED SEGMENTATION</span><span>RESEARCH PROTOTYPE <span class="footer-star">✳</span></span></div>
<div class="medical-note">This research interface is not a medical diagnostic tool. TTA disagreement is a heuristic measure of transformation sensitivity, not a calibrated probability of error.</div>
"""

CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600;700&family=Manrope:wght@300;400;500;600;700;800&display=swap');
:root{color-scheme:dark;--ll-bg:#070707;--ll-surface:#111111;--ll-panel:#121113;--ll-hot:#ff3ca6;--ll-lime:#d7ff48;--ll-gold:#dbcaa5;--ll-ink:#f7f3f0;--ll-muted:#a39da6;--ll-border:rgba(250,243,247,.12);--ll-font:'Manrope',Inter,Arial,sans-serif;--ll-display:'Cormorant Garamond',Georgia,serif}
html,body{background:var(--ll-bg)!important;color:var(--ll-ink)!important;font-family:var(--ll-font)!important;scroll-behavior:smooth}
body{overflow-x:hidden;background-image:radial-gradient(ellipse at 50% -20%,rgba(255,60,166,.105),transparent 42%),radial-gradient(ellipse at 95% 48%,rgba(215,255,72,.045),transparent 30%)!important}
.gradio-container,.gradio-container main,.gradio-container .main{background:transparent!important;color:var(--ll-ink)!important;font-family:var(--ll-font)!important;max-width:100%!important}
.gradio-container{padding:0!important;margin:0!important}
.gradio-container .prose,.gradio-container label,.gradio-container p,.gradio-container span,.gradio-container h1,.gradio-container h2,.gradio-container h3{font-family:var(--ll-font)}
.gradio-container button{font-family:var(--ll-font)!important;transition:transform .22s ease,border-color .22s ease,background .22s ease,box-shadow .22s ease}
.gradio-container button:focus-visible{outline:2px solid var(--ll-lime)!important;outline-offset:3px}
#ll-intro-container,#ll-cursor-container{height:0!important;min-height:0!important;overflow:visible!important;padding:0!important;margin:0!important;border:0!important}
#ll-intro{position:fixed;inset:0;z-index:100000;background:#050506;color:var(--ll-ink);display:flex;flex-direction:column;align-items:center;justify-content:center;overflow:hidden;opacity:1;visibility:visible;transition:opacity .9s cubic-bezier(.22,.7,.18,1),visibility .9s;pointer-events:none}
body.ll-intro-finished #ll-intro{opacity:0;visibility:hidden}
.intro-grain{position:absolute;inset:0;opacity:.14;background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 180 180' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.33'/%3E%3C/svg%3E");mix-blend-mode:screen}
.intro-orbit{position:absolute;width:min(80vw,760px);aspect-ratio:1;border:1px solid rgba(215,255,72,.08);border-radius:50%;animation:orbit-drift 9s linear infinite}.intro-orbit-one{transform:scaleY(.38) rotate(18deg)}.intro-orbit-two{width:min(58vw,560px);border-color:rgba(255,60,166,.1);transform:scaleY(.52) rotate(-34deg);animation-direction:reverse}
.feather-stage{position:relative;width:235px;height:280px;display:grid;place-items:center;animation:feather-float 3s cubic-bezier(.2,.65,.25,1) both;filter:drop-shadow(0 0 26px rgba(255,60,166,.2))}
.royal-feather{width:100%;height:100%;overflow:visible;transform:rotate(-8deg)}.feather-outline{stroke-dasharray:1000;stroke-dashoffset:1000;animation:feather-draw 2.1s ease forwards}.feather-quill{stroke-dasharray:600;stroke-dashoffset:600;animation:feather-draw 1.6s .3s ease forwards}.feather-barbs{opacity:0;animation:barbs-appear .75s 1.05s ease forwards}.feather-spark{position:absolute;width:5px;height:5px;border-radius:50%;background:var(--ll-lime);box-shadow:0 0 14px 4px rgba(215,255,72,.55);animation:spark-rise 2.4s ease-in-out infinite}.spark-a{top:17%;right:9%}.spark-b{bottom:20%;left:5%;animation-delay:.55s;background:var(--ll-hot);box-shadow:0 0 14px 4px rgba(255,60,166,.55)}.spark-c{top:50%;right:0;animation-delay:1.15s}
.intro-wordmark{text-align:center;margin-top:15px;animation:intro-title .9s .45s both}.intro-kicker{font-size:9px;letter-spacing:.35em;color:var(--ll-gold);font-weight:700}.intro-wordmark>div{font-family:var(--ll-display);font-size:clamp(38px,7vw,64px);letter-spacing:.07em;margin-top:6px;font-weight:500}.intro-wordmark>div span{color:var(--ll-hot)}.intro-wordmark p{color:#b3abb5;font-size:11px;letter-spacing:.18em;margin:0}.intro-baseline{position:absolute;bottom:7%;display:flex;align-items:center;gap:14px;font-size:8px;letter-spacing:.2em;color:#88808b}.intro-baseline span{width:28px;height:1px;background:linear-gradient(90deg,transparent,var(--ll-lime))}.intro-baseline span:last-child{background:linear-gradient(90deg,var(--ll-hot),transparent)}
@keyframes feather-draw{to{stroke-dashoffset:0}}@keyframes barbs-appear{to{opacity:1}}@keyframes feather-float{0%{opacity:0;transform:translateY(24px) rotate(-9deg) scale(.88)}45%{opacity:1}100%{opacity:1;transform:translateY(0) rotate(0) scale(1)}}@keyframes intro-title{from{opacity:0;transform:translateY(12px);filter:blur(5px)}to{opacity:1;transform:translateY(0);filter:blur(0)}}@keyframes orbit-drift{to{rotate:360deg}}@keyframes spark-rise{0%,100%{transform:translateY(8px) scale(.65);opacity:.25}50%{transform:translateY(-8px) scale(1.25);opacity:1}}
#plush-cursor-layer{position:fixed;inset:0;z-index:9998;pointer-events:none;overflow:hidden}.plush-tile{position:absolute;width:17px;height:17px;border-radius:5px;border:1px solid rgba(215,255,72,.38);background:radial-gradient(circle at 30% 25%,rgba(255,60,166,.6),rgba(215,255,72,.13) 60%,rgba(255,255,255,0) 100%);box-shadow:0 0 16px rgba(255,60,166,.13),inset 0 0 6px rgba(215,255,72,.13);animation:tile-disperse .8s ease-out forwards;will-change:transform,opacity}@keyframes tile-disperse{0%{opacity:0;transform:translate(-50%,-50%) scale(.2) rotate(0)}20%{opacity:.85}100%{opacity:0;transform:translate(calc(-50% + var(--dx)),calc(-50% + var(--dy))) scale(1.6) rotate(var(--dr))}}
#ll-main-shell{width:min(100%,1240px)!important;margin:0 auto!important;padding:26px clamp(18px,4vw,54px) 38px!important;gap:0!important}
#ll-header{margin-bottom:0!important}.brandbar{display:flex;align-items:center;justify-content:space-between;gap:16px;border-bottom:1px solid var(--ll-border);padding:11px 0 21px}.brand-lockup{display:flex;align-items:center;gap:12px;color:#f7f3ef;font-size:15px;font-weight:800;letter-spacing:.11em}.brand-hot{color:var(--ll-hot)}.brand-glyph{position:relative;width:25px;height:25px;display:flex;align-items:center;justify-content:center;gap:3px;transform:rotate(-22deg)}.brand-glyph i{display:block;width:4px;border-radius:6px;background:var(--ll-lime);box-shadow:0 0 13px rgba(215,255,72,.2)}.brand-glyph i:nth-child(1){height:13px}.brand-glyph i:nth-child(2){height:22px;background:var(--ll-hot)}.brand-glyph i:nth-child(3){height:16px;background:var(--ll-gold)}.brand-meta{font-size:9px;letter-spacing:.16em;color:#9e969f;display:flex;align-items:center;gap:9px}.status-pip{display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--ll-lime);box-shadow:0 0 10px rgba(215,255,72,.65)}.meta-divider{color:#534b53}
.hero-heading{text-align:center;padding:66px 0 38px}.eyebrow{display:flex;align-items:center;justify-content:center;gap:12px;font-size:9px;font-weight:800;letter-spacing:.22em;color:var(--ll-lime)}.eyebrow-line{height:1px;width:28px;background:linear-gradient(90deg,transparent,var(--ll-lime))}.eyebrow-line:last-child{background:linear-gradient(90deg,var(--ll-lime),transparent)}.hero-heading h1{font-family:var(--ll-display)!important;font-weight:500;font-size:clamp(42px,6.2vw,76px);line-height:.98;letter-spacing:-.025em;margin:19px 0 16px;color:#f9f5f0}.hero-heading h1 em,.results-heading h2 em{font-style:italic;color:var(--ll-hot);font-weight:500}.hero-heading h1 span{color:#c8c1c8}.hero-heading p{font-size:12px;color:#a79eaa;letter-spacing:.02em;font-weight:400;margin:0}
#upload-zone{position:relative!important;border-radius:24px!important;border:1px solid rgba(255,255,255,.16)!important;background:linear-gradient(135deg,rgba(255,255,255,.07),rgba(255,255,255,.018) 48%,rgba(255,60,166,.035))!important;backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px);padding:clamp(18px,3.5vw,38px)!important;box-shadow:0 26px 90px rgba(0,0,0,.36),inset 0 1px 0 rgba(255,255,255,.08);overflow:hidden;opacity:0;visibility:hidden;transform:translateY(18px) scale(.99);transition:opacity .9s ease,transform .9s cubic-bezier(.16,1,.3,1),visibility .9s;gap:15px!important}
body.ll-upload-revealed #upload-zone{opacity:1;visibility:visible;transform:translateY(0) scale(1)}#upload-zone:before{content:'';position:absolute;top:0;left:-30%;width:20%;height:1px;background:linear-gradient(90deg,transparent,var(--ll-gold),transparent);filter:drop-shadow(0 0 8px var(--ll-hot));animation:glass-shimmer 5s 1s ease-in-out infinite;pointer-events:none}@keyframes glass-shimmer{0%,42%{left:-30%;opacity:0}55%{opacity:1}85%,100%{left:115%;opacity:0}}
.upload-overline{display:flex;align-items:center;gap:9px;color:var(--ll-lime);font-size:9px;font-weight:800;letter-spacing:.17em}.mini-star{font-size:15px;color:var(--ll-hot)}.upload-step{margin-left:auto;color:#837a87;font-size:9px}.upload-step b{color:var(--ll-hot);padding:0 4px}.upload-title{font-family:var(--ll-display);font-size:clamp(26px,3.5vw,38px);font-weight:500;line-height:1.15;color:#f8f2f5}.upload-subtitle{color:#a69da9;font-size:11px;line-height:1.6;margin-top:-7px}
#upload-image{border:1px dashed rgba(215,255,72,.42)!important;border-radius:17px!important;background:linear-gradient(135deg,rgba(215,255,72,.026),rgba(255,60,166,.035))!important;min-height:174px!important;overflow:hidden;transition:border-color .25s,background .25s,box-shadow .25s}#upload-image:hover{border-color:rgba(255,60,166,.75)!important;background:rgba(255,60,166,.045)!important;box-shadow:inset 0 0 45px rgba(255,60,166,.04)}#upload-image label{color:#e8e0e8!important}#upload-image button{color:#f8f0f8!important;background:rgba(255,255,255,.06)!important;border:1px solid rgba(255,255,255,.16)!important;border-radius:9px!important}#upload-image img{object-fit:contain!important}#upload-image .wrap{background:transparent!important}
#analyze-btn{display:none!important}body.ll-has-image #analyze-btn{display:block!important;animation:button-reveal .45s ease both}#analyze-btn button{width:100%;min-height:54px;border:1px solid rgba(215,255,72,.75)!important;border-radius:12px!important;background:linear-gradient(100deg,#d7ff48,#c7f33b)!important;color:#121309!important;text-transform:uppercase;font-size:10px!important;font-weight:900!important;letter-spacing:.17em!important;box-shadow:0 9px 30px rgba(215,255,72,.08),inset 0 1px 0 rgba(255,255,255,.7)}#analyze-btn button:hover{transform:translateY(-2px);box-shadow:0 14px 34px rgba(215,255,72,.17)!important}#analyze-btn button:active{transform:translateY(0) scale(.995)}@keyframes button-reveal{from{opacity:0;transform:translateY(7px)}to{opacity:1;transform:translateY(0)}}
#upload-note{font-size:9px;color:#8f8791;line-height:1.5}#upload-note p{margin:0!important}
#loader-shell{display:none!important;margin-top:22px!important}body.ll-analyzing #loader-shell{display:block!important;animation:loader-appear .4s ease both}@keyframes loader-appear{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}#prismcoil-loader{position:relative;display:flex;align-items:center;justify-content:center;gap:clamp(18px,4vw,42px);padding:26px;border:1px solid rgba(255,60,166,.23);border-radius:19px;background:radial-gradient(ellipse at 30% 50%,rgba(255,60,166,.09),transparent 55%),linear-gradient(120deg,rgba(255,255,255,.04),rgba(255,255,255,.015));overflow:hidden;min-height:175px}#prismcoil-loader .coil-scene{width:100px;height:100px;position:relative;perspective:400px;flex-shrink:0}.coil-ring{position:absolute;inset:12px;border:2px solid transparent;border-radius:50%;transform-style:preserve-3d}.ring-one{border-top-color:var(--ll-hot);border-left-color:var(--ll-hot);box-shadow:0 0 15px rgba(255,60,166,.24);animation:coil-spin 1.4s linear infinite}.ring-two{inset:4px;border-right-color:var(--ll-lime);border-bottom-color:var(--ll-lime);animation:coil-spin 1.8s linear infinite reverse;transform:rotateX(65deg)}.ring-three{inset:16px;border-top-color:var(--ll-gold);border-bottom-color:var(--ll-gold);animation:coil-spin 1.1s linear infinite;transform:rotateY(68deg)}.coil-core{position:absolute;width:10px;height:10px;top:45px;left:45px;border-radius:50%;background:var(--ll-lime);box-shadow:0 0 18px 5px rgba(215,255,72,.32);animation:core-pulse .9s ease-in-out infinite alternate}.coil-satellite{position:absolute;left:46px;top:2px;width:7px;height:7px;border-radius:50%;background:var(--ll-hot);box-shadow:0 0 15px var(--ll-hot);transform-origin:4px 48px;animation:satellite-spin 2s linear infinite}@keyframes coil-spin{to{rotate:360deg}}@keyframes core-pulse{to{transform:scale(1.65);filter:brightness(1.5)}}@keyframes satellite-spin{to{transform:rotate(360deg)}}.loader-copy{max-width:420px}.loader-kicker{font-size:8px!important;color:var(--ll-lime)!important;letter-spacing:.22em;font-weight:800}.loader-title{font-family:var(--ll-display);font-size:clamp(25px,3vw,36px);line-height:1.15;margin:7px 0;color:#f9f4f8}.loader-title .loader-ellipsis{color:var(--ll-hot);display:inline-block;animation:ellipsis-pulse 1s infinite}@keyframes ellipsis-pulse{50%{opacity:.2}}.loader-copy p{font-size:10px;color:#a49ba6;line-height:1.5;margin:0}.loader-scanline{position:absolute;top:0;bottom:0;width:1px;left:0;background:linear-gradient(transparent,rgba(215,255,72,.5),transparent);animation:scanline 2.8s ease-in-out infinite;opacity:.45}@keyframes scanline{0%{left:0}100%{left:100%}}
#results-panel{max-height:0;opacity:0;overflow:hidden;transform:translateY(22px);pointer-events:none;margin-top:0!important;gap:18px!important;transition:max-height 1.2s ease,opacity .7s ease,transform .7s ease,margin .7s ease}body.ll-has-results #results-panel{max-height:6000px;opacity:1;overflow:visible;transform:translateY(0);pointer-events:auto;margin-top:52px!important}.results-heading{display:flex;align-items:flex-end;justify-content:space-between;gap:20px;padding-bottom:20px;border-bottom:1px solid var(--ll-border)}.results-heading h2{font-family:var(--ll-display)!important;font-size:clamp(32px,4.5vw,48px)!important;font-weight:500!important;line-height:1.05;margin:11px 0 0!important;color:#f8f2f8}.results-eyebrow{justify-content:flex-start;color:var(--ll-lime)}.results-badge{white-space:nowrap;display:flex;align-items:center;gap:8px;font-size:8px;letter-spacing:.15em;color:#aaa2ad;padding:9px 12px;border:1px solid var(--ll-border);border-radius:100px;background:rgba(255,255,255,.025)}
#result-stage{position:relative;display:flex;flex-direction:column;gap:0!important;border:1px solid rgba(255,255,255,.12);border-radius:20px;padding:clamp(10px,2vw,22px)!important;background:linear-gradient(145deg,rgba(255,255,255,.055),rgba(255,255,255,.012));box-shadow:0 20px 70px rgba(0,0,0,.24);min-height:320px;overflow:hidden}#result-stage:after{content:'';position:absolute;inset:0;pointer-events:none;border-radius:inherit;background:linear-gradient(120deg,transparent 20%,rgba(255,255,255,.018) 45%,transparent 65%);transform:translateX(-100%)}body.ll-has-results #result-stage:after{animation:result-sheen 1.5s .25s ease-out both}@keyframes result-sheen{to{transform:translateX(100%)}}#result-stage .result-slide{display:none!important;position:relative!important;width:100%!important;min-width:0!important;margin:0!important;padding:0!important;background:transparent!important;border:0!important;box-shadow:none!important}#result-stage .result-slide.slide-active{display:block!important;animation:slide-enter .5s cubic-bezier(.16,1,.3,1) both}@keyframes slide-enter{from{opacity:0;transform:translateX(20px) scale(.99);filter:blur(3px)}to{opacity:1;transform:translateX(0) scale(1);filter:blur(0)}}#result-stage .result-slide .wrap{background:transparent!important;border:0!important;box-shadow:none!important}#result-stage .result-slide img{max-height:min(66vh,620px)!important;object-fit:contain!important}#result-stage .result-slide label{color:#f0e9ef!important;font-size:11px!important;font-weight:700!important;letter-spacing:.04em}
.slider-shell{display:flex;align-items:center;justify-content:center;gap:18px;margin:4px 0 0}.slide-arrow{width:41px;height:41px;border-radius:50%;border:1px solid rgba(255,255,255,.16);color:#faf4fa;background:rgba(255,255,255,.035);font-size:19px;cursor:pointer}.slide-arrow:hover{border-color:var(--ll-hot);color:var(--ll-hot);transform:translateY(-1px)}.slide-meta{display:flex;align-items:center;justify-content:center;gap:13px;min-width:250px}.slide-meta #ll-slide-index{font-size:10px;font-weight:800;color:var(--ll-lime);letter-spacing:.12em}.slide-meta #ll-slide-index b{color:#68616b;padding:0 3px}.slide-meta #ll-slide-title{font-family:var(--ll-display);font-size:22px;color:#f5edf4}.slide-dots{display:flex;justify-content:center;gap:8px;margin-top:4px}.ll-dot{height:5px;width:18px;border:0!important;border-radius:99px;background:#3b363d!important;padding:0!important;min-width:0!important}.ll-dot.is-active{width:38px;background:var(--ll-hot)!important;box-shadow:0 0 13px rgba(255,60,166,.23)}#analysis-status{border:1px solid var(--ll-border);border-radius:14px;padding:17px 19px;background:rgba(255,255,255,.025);font-size:11px;line-height:1.85;color:#d4cad5}#analysis-status h3{font-size:15px!important;color:var(--ll-lime)!important;margin:0 0 7px!important;font-family:var(--ll-font)!important}#analysis-status blockquote{margin:12px 0 0;padding-left:12px;border-left:2px solid var(--ll-hot);color:#a89eab;font-size:10px}
#mask-download{border:1px solid rgba(215,255,72,.25)!important;border-radius:12px!important;background:rgba(215,255,72,.035)!important;padding:6px!important}#mask-download label{color:#d7ff48!important}#mask-download button{background:rgba(215,255,72,.09)!important;color:var(--ll-lime)!important;border:1px solid rgba(215,255,72,.22)!important}
#ll-footer{margin-top:58px!important}.luxury-footer{display:flex;align-items:center;justify-content:space-between;gap:15px;padding:18px 0;border-top:1px solid var(--ll-border);font-size:8px;font-weight:700;letter-spacing:.15em;color:#7e7682}.luxury-footer b,.footer-star{color:var(--ll-hot)}.medical-note{max-width:800px;font-size:9px;line-height:1.65;color:#716975;padding-bottom:15px}
#plush-cursor-layer .plush-tile:nth-child(even){border-color:rgba(255,60,166,.48);background:radial-gradient(circle,rgba(255,60,166,.25),rgba(215,255,72,.07) 75%,transparent)}
@media(max-width:700px){#ll-main-shell{padding:15px 14px 25px!important}.brand-meta{font-size:7px;letter-spacing:.09em}.hero-heading{padding:47px 0 28px}.hero-heading h1{font-size:45px}.hero-heading p{max-width:290px;margin:auto;line-height:1.7}.brand-lockup{font-size:12px}.brandbar{padding-bottom:15px}.upload-subtitle{line-height:1.7}#prismcoil-loader{flex-direction:column;text-align:center;padding:25px 17px;gap:14px}.coil-scene{transform:scale(.85)}.loader-copy p{max-width:260px}.results-heading{align-items:flex-start;flex-direction:column}.results-badge{margin-top:0}.slide-meta{min-width:0;flex-direction:column;gap:2px}.slide-meta #ll-slide-title{font-size:19px}.slide-arrow{width:36px;height:36px;flex-shrink:0}#result-stage{min-height:240px}.luxury-footer{flex-direction:column;align-items:flex-start;line-height:1.8}.feather-stage{width:190px;height:235px}.intro-wordmark>div{font-size:45px}.intro-baseline{font-size:7px;gap:7px}.intro-baseline span{width:14px}}
@media(pointer:coarse),(prefers-reduced-motion:reduce){#plush-cursor-layer{display:none!important}*,*:before,*:after{scroll-behavior:auto!important;animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important}}

/* Boot-sequence failsafe: the intro and upload reveal run in CSS even if custom JS fails. */
#ll-intro{animation:ll-intro-exit .82s ease 2.55s forwards}
@keyframes ll-intro-exit{0%{opacity:1;visibility:visible}98%{opacity:0;visibility:visible}100%{opacity:0;visibility:hidden}}
#upload-zone{animation:ll-upload-entrance .82s cubic-bezier(.16,1,.3,1) 2.55s both}
@keyframes ll-upload-entrance{0%{opacity:0;visibility:hidden;transform:translateY(18px) scale(.99)}1%{visibility:visible}100%{opacity:1;visibility:visible;transform:translateY(0) scale(1)}}
/* Fallback for the analyze button if Gradio's page JavaScript is blocked. */
#upload-zone:has(#upload-image img[src]) #analyze-btn{display:block!important;animation:button-reveal .45s ease both}
/* Keep the original-image result available as a graceful fallback if slider JS fails. */
body:not(.ll-js-ready) #result-stage .result-slide:first-child{display:block!important}
#results-panel:has(#result-original img[src]){max-height:6000px;opacity:1;overflow:visible;transform:translateY(0);pointer-events:auto;margin-top:34px!important}
"""

APP_JS = r"""() => {
  const body = document.body;
  body.classList.add('ll-js-ready');
  let analysisStarted = false;
  let lastPreview = '';
  let activeSlide = 0;
  const slideIds = ['result-original', 'result-mask', 'result-overlay', 'result-probability', 'result-uncertainty'];
  const slideTitles = ['Original image', 'Predicted binary mask', 'Segmentation overlay', 'Mean probability map', 'TTA uncertainty map'];

  // Royal feather intro: dissolve first, reveal the glass uploader shortly after.
  window.setTimeout(() => {
    body.classList.add('ll-intro-finished');
    window.setTimeout(() => body.classList.add('ll-upload-revealed'), 520);
  }, 2850);

  // Plush-tile pointer trail. Disabled for touch screens and reduced-motion users.
  const reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const coarse = window.matchMedia && window.matchMedia('(pointer: coarse)').matches;
  const layer = document.getElementById('plush-cursor-layer');
  let lastTileAt = 0;
  if (layer && !reduced && !coarse) {
    document.addEventListener('pointermove', (event) => {
      const now = performance.now();
      if (now - lastTileAt < 38) return;
      lastTileAt = now;
      const tile = document.createElement('span');
      tile.className = 'plush-tile';
      tile.style.left = event.clientX + 'px';
      tile.style.top = event.clientY + 'px';
      tile.style.setProperty('--dx', ((Math.random() - .5) * 46) + 'px');
      tile.style.setProperty('--dy', (15 + Math.random() * 34) + 'px');
      tile.style.setProperty('--dr', ((Math.random() - .5) * 100) + 'deg');
      tile.style.opacity = String(.45 + Math.random() * .45);
      layer.appendChild(tile);
      window.setTimeout(() => tile.remove(), 850);
    }, {passive:true});
  }

  // Reveal the analysis button only after Gradio displays an uploaded image.
  function inspectUpload() {
    const root = document.getElementById('upload-image');
    if (!root) return;
    const images = Array.from(root.querySelectorAll('img[src]'));
    const preview = images.find(img => img.getAttribute('src') && img.getBoundingClientRect().width > 20);
    const src = preview ? preview.getAttribute('src') : '';
    const hasImage = Boolean(src && src.length > 10);
    body.classList.toggle('ll-has-image', hasImage);
    if (hasImage && src !== lastPreview) {
      lastPreview = src;
      body.classList.remove('ll-has-results');
    }
    if (!hasImage) {
      lastPreview = '';
      body.classList.remove('ll-has-results');
    }
  }
  function watchUpload() {
    const root = document.getElementById('upload-image');
    if (!root || root.dataset.llUploadObserved === '1') { inspectUpload(); return; }
    root.dataset.llUploadObserved = '1';
    new MutationObserver(inspectUpload).observe(root, {subtree:true, childList:true, attributes:true, attributeFilter:['src','style']});
    root.addEventListener('change', inspectUpload, true);
    window.setInterval(inspectUpload, 650);
    inspectUpload();
  }
  watchUpload();

  // Prismcoil loading animation: begin on analysis click, finish when the status output returns.
  document.addEventListener('click', (event) => {
    if (event.target.closest('#analyze-btn')) {
      analysisStarted = true;
      body.classList.add('ll-analyzing');
      body.classList.remove('ll-has-results');
      window.setTimeout(() => {
        if (analysisStarted) {
          const statusNode = document.getElementById('analysis-status');
          if (statusNode && !statusNode.textContent.trim()) {
            statusNode.textContent = 'Still processing the eight transformed views. Please wait for inference to finish.';
          }
        }
      }, 45000);
    }
  }, true);

  // Result slides: the model output widgets are kept in Gradio; JS just switches views.
  function getSlides() {
    return slideIds.map(id => document.getElementById(id)).filter(Boolean);
  }
  function showSlide(index) {
    const slides = getSlides();
    if (!slides.length) return;
    activeSlide = (index + slideIds.length) % slideIds.length;
    slides.forEach((node, i) => node.classList.toggle('slide-active', slideIds.indexOf(node.id) === activeSlide));
    const idx = document.getElementById('ll-slide-index');
    const title = document.getElementById('ll-slide-title');
    if (idx) idx.innerHTML = String(activeSlide + 1).padStart(2,'0') + ' <b>/</b> 05';
    if (title) title.textContent = slideTitles[activeSlide];
    document.querySelectorAll('#ll-slide-dots .ll-dot').forEach((dot, i) => dot.classList.toggle('is-active', i === activeSlide));
  }
  document.addEventListener('click', (event) => {
    if (event.target.closest('#ll-slide-prev')) { event.preventDefault(); showSlide(activeSlide - 1); }
    else if (event.target.closest('#ll-slide-next')) { event.preventDefault(); showSlide(activeSlide + 1); }
    else {
      const dot = event.target.closest('#ll-slide-dots .ll-dot');
      if (dot) { event.preventDefault(); showSlide(Number(dot.dataset.slide || 0)); }
    }
  });

  const resultsRoot = document.getElementById('result-stage');
  if (resultsRoot) {
    let pointerStartX = null;
    resultsRoot.addEventListener('pointerdown', event => { pointerStartX = event.clientX; }, {passive:true});
    resultsRoot.addEventListener('pointerup', event => {
      if (pointerStartX === null) return;
      const delta = event.clientX - pointerStartX;
      if (Math.abs(delta) > 65) showSlide(activeSlide + (delta < 0 ? 1 : -1));
      pointerStartX = null;
    }, {passive:true});
  }

  const statusRoot = document.getElementById('analysis-status');
  function watchCompletion() {
    const root = document.getElementById('analysis-status');
    if (!root || root.dataset.llObserved === '1') return;
    root.dataset.llObserved = '1';
    new MutationObserver(() => {
      const message = root.textContent.trim();
      if (analysisStarted && message.length > 0) {
        analysisStarted = false;
        body.classList.remove('ll-analyzing');
        body.classList.add('ll-has-results');
        showSlide(0);
        window.setTimeout(() => document.getElementById('results-panel')?.scrollIntoView({behavior:'smooth',block:'start'}), 220);
      }
    }).observe(root, {subtree:true, childList:true, characterData:true});
  }
  watchCompletion();
  // Reinitialize after Gradio updates output containers dynamically.
  new MutationObserver(() => { watchCompletion(); watchUpload(); inspectUpload(); }).observe(document.body, {subtree:true, childList:true});
  document.addEventListener('keydown', event => {
    if (event.key === 'ArrowRight' && body.classList.contains('ll-has-results')) showSlide(activeSlide + 1);
    if (event.key === 'ArrowLeft' && body.classList.contains('ll-has-results')) showSlide(activeSlide - 1);
  });
}"""

# Gradio 6 moved theme/css/js from Blocks(...) to launch(...).
# Pass them to the correct place for the installed Gradio major version;
# otherwise the page styling and startup JavaScript can silently be ignored.
_GRADIO_MAJOR = int(str(getattr(gr, "__version__", "4")).split(".", 1)[0])
_THEME = gr.themes.Soft(primary_hue="pink", secondary_hue="lime", neutral_hue="slate")
_BLOCKS_OPTIONS = {"title": "LesionLens — Precision in every pixel"}
_LAUNCH_OPTIONS: dict[str, Any] = {}
if _GRADIO_MAJOR >= 6:
    _LAUNCH_OPTIONS.update({"theme": _THEME, "css": CSS, "js": APP_JS})
else:
    _BLOCKS_OPTIONS.update({"theme": _THEME, "css": CSS, "js": APP_JS})

with gr.Blocks(**_BLOCKS_OPTIONS) as demo:
    gr.HTML(INTRO_HTML, elem_id="ll-intro-container", container=False, show_label=False)
    gr.HTML(CURSOR_HTML, elem_id="ll-cursor-container", container=False, show_label=False)

    with gr.Column(elem_id="ll-main-shell", variant="panel"):
        gr.HTML(HEADER_HTML, elem_id="ll-header", container=False, show_label=False)

        with gr.Column(elem_id="upload-zone"):
            gr.HTML(UPLOAD_TOP_HTML, container=False, show_label=False)
            uploaded_image = gr.Image(
                type="pil",
                sources=["upload", "clipboard"],
                image_mode="RGB",
                label="",
                show_label=False,
                height=230,
                elem_id="upload-image",
            )
            analyze_button = gr.Button(
                "✳  Begin analysis",
                variant="primary",
                size="lg",
                elem_id="analyze-btn",
            )
            gr.Markdown(
                "Private by default when run locally · No retraining · TTA uses eight image views",
                elem_id="upload-note",
            )

        gr.HTML(LOADER_HTML, elem_id="loader-shell", container=False, show_label=False)

        with gr.Column(elem_id="results-panel"):
            gr.HTML(RESULTS_HEADING_HTML, container=False, show_label=False)
            with gr.Column(elem_id="result-stage"):
                original_view = gr.Image(
                    type="pil", label="Original image", show_label=True,
                    interactive=False, elem_id="result-original", elem_classes=["result-slide"],
                )
                mask_view = gr.Image(
                    type="pil", label="Predicted binary mask", show_label=True,
                    interactive=False, elem_id="result-mask", elem_classes=["result-slide"],
                )
                overlay_view = gr.Image(
                    type="pil", label="Segmentation overlay", show_label=True,
                    interactive=False, elem_id="result-overlay", elem_classes=["result-slide"],
                )
                probability_view = gr.Image(
                    type="pil", label="Mean probability map", show_label=True,
                    interactive=False, elem_id="result-probability", elem_classes=["result-slide"],
                )
                uncertainty_view = gr.Image(
                    type="pil", label="TTA uncertainty proxy", show_label=True,
                    interactive=False, elem_id="result-uncertainty", elem_classes=["result-slide"],
                )
            # The controls use ordinary HTML buttons so the slider remains smooth and the file stays self-contained.
            gr.HTML(r"""
<div class="slider-shell">
  <button id="ll-slide-prev" class="slide-arrow" type="button" aria-label="Previous result">←</button>
  <div class="slide-meta"><span id="ll-slide-index">01 <b>/</b> 05</span><span id="ll-slide-title">Original image</span></div>
  <button id="ll-slide-next" class="slide-arrow" type="button" aria-label="Next result">→</button>
</div>
<div id="ll-slide-dots" class="slide-dots" aria-label="Choose result image">
  <button class="ll-dot is-active" data-slide="0" aria-label="Original image"></button>
  <button class="ll-dot" data-slide="1" aria-label="Predicted mask"></button>
  <button class="ll-dot" data-slide="2" aria-label="Segmentation overlay"></button>
  <button class="ll-dot" data-slide="3" aria-label="Probability map"></button>
  <button class="ll-dot" data-slide="4" aria-label="Uncertainty heatmap"></button>
</div>
""", elem_id="slider-controls", container=False, show_label=False)
            mask_download = gr.File(label="Download predicted mask (PNG)", elem_id="mask-download")
            status_view = gr.Markdown(
                "Upload a dermoscopic image to begin. The analysis summary will appear here.",
                elem_id="analysis-status",
            )

        gr.HTML(FOOTER_HTML, elem_id="ll-footer", container=False, show_label=False)

    analyze_button.click(
        fn=segment_image,
        inputs=[uploaded_image],
        outputs=[
            original_view,
            mask_view,
            overlay_view,
            probability_view,
            uncertainty_view,
            mask_download,
            status_view,
        ],
        api_name="segment_with_uncertainty",
    )


if __name__ == "__main__":
    share = os.environ.get("GRADIO_SHARE", "false").lower() in {"1", "true", "yes"}
    server_name = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    port = int(os.environ.get("PORT", "7860"))
    demo.queue().launch(
        server_name=server_name,
        server_port=port,
        share=share,
        show_error=True,
        **_LAUNCH_OPTIONS,
    )
