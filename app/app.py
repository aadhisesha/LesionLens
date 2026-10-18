"""LesionLens Glass UI Web Application (FastAPI).

Delivers a state-of-the-art Glassmorphism interface for skin-lesion segmentation,
uncertainty mapping, and ABCDE clinical morphological profiling.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.inference_service import analyze_lesion, get_model

app = FastAPI(
    title="LesionLens Glass UI",
    description="Glassmorphic Clinical Skin Lesion Segmentation & Morphological Profiler",
    version="2.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = REPO_ROOT / "app" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

PRESETS = [
    {
        "id": "ISIC_0000003",
        "filename": "ISIC_0000003.jpg",
        "title": "Atypical Nevus / Irregular Border",
        "description": "Noticeable boundary tortuosity and pigmentation asymmetry.",
    },
    {
        "id": "ISIC_0000013",
        "filename": "ISIC_0000013.jpg",
        "title": "Pigmented Skin Lesion",
        "description": "High-contrast dermoscopic border with central pigmentation.",
    },
    {
        "id": "ISIC_0000014",
        "filename": "ISIC_0000014.jpg",
        "title": "Elevated Dysplastic Lesion",
        "description": "Erythema ring and distinct color variegation zones.",
    },
    {
        "id": "ISIC_0000015",
        "filename": "ISIC_0000015.jpg",
        "title": "Circular Dermoscopic Nevus",
        "description": "High circular compactness with uniform melanocytic pattern.",
    },
    {
        "id": "ISIC_0000020",
        "filename": "ISIC_0000020.jpg",
        "title": "Superficial Spreading Lesion",
        "description": "Multi-tone color variance and irregular boundary indentations.",
    },
]


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the Glass UI single-page application."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.is_file():
        raise HTTPException(status_code=404, detail="Glass UI frontend template not found")
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health_check():
    """Return model runtime state, architecture parameters, and active checkpoint."""
    try:
        _, metadata = get_model()
        return {"status": "healthy", "service": "LesionLens Glass UI", **metadata}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.get("/api/presets")
async def list_presets():
    """Return list of curated demo dermoscopic images for 1-click testing."""
    available = []
    test_dir = REPO_ROOT / "data" / "raw" / "test_images"
    for p in PRESETS:
        full_path = test_dir / p["filename"]
        if full_path.is_file():
            available.append({
                **p,
                "url": f"/api/preset-image/{p['filename']}",
            })
    return {"presets": available}


@app.get("/api/preset-image/{filename}")
async def serve_preset_image(filename: str):
    """Serve preset image bytes."""
    test_dir = REPO_ROOT / "data" / "raw" / "test_images"
    img_path = (test_dir / filename).resolve()
    if not img_path.is_file() or not str(img_path).startswith(str(test_dir.resolve())):
        raise HTTPException(status_code=404, detail="Preset image not found")
    return FileResponse(img_path, media_type="image/jpeg")


@app.post("/api/analyze")
async def analyze_image(
    file: Optional[UploadFile] = File(None),
    preset_id: Optional[str] = Form(None),
    threshold: float = Form(0.50),
    use_tta: bool = Form(True),
):
    """Analyze a dermoscopy image, segment boundary, compute uncertainty and ABCDE metrics."""
    threshold = float(max(0.05, min(0.95, threshold)))

    image: Optional[Image.Image] = None

    if file and file.filename:
        try:
            content = await file.read()
            image = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
    elif preset_id:
        test_dir = REPO_ROOT / "data" / "raw" / "test_images"
        preset_filename = f"{preset_id}.jpg"
        preset_path = test_dir / preset_filename
        if not preset_path.is_file():
            raise HTTPException(status_code=404, detail=f"Preset {preset_id} not found on disk")
        try:
            image = Image.open(preset_path).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load preset: {e}")
    else:
        raise HTTPException(status_code=400, detail="Provide either an uploaded image file or a preset_id")

    try:
        results = analyze_lesion(image=image, threshold=threshold, use_tta=use_tta)
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation analysis failed: {e}")


def main():
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"\n=======================================================")
    print(f"Starting LesionLens Glass UI Frontend")
    print(f"Open your browser: http://{host}:{port}")
    print(f"=======================================================\n")
    uvicorn.run("app.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
