# -*- coding: utf-8 -*-
"""
Thin XAI adapter: reuse XAI.py pipeline 1:1 on the READY trajectory (no second denoising).
Adds a full diffusion-steps mosaic image and returns a JSON‑serializable result.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union
from PIL import Image
import numpy as np
import torch

# Import original notebook pipeline to guarantee pixel parity
from .XAI import (
    ModernXAIAnalyzer,
    MelanomaClassifierAdaptive,
    run_comprehensive_xai_pipeline,
    CLASS_NAMES, NUM_CLASSES
)

def _to_uint8_rgb(t: torch.Tensor) -> np.ndarray:
    """(1,C,H,W) or (C,H,W) in [-1,1] -> HxWx3 uint8 in [0,255]."""
    x = t.detach().cpu()
    if x.ndim == 4:
        x = x.squeeze(0)
    if x.shape[0] in (1, 3):
        x = x.permute(1, 2, 0)
    arr = (x.numpy() + 1.0) / 2.0
    arr = np.clip(arr, 0.0, 1.0)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return (arr * 255).astype(np.uint8)

def _save_full_trajectory_grid(trajectory: List[torch.Tensor], out_path: Path, cols: int = 10, pad: int = 2) -> None:
    """Save a mosaic with ALL diffusion steps (rows x cols)."""
    if not trajectory:
        return
    frames = [_to_uint8_rgb(t) for t in trajectory]
    h, w = frames[0].shape[:2]
    n = len(frames)
    cols = max(1, cols)
    rows = (n + cols - 1) // cols
    canvas = np.ones((rows * h + pad * (rows - 1),
                      cols * w + pad * (cols - 1),
                      3), dtype=np.uint8) * 255
    for idx, fr in enumerate(frames):
        r = idx // cols
        c = idx % cols
        y = r * (h + pad)
        x = c * (w + pad)
        canvas[y:y+h, x:x+w] = fr
    Image.fromarray(canvas).save(out_path)

def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy/torch/Path to JSON‑serializable types."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    try:
        import numpy as _np
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        if isinstance(obj, (_np.floating, _np.integer, _np.bool_)):
            return obj.item()
    except Exception:
        pass
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return obj

class IntegratedXAIAnalyzer:
    def __init__(self, device: Optional[torch.device] = None, verbose: bool = True):
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        # Create exactly the same classifier XAI.py uses
        self.classifier = MelanomaClassifierAdaptive(num_classes=NUM_CLASSES, architecture='auto', pretrained=True).to(self.device)
        self.classifier.eval()
        # And the same analyzer
        self.xai_analyzer = ModernXAIAnalyzer(self.classifier, self.device, verbose=verbose)

    def analyze_trajectory(self,
                           trajectory: List[torch.Tensor],
                           class_name: str,
                           seed: Optional[int],
                           inference_steps: int,
                           filename: str,
                           file_path: str,
                           timesteps: Optional[List[float]] = None) -> Optional[Dict[str, Any]]:
        if not trajectory:
            return None
        if timesteps is None or len(timesteps) != len(trajectory):
            timesteps = list(range(len(trajectory)))
        try:
            target_class_id = CLASS_NAMES.index(class_name)
        except ValueError:
            target_class_id = 0

        out_dir = Path(file_path).parent.parent / 'xai_results' / class_name / f"{Path(filename).stem}_{seed if seed is not None else 'n'}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Call the original pipeline (pixel-identical artifacts), but pass the ready trajectory
        results = run_comprehensive_xai_pipeline(
            trajectory=trajectory,
            timesteps=timesteps,
            xai_analyzer=self.xai_analyzer,
            classifier=self.classifier,
            target_class_id=target_class_id,
            target_class_name=class_name,
            save_results=True,
            results_dir=out_dir
        )

        # Extra artifact: mosaic of ALL diffusion steps
        traj_grid = out_dir / "trajectory_all_steps.png"
        try:
            _save_full_trajectory_grid(trajectory, traj_grid, cols=10, pad=2)
        except Exception as _e:
            # don't fail the whole analysis if mosaic creation fails
            pass

        # Ensure JSON‑serializable return (generator may json.dump this)
        safe = _json_safe(results if results is not None else {})
        if isinstance(safe, dict):
            safe.setdefault("artifacts", {})
            try:
                safe["artifacts"]["trajectory_all_steps"] = str(traj_grid)
            except Exception:
                pass
        return safe

def create_integrated_xai_analyzer(device: Optional[torch.device] = None) -> IntegratedXAIAnalyzer:
    return IntegratedXAIAnalyzer(device=device, verbose=True)

def run_xai_analysis(image_path: str,
                     device: Optional[torch.device] = None,
                     classifier_path: Optional[str] = None,
                     save_dir: Optional[str] = None) -> Tuple[Image.Image, str]:
    """GUI preview: if full artifacts exist, return one of them; otherwise, return original image."""
    img_path = Path(image_path)
    class_name = img_path.parent.name
    base = img_path.parents[2] / 'xai_results' if len(img_path.parents) >= 2 else Path.cwd() / 'xai_results'
    best = None
    cand_dir = base / class_name
    if cand_dir.exists():
        for patt in [f"{img_path.stem}_*/xai_step_*.png", f"{img_path.stem}_*/gradcam_most_important_*.png", f"{img_path.stem}_*/time_shap_analysis.png"]:
            for p in sorted(cand_dir.glob(patt)):
                best = p
                break
            if best:
                break
    if best and best.exists():
        return Image.open(best).convert('RGB'), str(best)
    return Image.open(img_path).convert('RGB'), str(img_path)
