"""
Интеграционный адаптер XAI для ISICGUI.

Экспортируемая функция: run_xai_analysis(image, device=None, classifier_path=None, save_dir=None)
Возвращает: (overlay_pil: PIL.Image, saved_path: pathlib.Path)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Union, Tuple, List

import numpy as np
from PIL import Image
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent


def _ensure_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        try:
            return torch.device(device)
        except Exception:
            pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_model_tensor(img: Union[str, Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
    if isinstance(img, str):
        pil = Image.open(img).convert("RGB")
    elif isinstance(img, Image.Image):
        pil = img.convert("RGB")
    elif isinstance(img, np.ndarray):
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        pil = Image.fromarray(img.astype(np.uint8)) if img.dtype != np.float32 else Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    elif torch.is_tensor(img):
        t = img.detach().clone()
        if t.dim() == 3:
            t = t.unsqueeze(0)
        if t.shape[1] == 1:
            t = t.repeat(1, 3, 1, 1)
        if t.min() >= 0 and t.max() <= 1:
            t = t * 2.0 - 1.0
        if t.shape[-2:] != (128, 128):
            t = F.interpolate(t, size=(128, 128), mode="bilinear", align_corners=False, antialias=True)
        return t
    else:
        raise TypeError("Unsupported image type for XAI integration")

    pil = pil.resize((128, 128))
    arr = np.asarray(pil).astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    arr = np.transpose(arr, (2, 0, 1))
    t = torch.from_numpy(arr).unsqueeze(0)
    return t


def _to_overlay_pil(base_tensor: torch.Tensor, attribution: torch.Tensor) -> Image.Image:
    base = base_tensor.detach().cpu().clone()
    if base.dim() == 4:
        base = base[0]
    base = torch.clamp((base + 1.0) / 2.0, 0, 1)
    base_np = np.transpose(base.numpy(), (1, 2, 0))

    attr = attribution.detach().cpu().clone()
    if attr.dim() == 4:
        attr = attr[0]
    if attr.dim() == 3 and attr.shape[0] == 3:
        attr = torch.linalg.vector_norm(attr, dim=0)
    else:
        attr = torch.abs(attr)
    attr = attr.float()
    attr -= attr.min()
    denom = (attr.max() - attr.min()).clamp(min=1e-8)
    attr = (attr / denom).numpy()

    import matplotlib.cm as cm
    heat = cm.get_cmap("jet")(attr)[..., :3]
    overlay = (0.5 * base_np + 0.5 * heat)
    overlay = np.clip(overlay, 0, 1)
    return Image.fromarray((overlay * 255).astype(np.uint8))


CLASSIFIER_IMAGE_SIZE = 224


class MelanomaClassifierAdaptive(nn.Module):
    def __init__(self, num_classes: int = 8, architecture: str = 'resnet18', pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.architecture = architecture
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_classes)

    def preprocess_for_classifier(self, x: torch.Tensor) -> torch.Tensor:
        if x.device != next(self.parameters()).device:
            x = x.to(next(self.parameters()).device)
        x = torch.clamp((x + 1.0) / 2.0, 0, 1)
        if x.shape[-2:] != (CLASSIFIER_IMAGE_SIZE, CLASSIFIER_IMAGE_SIZE):
            x = F.interpolate(x, size=(CLASSIFIER_IMAGE_SIZE, CLASSIFIER_IMAGE_SIZE), mode='bilinear', align_corners=False, antialias=True)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        x = normalize(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess_for_classifier(x)
        return self.model(x)

    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def get_per_class_score(self, x: torch.Tensor, target_class: int) -> torch.Tensor:
        probs = self.get_probabilities(x)
        return torch.log(probs[:, target_class] + 1e-8)


try:
    from captum.attr import IntegratedGradients
    _CAPTUM_OK = True
except Exception:
    _CAPTUM_OK = False


class ModernXAIAnalyzer:
    def __init__(self, classifier: nn.Module, device: torch.device, verbose: bool = False):
        self.classifier = classifier
        self.device = device
        self.verbose = verbose
        self.ig_method = IntegratedGradients(self._model_wrapper) if _CAPTUM_OK else None

    def _model_wrapper(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

    def _get_baseline(self, image: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(image)

    def compute_integrated_gradients(self, image: torch.Tensor, target_class: int, n_steps: int = 50) -> torch.Tensor:
        image = image.to(self.device)
        if self.ig_method is not None:
            baseline = self._get_baseline(image)
            def target_func(inp):
                return self.classifier.get_per_class_score(inp, target_class)
            original = self.ig_method.forward_func
            self.ig_method.forward_func = target_func
            try:
                attr = self.ig_method.attribute(image, baselines=baseline, n_steps=n_steps, method='riemann_right')
            finally:
                self.ig_method.forward_func = original
            return attr
        image.requires_grad_(True)
        score = self.classifier.get_per_class_score(image, target_class)
        score.backward()
        grad = image.grad.clone()
        image.grad.zero_()
        image.requires_grad_(False)
        return grad

    def compute_shap_approximation(self, image: torch.Tensor, target_class: int, n_samples: int = 128, patch_size: int = 16) -> torch.Tensor:
        image = image.to(self.device)
        batch_size, channels, height, width = image.shape
        n_patches_h = max(1, height // patch_size)
        n_patches_w = max(1, width // patch_size)

        attribution = torch.zeros_like(image)

        # Базовый скор (нулевой baseline)
        baseline_image = torch.zeros_like(image)
        with torch.no_grad():
            baseline_score = self.classifier.get_per_class_score(baseline_image, target_class).item()

        # Сэмплируем маски патчей
        for _ in range(n_samples):
            patch_mask = (torch.rand(n_patches_h, n_patches_w) > 0.5)

            full_mask = torch.zeros(height, width, dtype=torch.bool)
            for i in range(n_patches_h):
                for j in range(n_patches_w):
                    if patch_mask[i, j]:
                        y_start, y_end = i * patch_size, min((i + 1) * patch_size, height)
                        x_start, x_end = j * patch_size, min((j + 1) * patch_size, width)
                        full_mask[y_start:y_end, x_start:x_end] = True

            masked_image = image.clone()
            masked_image[:, :, ~full_mask] = 0

            with torch.no_grad():
                masked_score = self.classifier.get_per_class_score(masked_image, target_class).item()

            contribution = masked_score - baseline_score
            mask_tensor = full_mask.to(self.device).float().unsqueeze(0).unsqueeze(0)
            attribution += contribution * mask_tensor

        attribution /= max(n_samples, 1)
        return attribution

    def compute_combined_attribution(self, image: torch.Tensor, target_class: int, methods=None, weights=None):
        if methods is None:
            methods = ['ig']
        if weights is None:
            weights = [1.0 / len(methods)] * len(methods)
        atts = []
        details = {}
        for m, w in zip(methods, weights):
            if m == 'ig':
                a = self.compute_integrated_gradients(image, target_class)
            elif m == 'shap':
                a = self.compute_shap_approximation(image, target_class)
            elif m == 'gradient':
                # простой градиент как резерв
                image = image.to(self.device)
                image.requires_grad_(True)
                score = self.classifier.get_per_class_score(image, target_class)
                score.backward()
                a = image.grad.clone()
                image.grad.zero_()
                image.requires_grad_(False)
            else:
                # неизвестный метод — пропускаем
                continue
            atts.append(a * w)
            details[m] = {'weight': float(w)}
        combined = torch.stack(atts).sum(dim=0)
        return combined, details


# ————————————————————————————————————————————————
# ОСНОВНОЕ API ДЛЯ GUI
# ————————————————————————————————————————————————

_CACHED = {"classifier": None, "xai": None, "device_str": None}


def _get_classifier(device: torch.device, classifier_path: Path, num_classes: int = 8) -> torch.nn.Module:
    device_str = str(device)
    if _CACHED["classifier"] is not None and _CACHED["device_str"] == device_str:
        return _CACHED["classifier"]

    inferred_num_classes = num_classes
    try:
        if classifier_path.exists():
            ckpt_cpu = torch.load(str(classifier_path), map_location="cpu")
            raw_sd = ckpt_cpu.get("model_state_dict", ckpt_cpu) if isinstance(ckpt_cpu, dict) else ckpt_cpu
            for key, value in raw_sd.items():
                if isinstance(value, torch.Tensor) and key.endswith("fc.weight") and value.dim() == 2:
                    inferred_num_classes = int(value.shape[0])
                    break
    except Exception:
        pass

    model = MelanomaClassifierAdaptive(num_classes=inferred_num_classes, architecture="resnet18", pretrained=True).to(device)
    try:
        if classifier_path.exists():
            ckpt = torch.load(str(classifier_path), map_location=device)
            state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            model_state = model.state_dict()
            compatible = {k: v for k, v in state_dict.items() if k in model_state and model_state[k].shape == v.shape}
            if compatible:
                model.load_state_dict(compatible, strict=False)
    except Exception:
        pass
    model.eval()
    _CACHED["classifier"] = model
    _CACHED["device_str"] = device_str
    return model


def _get_xai_analyzer(classifier: nn.Module, device: torch.device) -> ModernXAIAnalyzer:
    device_str = str(device)
    if _CACHED["xai"] is not None and _CACHED["device_str"] == device_str:
        return _CACHED["xai"]
    analyzer = ModernXAIAnalyzer(classifier=classifier, device=device, verbose=False)
    _CACHED["xai"] = analyzer
    _CACHED["device_str"] = device_str
    return analyzer


def run_xai_analysis(
    image: Union[str, Image.Image, np.ndarray, torch.Tensor],
    *,
    device: Optional[Union[str, torch.device]] = None,
    classifier_path: Optional[Union[str, Path]] = None,
    target_class_id: Optional[int] = None,
    save_dir: Optional[Union[str, Path]] = None,
    methods: Optional[List[str]] = None,
    weights: Optional[List[float]] = None,
    shap_samples: int = 128,
    shap_patch_size: int = 16,
) -> Tuple[Image.Image, Path]:
    dev = _ensure_device(device)
    model_path = Path(classifier_path) if classifier_path else _PROJECT_ROOT / "checkpoints" / "classifier.pth"
    out_dir = Path(save_dir) if save_dir else _PROJECT_ROOT / "xai_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    x = _to_model_tensor(image).to(dev)
    classifier = _get_classifier(dev, model_path)
    analyzer = _get_xai_analyzer(classifier, dev)

    methods = methods or ["ig", "shap"]
    if weights is None:
        weights = [1.0 / len(methods)] * len(methods)

    if target_class_id is None:
        with torch.no_grad():
            logits = classifier(x)
            target_class_id = int(torch.argmax(logits, dim=1).item())

    if hasattr(analyzer, "compute_shap_approximation"):
        orig_fn = analyzer.compute_shap_approximation
        def shap_with_params(img, tc):
            return orig_fn(img, tc, n_samples=shap_samples, patch_size=shap_patch_size)
        analyzer.compute_shap_approximation = shap_with_params  # type: ignore

    attribution, _ = analyzer.compute_combined_attribution(
        x, target_class_id, methods=methods, weights=weights
    )

    overlay = _to_overlay_pil(x, attribution)
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"xai_overlay_{time_str}.png"
    if isinstance(image, str):
        base_name = f"xai_{Path(image).stem}_{time_str}.png"
    save_path = out_dir / base_name
    overlay.save(save_path)
    return overlay, save_path


__all__ = [
    "run_xai_analysis",
]



