"""
run_experiments.py

CLI сценарий для задач:
--task {attribution,time_curves,latents,counterfactuals,checks}

Сохраняет артефакты в ./outputs/{task}/{timestamp}/:
- Heatmaps (IG/SHAP/Grad-CAM/Attention Rollout)
- Кривые важности vs t (суммарная релевантность)
- Проекции латентов (png + json метрик)
- Контрафакты и δ-гистограммы
- Логи проверок (json)

Пример:
python run_experiments.py --task attribution --root /data/derm --checkpoint path/to/unet.pth --split val --n_samples 8 --baseline mean --m 20

Notes
-----
- Для синтетических данных: добавьте --synthetic (установит DERM_SYNTHETIC=1).
"""

from __future__ import annotations

import os
import argparse
import json
from typing import List, Dict, Tuple

import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from utils import set_seeds, timestamp, ensure_dir, save_heatmap, save_json
from datasets import make_dataloaders, CLASSES
from train_or_load import load_or_init_unet, load_scheduler
from model_ddpm import AttentionHooks
from attribution_ig import integrated_gradients, trajectory_ig
from attribution_shap import shap_patches, time_shap
from attribution_gradcam import gradcam, attn_grad_weighted
from attention_rollout import attention_rollout
from latent_viz import extract_latents, project_latents, cluster_metrics
from counterfactuals import optimize_counterfactual_xT, optimize_counterfactual_xt, batch_effect_delta
from evaluation import (
    sanity_checks,
    check_ig_completeness,
    check_shap_efficiency,
    methods_consistency,
    counterfactual_stability,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, required=True,
                   choices=["attribution", "time_curves", "latents", "counterfactuals", "checks"])
    p.add_argument("--root", type=str, default="./data")
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--synthetic", action="store_true", help="использовать синтетические данные (если нет датасета)")
    # атрибуции
    p.add_argument("--baseline", type=str, default="mean", choices=["mean", "zeros", "noise"])
    p.add_argument("--m", type=int, default=20, help="число точек IG")
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--num_samples", type=int, default=512, help="число выборок для SHAP")
    p.add_argument("--t_list", type=int, nargs="*", default=[50, 250, 500, 750, 999])
    p.add_argument("--agg", type=str, default="mean", choices=["mean", "median"])
    # бюджеты/оптимизация
    p.add_argument("--eps", type=float, default=8/255)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--lp", type=str, default="l2", choices=["l2", "linf"])
    p.add_argument("--tv_weight", type=float, default=0.0)
    p.add_argument("--target_class", type=int, default=0)
    p.add_argument("--n_samples", type=int, default=16)
    return p.parse_args()


def _to_vis_np(x: torch.Tensor) -> np.ndarray:
    """
    x: (B,3,H,W) in [-1,1] or normalized approx
    returns uint8 RGB array (B,H,W,3)
    """
    x = x.detach().cpu()
    # assume dataset normalization to [-1,1]; bring to [0,1]
    x = (x * 0.5) + 0.5
    x = x.clamp(0, 1).permute(0, 2, 3, 1).numpy()
    x = (x * 255).astype("uint8")
    return x


def _save_lineplot(xy_list: List[Tuple[int, float]], path: str, title: str, ylabel: str = "relevance"):
    xs = [a for a, _ in xy_list]
    ys = [b for _, b in xy_list]
    plt.figure(figsize=(6, 4))
    sns.lineplot(x=xs, y=ys, marker="o")
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _save_scatter(Y: np.ndarray, labels: np.ndarray, path: str, title: str):
    plt.figure(figsize=(6, 6))
    palette = sns.color_palette("tab10", n_colors=len(CLASSES))
    for i, c in enumerate(CLASSES):
        idx = (labels == i)
        plt.scatter(Y[idx, 0], Y[idx, 1], s=16, alpha=0.8, label=c, color=palette[i % len(palette)])
    plt.legend(markerscale=1.5, fontsize=8, frameon=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    args = parse_args()

    if args.synthetic:
        os.environ["DERM_SYNTHETIC"] = "1"

    set_seeds(args.seed)
    device = torch.device(args.device)

    out_dir = os.path.join("./outputs", args.task, timestamp())
    ensure_dir(out_dir)

    dls = make_dataloaders(args.root, img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers)
    dl = dls.get(args.split, dls["val"])

    unet = load_or_init_unet(args.checkpoint, img_size=args.img_size).to(device).eval()
    scheduler = load_scheduler()

    # простая функция релевантности: скаляризация выхода UNet для фиксированного t
    def f_score(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            t0 = torch.tensor([args.t_list[0]] * x.size(0), dtype=torch.long, device=x.device)
            y = unet(x, t0).sample
        return y.mean(dim=(1, 2, 3))  # (B,)

    hooks = AttentionHooks(unet)

    # collect a small batch
    xs_list, ys_list = [], []
    for xb, yb in dl:
        xs_list.append(xb)
        ys_list.append(yb)
        if sum(len(z) for z in xs_list) >= args.n_samples:
            break
    x = torch.cat(xs_list, dim=0)[:args.n_samples].to(device)
    y = torch.cat(ys_list, dim=0)[:args.n_samples].to(device)
    x_vis = _to_vis_np(x)  # (B,H,W,3) uint8

    if args.task == "attribution":
        # IG
        IG = integrated_gradients(x, f_score, x0=args.baseline, m=args.m).detach().cpu()  # (B,C,H,W)
        IG_map = IG.abs().mean(dim=1).numpy()  # (B,H,W)
        for i in range(IG_map.shape[0]):
            save_heatmap(x_vis[i], IG_map[i], os.path.join(out_dir, f"ig_{i:03d}.png"),
                         title=f"IG baseline={args.baseline}")

        # SHAP
        phi_up, phi_p = shap_patches(
            x, f_score, patch_size=args.patch_size, num_samples=args.num_samples, baseline=args.baseline
        )
        phi_map = phi_up.abs().mean(dim=1).cpu().numpy()
        for i in range(phi_map.shape[0]):
            save_heatmap(x_vis[i], phi_map[i], os.path.join(out_dir, f"shap_{i:03d}.png"),
                         title=f"SHAP baseline={args.baseline}")

        # Grad-CAM (используем mid_block фичи)
        def feature_getter_wrap():
            fm = hooks.get_feature_maps().get("mid_block", None)
            if fm is None:
                # прогон для заполнения
                t0 = torch.tensor([args.t_list[0]] * x.size(0), dtype=torch.long, device=device)
                with hooks.capture():
                    _ = unet(x, t0)
                fm = hooks.get_feature_maps().get("mid_block")
            fm.requires_grad_(True)
            return fm

        # прогон с capture для заполнения хуков
        t0 = torch.tensor([args.t_list[0]] * x.size(0), dtype=torch.long, device=device)
        with hooks.capture():
            _ = unet(x, t0)

        G = gradcam(x, f_score, feature_getter_wrap)
        G_map = G.detach().cpu().numpy()
        for i in range(G_map.shape[0]):
            save_heatmap(x_vis[i], G_map[i], os.path.join(out_dir, f"gradcam_{i:03d}.png"),
                         title="Grad-CAM mid_block")

        # Attention grad-weighted
        def attn_getter():
            return hooks.get_attentions()

        Aheat = attn_grad_weighted(x, f_score, attn_getter, head_agg="mean")
        A_map = Aheat.detach().cpu().numpy()
        for i in range(A_map.shape[0]):
            save_heatmap(x_vis[i], A_map[i], os.path.join(out_dir, f"attn_grad_{i:03d}.png"),
                         title="Attention grad-weighted")

        # Attention Rollout (агрегация по слоям; head-agg mean)
        # Сбор внимания как np массивов
        atts_dict = hooks.get_attentions()  # {layer_name: (B,H,HW,HW)}
        attn_stack = []
        # Сортируем слои по порядку имен для детерминизма
        for k in sorted(atts_dict.keys()):
            attn_stack.append(atts_dict[k].detach().cpu().numpy())
        if len(attn_stack) > 0:
            A_eff = attention_rollout(attn_stack, add_identity=True, normalize="row", head_agg="mean")  # (B,HW,HW)
            # Начальный вектор единичный -> итоговое распределение внимания
            B, HW, _ = A_eff.shape
            weights = (A_eff @ (np.ones((B, HW, 1), dtype=np.float32) / float(HW))).squeeze(-1)  # (B,HW)
            # Найдём ближайшее пространственное разрешение из mid_block
            mid_fm = hooks.get_feature_maps().get("mid_block", None)
            if mid_fm is not None:
                Hm, Wm = mid_fm.shape[-2], mid_fm.shape[-1]
            else:
                # без фич — предполагаем sqrt(HW)
                side = int(np.sqrt(HW))
                Hm = Wm = side
            weights_maps = weights.reshape(B, Hm, Wm)
            # апсемпл до img_size
            weights_t = torch.from_numpy(weights_maps).unsqueeze(1).float()
            weights_up = torch.nn.functional.interpolate(
                weights_t, size=(args.img_size, args.img_size), mode="bilinear", align_corners=False
            ).squeeze(1).numpy()
            for i in range(B):
                save_heatmap(x_vis[i], weights_up[i], os.path.join(out_dir, f"attn_rollout_{i:03d}.png"),
                             title="Attention Rollout")

        print(f"[attribution] Saved IG/SHAP/Grad-CAM/Attention maps to {out_dir}")

    elif args.task == "time_curves":
        # По времени t считаем IG и SHAP
        def f_t(xi: torch.Tensor, tt: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                y = unet(xi, tt).sample
            return y.mean(dim=(1, 2, 3))  # (B,)

        def x0_t_fn(xi: torch.Tensor, tt: torch.Tensor) -> torch.Tensor:
            # бейзлайн: zeros
            return torch.zeros_like(xi)

        ig_res = trajectory_ig(x, f_t, x0_t_fn, t_list=args.t_list, m=max(8, args.m // 2), agg=args.agg)
        # суммарная важность по t
        curve_ig = []
        for t in args.t_list:
            mval = ig_res[int(t)].abs().mean().item()
            curve_ig.append([int(t), float(mval)])
        save_json({"curve_ig": curve_ig}, os.path.join(out_dir, "time_curves.json"))
        _save_lineplot(curve_ig, os.path.join(out_dir, "time_curve_ig.png"),
                       title=f"IG relevance vs t (baseline={args.baseline})")

        shap_res = time_shap(
            x,
            f_t,
            args.t_list,
            patch_size=args.patch_size,
            num_samples=max(64, args.num_samples // 4),
            baseline=args.baseline,
        )
        curve_shap = []
        for t in args.t_list:
            phi_t = shap_res[int(t)]  # (B,C,H,W)
            curve_shap.append([int(t), float(phi_t.abs().mean().item())])
        # append to same json
        with open(os.path.join(out_dir, "time_curves.json"), "r") as f:
            d = json.load(f)
        d["curve_shap"] = curve_shap
        save_json(d, os.path.join(out_dir, "time_curves.json"))
        _save_lineplot(curve_shap, os.path.join(out_dir, "time_curve_shap.png"),
                       title=f"SHAP relevance vs t (baseline={args.baseline})")

        # Сохраним несколько heatmaps для минимального/максимального t
        min_t, max_t = int(args.t_list[0]), int(args.t_list[-1])
        ig_min = ig_res[min_t].abs().mean(dim=1).detach().cpu().numpy()
        ig_max = ig_res[max_t].abs().mean(dim=1).detach().cpu().numpy()
        phi_min = shap_res[min_t].abs().mean(dim=1).detach().cpu().numpy()
        phi_max = shap_res[max_t].abs().mean(dim=1).detach().cpu().numpy()
        for i in range(min(ig_min.shape[0], 8)):
            save_heatmap(x_vis[i], ig_min[i], os.path.join(out_dir, f"ig_t{min_t}_{i:03d}.png"),
                         title=f"IG t={min_t} baseline={args.baseline}")
            save_heatmap(x_vis[i], ig_max[i], os.path.join(out_dir, f"ig_t{max_t}_{i:03d}.png"),
                         title=f"IG t={max_t} baseline={args.baseline}")
            save_heatmap(x_vis[i], phi_min[i], os.path.join(out_dir, f"shap_t{min_t}_{i:03d}.png"),
                         title=f"SHAP t={min_t} baseline={args.baseline}")
            save_heatmap(x_vis[i], phi_max[i], os.path.join(out_dir, f"shap_t{max_t}_{i:03d}.png"),
                         title=f"SHAP t={max_t} baseline={args.baseline}")

        print(f"[time_curves] Saved curves and sample heatmaps to {out_dir}")

    elif args.task == "latents":
        # Извлечь латенты по t, проектировать и оценить кластеризацию
        t_list = args.t_list
        lat_dict = extract_latents(unet, x, t_list=t_list, mode="bottleneck")
        # Для простоты — проектируем конкатенацию по t
        Xs = []
        for t in t_list:
            z_t = lat_dict[int(t)]  # (B, D_t or HxWxc)
            z_t = z_t.reshape(z_t.shape[0], -1)
            Xs.append(z_t.detach().cpu().numpy())
        X = np.concatenate(Xs, axis=1)  # (B, sum_dims)
        Y = project_latents(X, method="umap", pca_dim=64, out_dim=2, seed=args.seed)
        mets = cluster_metrics(Y, labels=y.detach().cpu().numpy())
        _save_scatter(Y, y.detach().cpu().numpy(), os.path.join(out_dir, "latents_umap.png"),
                      title="Latent UMAP projection")
        save_json({"metrics": mets, "classes": CLASSES, "t_list": [int(t) for t in t_list]},
                  os.path.join(out_dir, "latents_metrics.json"))
        # также сохраним сами координаты
        np.save(os.path.join(out_dir, "latents_coords.npy"), Y)
        print(f"[latents] Saved projection and metrics to {out_dir}")

    elif args.task == "counterfactuals":
        # Контрафакты в пиксельном пространстве (x_T)
        # Маска не задана — оптимизируем по всему изображению
        x_cf = optimize_counterfactual_xT(
            x, f_score, y_target=args.target_class, mask=None,
            lp=args.lp, eps=args.eps, lam=1.0, iters=args.iters, tv_weight=args.tv_weight
        ).detach()
        # Сохраняем сравнение
        x_cf_vis = _to_vis_np(x_cf)
        for i in range(min(x.shape[0], 16)):
            # Сохраняем side-by-side
            canvas = Image.new("RGB", (args.img_size * 2, args.img_size))
            canvas.paste(Image.fromarray(x_vis[i]), (0, 0))
            canvas.paste(Image.fromarray(x_cf_vis[i]), (args.img_size, 0))
            canvas.save(os.path.join(out_dir, f"cf_pair_{i:03d}.png"))
        # Дельта и пермутационный тест
        stats = batch_effect_delta(f_score, x, x_cf)
        save_json({"delta_stats": stats}, os.path.join(out_dir, "counterfactuals_stats.json"))

        # Гистограмма дельт
        deltas = np.asarray(stats.get("delta", []), dtype=float)
        if deltas.size > 0:
            plt.figure(figsize=(5, 3))
            sns.histplot(deltas, bins=20, kde=True)
            plt.title("Δ distribution")
            plt.xlabel("f(x_cf) - f(x)")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "delta_hist.png"), dpi=150)
            plt.close()

        print(f"[counterfactuals] Saved counterfactuals and δ stats to {out_dir}")

    elif args.task == "checks":
        # Санити-чеки и согласованность методов
        report = {}

        # Атрибутивные функции-обёртки для sanity_checks
        def fn_ig(xb: torch.Tensor) -> torch.Tensor:
            IGb = integrated_gradients(xb, f_score, x0=args.baseline, m=max(8, args.m // 2))
            return IGb.abs().mean(dim=1)  # (B,H,W)

        def fn_shap(xb: torch.Tensor) -> torch.Tensor:
            phi_up_b, _ = shap_patches(
                xb, f_score, patch_size=args.patch_size, num_samples=max(128, args.num_samples // 2),
                baseline=args.baseline
            )
            return phi_up_b.abs().mean(dim=1)

        def fn_gradcam(xb: torch.Tensor) -> torch.Tensor:
            # ensure hooks populated
            t0 = torch.tensor([args.t_list[0]] * xb.size(0), dtype=torch.long, device=xb.device)
            with hooks.capture():
                _ = unet(xb, t0)
            def _fg():
                fm = hooks.get_feature_maps().get("mid_block")
                fm.requires_grad_(True)
                return fm
            G = gradcam(xb, f_score, _fg)
            return G

        # базовые sanity_checks (рандомизация слоёв и деградация)
        report["sanity"] = sanity_checks(unet, [fn_ig, fn_shap, fn_gradcam])

        # IG completeness
        IG = integrated_gradients(x, f_score, x0=args.baseline, m=args.m)
        with torch.no_grad():
            f_x = f_score(x).detach().cpu()
            # сформируем x0
            if args.baseline == "zeros":
                x0 = torch.zeros_like(x)
            elif args.baseline == "mean":
                x0 = x.mean(dim=(2, 3), keepdim=True).expand_as(x) * 0.0  # mean->0 for standardized data
            else:
                torch.manual_seed(args.seed)
                x0 = torch.randn_like(x) * 0.0
            f_x0 = f_score(x0).detach().cpu()
        report["ig_completeness"] = check_ig_completeness(IG.detach().cpu(), f_x, f_x0, tol=0.05)

        # SHAP efficiency
        phi_up, _ = shap_patches(x, f_score, patch_size=args.patch_size, num_samples=args.num_samples,
                                 baseline=args.baseline)
        phi_sum = phi_up.sum(dim=(1, 2, 3)).detach().cpu()  # (B,)
        report["shap_efficiency"] = check_shap_efficiency(phi_sum, f_x, f_x0, tol=0.1)

        # Spearman между методами
        # получаем ранги (векторизованные карты)
        def flatten_rank(t: torch.Tensor) -> np.ndarray:
            a = t.detach().cpu().numpy().reshape(t.shape[0], -1)
            return a
        maps_ig = fn_ig(x)
        maps_shap = fn_shap(x)
        maps_gc = fn_gradcam(x)
        ranks_list = [flatten_rank(maps_ig), flatten_rank(maps_shap), flatten_rank(maps_gc)]
        report["methods_consistency"] = methods_consistency(ranks_list)

        # Устойчивость δ (контрафакты vs контроль)
        x_cf = optimize_counterfactual_xT(
            x, f_score, y_target=args.target_class, mask=None, lp=args.lp, eps=args.eps,
            lam=1.0, iters=max(50, args.iters // 2), tv_weight=args.tv_weight
        ).detach()
        # Контроль: перестановка батча
        perm = torch.randperm(x.size(0), device=x.device)
        x_ctrl = x[perm]
        report["counterfactual_stability"] = counterfactual_stability(f_score, x, x_cf, x_ctrl)

        save_json(report, os.path.join(out_dir, "checks.json"))
        print(f"[checks] Saved checks report to {out_dir}")

    else:
        raise ValueError(f"Unknown task: {args.task}")

    print("Done.")


if __name__ == "__main__":
    main()