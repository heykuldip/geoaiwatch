#!/usr/bin/env python
"""Train one (fold, seed) cell of the RGB vs RGB+NIR comparison.

    python posts/2026-11-near-infrared/scripts/train.py --fold NW --seed 0 --smoke
    python posts/2026-11-near-infrared/scripts/train.py --fold NW --seed 0 --epochs 120

Both arms train from the same batch stream and differ only in the fourth channel. The
held-out quadrant is evaluated at each epoch in --eval-epochs, since there is no validation
split, and each evaluation stores the full confusion matrix. mIoU is strict: undefined unless
every class has support in the cell. METHOD.md describes the design.
"""
from __future__ import annotations

import argparse
import json
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from geoaiwatch import config as C
from geoaiwatch.data import (AblationDataset, FrameCache, assert_contract, build_manifest,
                             fold_stats)

POST = Path(__file__).resolve().parents[1]
ART = POST / "artifacts"
RUNS = POST / "runs"

ARMS = ("rgbn", "rgb0")
DEFAULT_EVAL_EPOCHS = "20,40,80,120"
PRIMARY_EVAL_EPOCH = 120              # declared before any held-out evaluation

# smallest per-class IoU difference of practical interest
MIN_MEANINGFUL_DELTA_IOU = 0.02

# confusion diagnostic. Expectation: NIR should help vegetation vs non-vegetated
# but not necessarily agriculture vs forest, since both are vegetated and a
# single date carries no phenology.
CONFUSION_PAIRS = [("agriculture", "forest_veg"), ("forest_veg", "agriculture")]


# determinism

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False      # no autotuning between runs
        torch.backends.cudnn.deterministic = True


# model

def build_model(n_classes: int, seed: int) -> nn.Module:
    """ResNet-34 U-Net with an ImageNet encoder and four input channels.

    smp copies the pretrained first-layer filters cyclically (the fourth channel takes the red
    filters) and scales all four by 3/4, so both arms start from the same weights; in the RGB arm
    the fourth channel is zero.
    """
    import segmentation_models_pytorch as smp

    seed_all(seed)                                  # identical init across arms
    return smp.Unet("resnet34", encoder_weights="imagenet",
                    in_channels=4, classes=n_classes)


def first_conv_report(model: nn.Module) -> dict:
    """Record how smp adapted the pretrained first convolution to four channels."""
    w = model.encoder.conv1.weight.detach()
    return {"shape": list(w.shape),
            "abs_mean_per_channel": [round(float(v), 6)
                                     for v in w.abs().mean(dim=(0, 2, 3))]}


# metrics

def confusion(pred: torch.Tensor, true: torch.Tensor, k: int) -> np.ndarray:
    """Rows = truth, columns = prediction. IGNORE_INDEX excluded."""
    keep = true != C.IGNORE_INDEX
    t, p = true[keep], pred[keep]
    return (torch.bincount(t * k + p, minlength=k * k)
            .reshape(k, k).cpu().numpy().astype(np.int64))


def metrics(cm: np.ndarray, classes: list[str]) -> dict:
    """Everything derived from one confusion matrix; mIoU is None unless every class has support."""
    tp = np.diag(cm).astype(float)
    sup = cm.sum(1).astype(float)              # truth totals
    prd = cm.sum(0).astype(float)
    union = sup + prd - tp

    iou = np.divide(tp, union, out=np.full(len(classes), np.nan),
                    where=(sup > 0) & (union > 0))

    out = {
        "iou": {c: (None if not np.isfinite(v) else float(v))
                for c, v in zip(classes, iou)},
        "support": {c: int(s) for c, s in zip(classes, sup)},
        "eligible": {c: bool(s >= C.frozen(ART)["min_test_support_pixels"])
                     for c, s in zip(classes, sup)},
        "miou_strict": (float(iou.mean()) if np.all(sup > 0)
                        and np.all(np.isfinite(iou)) else None),
        "confusion": cm.tolist(),
    }

    # agriculture <-> forest diagnostic
    conf = {}
    for a, b in CONFUSION_PAIRS:
        i, j = classes.index(a), classes.index(b)
        conf[f"{a}_to_{b}"] = float(cm[i, j] / sup[i]) if sup[i] > 0 else None
    out["confusion_rate"] = conf
    return out


def paired_delta(rn: dict, rz: dict, classes: list[str]) -> dict:
    d = {}
    for c in classes:
        a, b = rn["iou"][c], rz["iou"][c]
        d[c] = None if (a is None or b is None) else float(a - b)
    meaningful = {c: (None if v is None else bool(abs(v) >= MIN_MEANINGFUL_DELTA_IOU))
                  for c, v in d.items()}
    dm = (None if rn["miou_strict"] is None or rz["miou_strict"] is None
          else float(rn["miou_strict"] - rz["miou_strict"]))
    return {"delta_iou": d, "meaningful": meaningful, "delta_miou_strict": dm}


# evaluation

@torch.no_grad()
def evaluate(model, loader, arm, k, device, classes):
    """Per-cell and aggregate metrics. One frame = one test cell."""
    model.eval()
    agg = np.zeros((k, k), dtype=np.int64)
    cells = []
    for b in loader:
        x = b[arm].to(device, non_blocking=True)
        y = b["y"].to(device, non_blocking=True)
        pred = model(x).argmax(1)
        cm = confusion(pred, y, k)
        agg += cm
        cells.append({"cube": b["cube"][0], "date": b["date"][0],
                      **metrics(cm, classes)})
    return metrics(agg, classes), cells


def checkpoint_record(models, loader, k, device, classes) -> dict:
    rec = {"arms": {}}
    for arm in ARMS:
        aggm, cells = evaluate(models[arm], loader, arm, k, device, classes)
        rec["arms"][arm] = {**aggm, "per_cell": cells}

    rec.update(paired_delta(rec["arms"]["rgbn"], rec["arms"]["rgb0"], classes))

    # per-cell paired deltas, keyed by (cube, date), the unit of the QA-status
    # heterogeneity analysis
    zn = {(c["cube"], c["date"]): c for c in rec["arms"]["rgbn"]["per_cell"]}
    zz = {(c["cube"], c["date"]): c for c in rec["arms"]["rgb0"]["per_cell"]}
    rec["per_cell_delta"] = [
        {"cube": key[0], "date": key[1], **paired_delta(zn[key], zz[key], classes)}
        for key in sorted(zn)
    ]
    return rec


def run(args) -> dict:
    """Train both arms; record any warning that an operation has no deterministic version."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = _run(args)
    msgs = sorted({str(w.message) for w in caught})
    for m in msgs:
        print(f"warning: {m}")
    res["nondeterministic_ops"] = [m for m in msgs if "deterministic" in m]
    return res


def _run(args) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frozen = C.frozen(ART)
    classes = frozen["retained_classes"]
    k = len(classes)

    eval_at = sorted({int(e) for e in args.eval_epochs.split(",") if e.strip()}
                     | {args.epochs})
    eval_at = [e for e in eval_at if 1 <= e <= args.epochs]

    print(f"fold {args.fold}  seed {args.seed}  device {device}")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")
    print(f"evaluating at epochs {eval_at}")

    # data
    man = build_manifest(frozen["eligible_cubes"])
    if args.max_frames:
        man = man[: args.max_frames]
    print(f"manifest {len(man)} frames")

    t0 = time.time()
    cache = FrameCache(man)
    mean, std = fold_stats(cache, args.fold)
    print(f"cache + stats {time.time() - t0:.1f}s")
    print(f"  mean {mean.round(4)}\n  std  {std.round(4)}")

    tr = AblationDataset(cache, args.fold, classes, mean, std, train=True,
                         crop=args.crop, crops_per_frame=args.crops_per_frame,
                         seed=args.seed)
    te = AblationDataset(cache, args.fold, classes, mean, std, train=False,
                         seed=args.seed, augment=False)

    g = torch.Generator().manual_seed(args.seed)
    tl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, generator=g,
                    num_workers=args.workers,
                    pin_memory=(device.type == "cuda"), drop_last=True)
    el = DataLoader(te, batch_size=1, shuffle=False, num_workers=0)

    assert_contract(next(iter(tl)), classes)
    print("contract OK")

    # two models, identical init
    models, opts = {}, {}
    for arm in ARMS:
        m = build_model(k, args.seed).to(device)
        models[arm] = m
        opts[arm] = torch.optim.AdamW(m.parameters(), lr=args.lr,
                                      weight_decay=args.wd)

    conv1 = first_conv_report(models["rgbn"])
    print(f"conv1 {tuple(conv1['shape'])}  |w| per channel "
          f"{conv1['abs_mean_per_channel']}")

    ref = dict(models["rgbn"].named_parameters())
    for n, p in models["rgb0"].named_parameters():
        assert torch.equal(p, ref[n]), f"arms differ at init: {n}"
    print("arms identical at initialization")

    crit = nn.CrossEntropyLoss(ignore_index=C.IGNORE_INDEX)
    hist, ckpts = [], {}

    for ep in range(1, args.epochs + 1):
        for m in models.values():
            m.train()
        run_loss = {a: 0.0 for a in ARMS}
        nb = 0
        t1 = time.time()

        for i, b in enumerate(tl):
            if args.max_batches and i >= args.max_batches:
                break
            y = b["y"].to(device, non_blocking=True)
            for arm in ARMS:                         # same batch, both arms
                x = b[arm].to(device, non_blocking=True)
                if arm == "rgb0":
                    assert torch.count_nonzero(x[:, 3]) == 0, "null arm polluted"
                opts[arm].zero_grad(set_to_none=True)
                loss = crit(models[arm](x), y)
                loss.backward()
                opts[arm].step()
                run_loss[arm] += float(loss.detach())
            nb += 1

        losses = {a: run_loss[a] / max(nb, 1) for a in ARMS}
        hist.append({"epoch": ep, "batches": nb, **losses})
        line = (f"epoch {ep}/{args.epochs}  {nb} batches  "
                f"{time.time() - t1:.1f}s  loss  "
                + "  ".join(f"{a} {losses[a]:.4f}" for a in ARMS))

        if ep in eval_at:
            rec = checkpoint_record(models, el, k, device, classes)
            rec["train_loss"] = losses
            ckpts[str(ep)] = rec

            if args.save_models and ep == (PRIMARY_EVAL_EPOCH
                                           if PRIMARY_EVAL_EPOCH in eval_at
                                           else max(eval_at)):
                mdir = RUNS / "models"
                mdir.mkdir(parents=True, exist_ok=True)
                for arm in ARMS:
                    torch.save({
                        "state_dict": models[arm].state_dict(),
                        "arm": arm, "fold": args.fold, "seed": args.seed,
                        "epoch": ep, "classes": classes,
                        "norm_mean": mean.tolist(), "norm_std": std.tolist(),
                        "conv1_init": conv1,
                    }, mdir / f"{args.fold}_s{args.seed}_{arm}_ep{ep}.pt")
                print(f"  saved models to {mdir}", flush=True)

            dm = rec["delta_miou_strict"]
            line += (f"   |  EVAL  dmIoU {dm:+.4f}  "
                     f"rgbn {rec['arms']['rgbn']['miou_strict']:.4f}  "
                     f"rgb0 {rec['arms']['rgb0']['miou_strict']:.4f}"
                     if dm is not None else "   |  EVAL  (strict mIoU undefined)")
        print(line, flush=True)

    peak = (torch.cuda.max_memory_allocated() / 2 ** 30
            if device.type == "cuda" else None)
    if peak:
        print(f"peak GPU memory {peak:.2f} GiB  "
              f"(batch {args.batch_size}, crop {args.crop})")

    # summary across checkpoints
    if PRIMARY_EVAL_EPOCH in eval_at:
        primary = PRIMARY_EVAL_EPOCH
    elif args.smoke:
        primary = min(eval_at, key=lambda e: abs(e - PRIMARY_EVAL_EPOCH))
    else:
        primary = None

    print(f"\n{'epoch':>6}   {'rgbn':>7} {'rgb0':>7} {'delta':>8}   per-class delta "
          f"(* = |d| >= {MIN_MEANINGFUL_DELTA_IOU})")
    for e in eval_at:
        r = ckpts[str(e)]
        pc = "  ".join(
            (f"{c[:4]} {v:+.3f}{'*' if r['meaningful'][c] else ' '}"
             if v is not None else f"{c[:4]}   nan ")
            for c, v in r["delta_iou"].items())
        mark = " *" if e == primary else "  "
        print(f"{e:>6}{mark} {r['arms']['rgbn']['miou_strict']:>7.4f} "
              f"{r['arms']['rgb0']['miou_strict']:>7.4f} "
              f"{r['delta_miou_strict']:>+8.4f}   {pc}")
    print("  * primary checkpoint (declared before any held-out evaluation)")

    pr = ckpts[str(primary)]
    print("\nagriculture <-> forest confusion at primary checkpoint:")
    for arm in ARMS:
        cr = pr["arms"][arm]["confusion_rate"]
        print(f"  {arm}  " + "  ".join(
            f"{k2} {v:.4f}" if v is not None else f"{k2} nan"
            for k2, v in cr.items()))

    return {
        "fold": args.fold,
        "seed": args.seed,
        "classes": classes,
        "n_frames": len(man),
        "epochs": args.epochs,
        "eval_epochs": eval_at,
        "primary_eval_epoch": primary,
        "batch_size": args.batch_size,
        "crop": args.crop,
        "crops_per_frame": args.crops_per_frame,
        "lr": args.lr,
        "wd": args.wd,
        "min_meaningful_delta_iou": MIN_MEANINGFUL_DELTA_IOU,
        "min_test_support_pixels": frozen["min_test_support_pixels"],
        "smoke": args.smoke,
        "max_batches": args.max_batches or None,
        "max_frames": args.max_frames or None,
        "peak_gib": peak,
        "torch_version": torch.__version__,
        "smp_version": __import__("segmentation_models_pytorch").__version__,
        "deterministic": "requested, warn-only",
        "conv1_init": conv1,
        "fold_stats": {"mean": mean.tolist(), "std": std.tolist()},
        "history": hist,
        "checkpoints": ckpts,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", choices=list(C.QUADS), required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--eval-epochs", default=DEFAULT_EVAL_EPOCHS,
                   help="comma-separated; the final epoch is included")
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--crops-per-frame", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=0,
                   help="0 suits this loader: the dataset is in RAM, so workers "
                        "overlap no I/O and their temp-dir cleanup fails "
                        "noisily on NFS")
    p.add_argument("--max-batches", type=int, default=0, help="smoke cap")
    p.add_argument("--max-frames", type=int, default=0, help="smoke cap")
    p.add_argument("--smoke", action="store_true",
                   help="write to runs/smoke/ and mark the result discardable")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--save-models", action="store_true",
                help="save both arms' weights at the primary checkpoint, "
                    "for figure generation")
    args = p.parse_args()
    if not args.smoke and (args.max_batches or args.max_frames):
        p.error("--max-batches and --max-frames are for --smoke runs only")

    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.eval_epochs = "1,2"
        args.max_batches = args.max_batches or 20
        args.max_frames = args.max_frames or 8

    res = run(args)

    tag = "smoke" if args.smoke else "grid_isolated"
    out = args.out or (RUNS / tag / f"{args.fold}_s{args.seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out}")
    if args.smoke:
        print("SMOKE RUN -- diagnostic only, not analysis output")


if __name__ == "__main__":
    main()
