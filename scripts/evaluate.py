"""Evaluate a trained model checkpoint on validation or test splits (Phase 3 & Phase 6).

Usage:
    python scripts/evaluate.py --checkpoint outputs/baseline3_gat/best_model.pt [--split test]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src/ is importable even without editable install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from seismic_ml.config import load_config
from seismic_ml.data.loader import create_dataloader
from seismic_ml.data.processed import FastProcessedDataset
from seismic_ml.data.splits import load_split_table
from seismic_ml.evaluation.calibration import reliability_curve
from seismic_ml.evaluation.ood import (
    area_under_risk_coverage,
    id_quantile_threshold,
    ood_auprc,
    ood_auroc,
    ood_detection_rate,
    risk_coverage_curve,
)
from seismic_ml.features.cnn_encoder import CNNEncoder
from seismic_ml.models.cnn import CNNBaseline
from seismic_ml.models.gat import GAT, GATClassifier
from seismic_ml.paths import project_root_from_config, resolve_paths
from seismic_ml.training.trainer import Trainer
from seismic_ml.utils.logging import configure_logging

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Entry config file.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint path.")
    parser.add_argument("--model", choices=["auto", "cnn", "gat"], default="auto", help="Model type ('auto', 'cnn', 'gat').")
    parser.add_argument("--data-root", default=None, help="Data root directory.")
    parser.add_argument("--output-root", default=None, help="Output root directory.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "ood"], help="Split to evaluate on.")
    parser.add_argument("--collection", default="comcat", help="Collection to evaluate ('comcat' or 'exotic'). Default: 'comcat'.")
    parser.add_argument(
        "--ood-collection",
        default=None,
        help="Optional OOD collection (e.g. 'exotic') to jointly evaluate OOD AUROC, AUPRC, and detection rate against --split.",
    )
    parser.add_argument("--ood-split", default="ood", help="Split name inside --ood-collection (default: 'ood').")
    parser.add_argument(
        "--graph-method",
        choices=["knn", "quadrant_knn", "radius", "gaussian_kernel", "fully_connected"],
        default=None,
        help="Override spatial graph construction method.",
    )
    parser.add_argument(
        "--edge-dim",
        type=int,
        default=None,
        help="Override GATv2 edge_dim (auto-detected from checkpoint when omitted).",
    )
    parser.add_argument("--device", default=None, help="Device to use ('cpu', 'cuda', 'auto').")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for evaluation.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger = configure_logging()

    config = load_config(args.config)
    output_root_override = args.output_root
    if output_root_override is None and args.data_root is not None:
        output_root_override = Path(args.data_root) / "outputs"

    paths = resolve_paths(
        config,
        project_root=project_root_from_config(args.config),
        data_root=args.data_root,
        output_root=output_root_override,
    )

    device_arg = args.device or config.get("training", {}).get("device", "auto")
    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg
    logger.info("Using device: %s", device)

    split_path = paths.processed / "splits" / "s1_temporal" / "events.csv"
    if not split_path.exists():
        logger.error("Split table not found at %s.", split_path)
        return 1
    split_table = load_split_table(split_path)

    processed_data_dir = paths.processed / "pnw" / args.collection
    if not processed_data_dir.exists():
        logger.error("Processed data directory not found at %s.", processed_data_dir)
        return 1

    class_names = config.get("data", {}).get("labels", {}).get("class_names") or [
        "earthquake",
        "anthropogenic",
    ]

    dataset = FastProcessedDataset(
        processed_data_dir,
        split_table,
        splits=[args.split],
        class_names=class_names,
        allow_unlabeled=(args.split == "ood"),
        cache_in_memory=True,
    )
    logger.info("Loaded %d events from split %r (collection: %s)", len(dataset), args.split, args.collection)

    if len(dataset) == 0:
        logger.error("Dataset has 0 events for split %r", args.split)
        return 1

    loader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        class_names=class_names,
        allow_unlabeled=(args.split == "ood"),
        pin_memory=device.startswith("cuda"),
    )

    # Determine model architecture
    model_cfg = config.get("model", {})
    encoder_cfg = model_cfg.get("encoder", {})
    embedding_dim = int(encoder_cfg.get("embedding_dim") or 128)
    in_channels = int(encoder_cfg.get("in_channels") or 3)
    input_repr = encoder_cfg.get("input_representation") or "waveform"
    station_pooling = model_cfg.get("station_pooling") or "attention"

    detected_edge_dim = int((model_cfg.get("gat") or {}).get("edge_dim") or 0)
    if args.checkpoint.exists():
        try:
            raw_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            ckpt_sd = raw_ckpt.get("model_state_dict", raw_ckpt)
            if "gat.layers.0.lin_edge.weight" in ckpt_sd:
                detected_edge_dim = int(ckpt_sd["gat.layers.0.lin_edge.weight"].shape[1])
                logger.info("Auto-detected edge_dim=%d from checkpoint.", detected_edge_dim)
            if "encoder.fusion.out_proj.weight" in ckpt_sd:
                input_repr = "dual"
                logger.info("Auto-detected dual-stream encoder from checkpoint.")
            elif "encoder.spec_branch.net.0.weight" in ckpt_sd and "encoder.init_conv.0.weight" not in ckpt_sd:
                input_repr = "spectrogram"
                logger.info("Auto-detected spectrogram encoder from checkpoint.")
        except Exception as exc:
            logger.warning("Could not inspect checkpoint architecture: %s", exc)

    encoder = CNNEncoder(
        embedding_dim=embedding_dim,
        in_channels=in_channels,
        input_representation=input_repr,
        pool_time=True,
    )

    ckpt_str = str(args.checkpoint).lower()
    if args.model == "auto":
        is_gat = "gat" in ckpt_str or bool(model_cfg.get("gat", {}).get("enabled", False))
    else:
        is_gat = (args.model == "gat")

    is_edl = "edl" in ckpt_str or model_cfg.get("head", {}).get("type") == "evidential"
    head_type = "evidential" if is_edl else "softmax"

    if is_gat:
        gat_cfg = model_cfg.get("gat", {})
        graph_cfg = model_cfg.get("graph", {})
        spatial_cfg = graph_cfg.get("spatial", {})
        hidden_dim = int(gat_cfg.get("hidden_dim") or 128)
        num_heads = int(gat_cfg.get("num_heads") or 4)
        num_layers = int(gat_cfg.get("num_layers") or 2)
        dropout = float(gat_cfg.get("dropout") or 0.1)

        if args.edge_dim is not None:
            detected_edge_dim = int(args.edge_dim)

        graph_method = args.graph_method or spatial_cfg.get("method") or "knn"
        vp_km_s = float(spatial_cfg.get("vp_km_s") or 6.2)

        gat = GAT(
            in_dim=embedding_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            edge_dim=detected_edge_dim,
        )
        model = GATClassifier(
            encoder=encoder,
            gat=gat,
            num_classes=len(class_names),
            station_pooling=station_pooling,
            graph_method=graph_method,
            k_neighbors=int(spatial_cfg.get("k_neighbors") or 5),
            radius_km=spatial_cfg.get("radius_km"),
            sigma_km=spatial_cfg.get("sigma_km"),
            head_type=head_type,
            vp_km_s=vp_km_s,
        )
    else:
        model = CNNBaseline(
            encoder=encoder,
            num_classes=len(class_names),
            station_pooling=station_pooling,
            head_type=head_type,
        )

    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        class_names=class_names,
    )

    logger.info("Loading checkpoint from %s ...", args.checkpoint)
    trainer.load_checkpoint(args.checkpoint)

    logger.info("Evaluating on split %r ...", args.split)
    metrics = trainer.evaluate(loader)

    # Optional joint OOD benchmark (Phase 9)
    if args.ood_collection:
        ood_data_dir = paths.processed / "pnw" / args.ood_collection
        if ood_data_dir.exists():
            ood_dataset = FastProcessedDataset(
                ood_data_dir,
                split_table,
                splits=[args.ood_split],
                class_names=class_names,
                allow_unlabeled=True,
                cache_in_memory=True,
            )
            if len(ood_dataset) > 0:
                logger.info(
                    "Loaded %d OOD events from split %r (collection: %s)",
                    len(ood_dataset),
                    args.ood_split,
                    args.ood_collection,
                )
                ood_loader = create_dataloader(
                    ood_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    class_names=class_names,
                    allow_unlabeled=True,
                    pin_memory=device.startswith("cuda"),
                )
                ood_metrics = trainer.evaluate(ood_loader)
                import numpy as np

                in_scores = (
                    np.array(metrics["uncertainties"], dtype=np.float64)
                    if "uncertainties" in metrics and len(metrics["uncertainties"]) > 0
                    else 1.0 - np.max(np.array(metrics["probabilities"], dtype=np.float64), axis=1)
                )
                ood_scores = (
                    np.array(ood_metrics["uncertainties"], dtype=np.float64)
                    if "uncertainties" in ood_metrics and len(ood_metrics["uncertainties"]) > 0
                    else 1.0 - np.max(np.array(ood_metrics["probabilities"], dtype=np.float64), axis=1)
                )
                tau_95 = id_quantile_threshold(in_scores, id_tpr=0.95)
                metrics["ood_num_events"] = int(len(ood_scores))
                metrics["ood_mean_uncertainty"] = float(np.mean(ood_scores))
                metrics["ood_auroc"] = ood_auroc(in_scores, ood_scores)
                metrics["ood_auprc"] = ood_auprc(in_scores, ood_scores)
                metrics["ood_threshold_95id"] = tau_95
                metrics["ood_detection_rate_95id"] = ood_detection_rate(ood_scores, threshold=tau_95)
        else:
            logger.warning("OOD collection directory not found at %s; skipping joint OOD metrics.", ood_data_dir)

    print("\n" + "=" * 45)
    print(f"Evaluation Results on split: {args.split}")
    print("=" * 45)
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"  {k:25s}: {v:.4f}")
    print("=" * 45 + "\n")

    out_dir = args.checkpoint.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float, list))}
    metrics_json_path = out_dir / f"evaluation_metrics_{args.split}.json"
    metrics_json_path.write_text(json.dumps(summary_metrics, indent=2), encoding="utf-8")
    logger.info("Saved evaluation metrics JSON to %s", metrics_json_path)

    # Save confusion matrix, calibration, risk-coverage, and uncertainty plots for reports
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from seismic_ml.utils.visualization import (
            plot_confusion_matrix,
            plot_reliability_diagram,
            plot_risk_coverage_curve,
            plot_uncertainty_distribution,
        )

        cm = np.array(metrics.get("confusion_matrix", []))
        if cm.size > 0:
            cm_fig = plot_confusion_matrix(cm, class_names)
            cm_path = out_dir / f"confusion_matrix_{args.split}.png"
            cm_fig.savefig(cm_path, dpi=200, bbox_inches="tight")
            plt.close(cm_fig)
            logger.info("Saved confusion matrix plot to %s", cm_path)

        if "probabilities" in metrics and "targets" in metrics and len(metrics["targets"]) > 0:
            probs_np = np.array(metrics["probabilities"], dtype=np.float64)
            targets_np = np.array(metrics["targets"], dtype=np.int64)
            rel = reliability_curve(probs_np, targets_np, num_bins=10)
            rel_fig = plot_reliability_diagram(rel.bin_confidence, rel.bin_accuracy, rel.bin_counts)
            rel_path = out_dir / f"reliability_diagram_{args.split}.png"
            rel_fig.savefig(rel_path, dpi=200, bbox_inches="tight")
            plt.close(rel_fig)
            logger.info("Saved reliability diagram to %s", rel_path)

            unc_arr = (
                np.array(metrics["uncertainties"], dtype=np.float64)
                if "uncertainties" in metrics and len(metrics["uncertainties"]) == len(targets_np)
                else 1.0 - np.max(probs_np, axis=1)
            )
            preds_np = np.argmax(probs_np, axis=1)
            rc = risk_coverage_curve(unc_arr, preds_np == targets_np)
            rc_fig = plot_risk_coverage_curve(rc.coverage, rc.risk, aurc=area_under_risk_coverage(rc))
            rc_path = out_dir / f"risk_coverage_{args.split}.png"
            rc_fig.savefig(rc_path, dpi=200, bbox_inches="tight")
            plt.close(rc_fig)
            logger.info("Saved risk-coverage plot to %s", rc_path)

        if "uncertainties" in metrics and len(metrics["uncertainties"]) > 0:
            split_targets = [dataset[i].label for i in range(len(dataset))]
            has_labels = all(lbl in class_names for lbl in split_targets if lbl is not None) and any(lbl is not None for lbl in split_targets)
            if has_labels:
                target_indices = [class_names.index(lbl) for lbl in split_targets if lbl is not None]
                u_fig = plot_uncertainty_distribution(metrics["uncertainties"], target_indices, class_names)
            else:
                # OOD histogram
                u_fig, ax = plt.subplots(figsize=(7, 4.5))
                ax.hist(metrics["uncertainties"], bins=20, color="#d9534f", alpha=0.75, edgecolor="black")
                ax.set_title(f"Evidential Vacuity/Uncertainty Distribution ({args.split.upper()} Events)", fontsize=13, fontweight="bold")
                ax.set_xlabel(r"Uncertainty Mass $u \in [0, 1]$", fontsize=11)
                ax.set_ylabel("Event Count", fontsize=11)
                ax.grid(True, linestyle="--", alpha=0.5)
            u_path = out_dir / f"uncertainty_distribution_{args.split}.png"
            u_fig.savefig(u_path, dpi=200, bbox_inches="tight")
            plt.close(u_fig)
            logger.info("Saved uncertainty distribution plot to %s", u_path)
    except Exception as exc:
        logger.warning("Could not generate evaluation plots: %s", exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
