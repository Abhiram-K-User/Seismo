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
from seismic_ml.data.processed import ProcessedEventDataset
from seismic_ml.data.splits import load_split_table
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

    processed_comcat_dir = paths.processed / "pnw" / "comcat"
    if not processed_comcat_dir.exists():
        logger.error("Processed data directory not found at %s.", processed_comcat_dir)
        return 1

    dataset = ProcessedEventDataset(processed_comcat_dir, split_table, splits=[args.split])
    logger.info("Loaded %d events from split %r", len(dataset), args.split)

    if len(dataset) == 0:
        logger.error("Dataset has 0 events for split %r", args.split)
        return 1

    class_names = config.get("data", {}).get("labels", {}).get("class_names") or [
        "earthquake",
        "anthropogenic",
    ]

    loader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        class_names=class_names,
        allow_unlabeled=(args.split == "ood"),
    )

    # Determine model architecture
    model_cfg = config.get("model", {})
    encoder_cfg = model_cfg.get("encoder", {})
    embedding_dim = int(encoder_cfg.get("embedding_dim") or 128)
    in_channels = int(encoder_cfg.get("in_channels") or 3)
    station_pooling = model_cfg.get("station_pooling") or "attention"

    encoder = CNNEncoder(embedding_dim=embedding_dim, in_channels=in_channels, pool_time=True)

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

        gat = GAT(
            in_dim=embedding_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        model = GATClassifier(
            encoder=encoder,
            gat=gat,
            num_classes=len(class_names),
            station_pooling=station_pooling,
            graph_method=spatial_cfg.get("method") or "knn",
            k_neighbors=int(spatial_cfg.get("k_neighbors") or 5),
            radius_km=spatial_cfg.get("radius_km"),
            sigma_km=spatial_cfg.get("sigma_km"),
            head_type=head_type,
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

    print("\n" + "=" * 45)
    print(f"Evaluation Results on split: {args.split}")
    print("=" * 45)
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"  {k:25s}: {v:.4f}")
    print("=" * 45 + "\n")

    # Save confusion matrix plot for reports
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from seismic_ml.utils.visualization import plot_confusion_matrix, plot_uncertainty_distribution

        cm = np.array(metrics.get("confusion_matrix", []))
        out_dir = args.checkpoint.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        if cm.size > 0:
            cm_fig = plot_confusion_matrix(cm, class_names)
            cm_path = out_dir / f"confusion_matrix_{args.split}.png"
            cm_fig.savefig(cm_path, dpi=200, bbox_inches="tight")
            plt.close(cm_fig)
            logger.info("Saved confusion matrix plot to %s", cm_path)

        if "uncertainties" in metrics and len(metrics["uncertainties"]) > 0:
            split_targets = [dataset[i].label for i in range(len(dataset))]
            if all(lbl in class_names for lbl in split_targets if lbl is not None):
                target_indices = [class_names.index(lbl) for lbl in split_targets if lbl is not None]
                u_fig = plot_uncertainty_distribution(metrics["uncertainties"], target_indices, class_names)
                u_path = out_dir / f"uncertainty_distribution_{args.split}.png"
                u_fig.savefig(u_path, dpi=200, bbox_inches="tight")
                plt.close(u_fig)
                logger.info("Saved uncertainty distribution plot to %s", u_path)
    except Exception as exc:
        logger.warning("Could not generate evaluation plots: %s", exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
