"""Train Baseline 1 (CNN + Softmax) on the preprocessed seismic dataset (Phase 3).

Usage:
    python scripts/train.py [--config configs/config.yaml] [--data-root PATH] [--epochs N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from seismic_ml.config import load_config
from seismic_ml.data.loader import create_dataloader
from seismic_ml.data.processed import ProcessedEventDataset
from seismic_ml.data.splits import load_split_table
from seismic_ml.features.cnn_encoder import CNNEncoder
from seismic_ml.models.cnn import CNNBaseline
from seismic_ml.paths import project_root_from_config, resolve_paths
from seismic_ml.training.trainer import Trainer
from seismic_ml.utils.logging import configure_logging
from seismic_ml.utils.reproducibility import set_seed

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Entry config file.")
    parser.add_argument("--data-root", default=None, help="Data root directory.")
    parser.add_argument("--output-root", default=None, help="Output root directory.")
    parser.add_argument("--device", default=None, help="Device to use ('cpu', 'cuda', 'auto').")
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--clean", action="store_true", help="Remove existing output directory before starting fresh.")
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

    seed = int(config.get("seed", 42))
    set_seed(seed)

    # Determine device
    device_arg = args.device or config.get("training", {}).get("device", "auto")
    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg
    logger.info("Using device: %s", device)

    # Load split table
    split_path = paths.processed / "splits" / "s1_temporal" / "events.csv"
    if not split_path.exists():
        logger.error(
            "Split table not found at %s. Please run `python scripts/prepare_data.py` first.",
            split_path,
        )
        return 1
    split_table = load_split_table(split_path)

    # Setup datasets
    processed_comcat_dir = paths.processed / "pnw" / "comcat"
    if not processed_comcat_dir.exists():
        logger.error(
            "Processed collection not found at %s. Run `python scripts/prepare_data.py` first.",
            processed_comcat_dir,
        )
        return 1

    train_dataset = ProcessedEventDataset(processed_comcat_dir, split_table, splits=["train"])
    val_dataset = ProcessedEventDataset(processed_comcat_dir, split_table, splits=["val"])
    logger.info("Loaded datasets: %d train events, %d val events", len(train_dataset), len(val_dataset))

    if len(train_dataset) == 0:
        logger.error("No training events found in the processed store.")
        return 1

    # Configuration overrides
    train_cfg = config.get("training", {})
    batch_size = args.batch_size or train_cfg.get("batch_size") or 16
    if args.epochs is not None:
        config.setdefault("training", {})["max_epochs"] = args.epochs
    if args.lr is not None:
        config.setdefault("training", {}).setdefault("optimizer", {})["lr"] = args.lr

    class_names = config.get("data", {}).get("labels", {}).get("class_names") or [
        "earthquake",
        "anthropogenic",
    ]

    # Data loaders
    num_workers = int(train_cfg.get("num_workers") or 0)
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        class_names=class_names,
        num_workers=num_workers,
        seed=seed,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        class_names=class_names,
        num_workers=num_workers,
        seed=seed,
    )

    # Build model (Baseline 1: 1D CNN + Softmax)
    model_cfg = config.get("model", {})
    encoder_cfg = model_cfg.get("encoder", {})
    embedding_dim = int(encoder_cfg.get("embedding_dim") or 128)
    in_channels = int(encoder_cfg.get("in_channels") or 3)
    station_pooling = model_cfg.get("station_pooling") or "mean"

    encoder = CNNEncoder(
        embedding_dim=embedding_dim,
        in_channels=in_channels,
        pool_time=True,
    )
    model = CNNBaseline(
        encoder=encoder,
        num_classes=len(class_names),
        station_pooling=station_pooling,
    )

    run_output_dir = paths.output_root / "baseline1_cnn"
    if args.clean and run_output_dir.exists():
        import shutil
        shutil.rmtree(run_output_dir)
        logger.info("Cleaned previous output directory: %s", run_output_dir)
    run_output_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        output_dir=run_output_dir,
        class_names=class_names,
    )

    logger.info("Starting training Baseline 1 (CNN + %s pooling)...", station_pooling)
    results = trainer.fit(train_loader, val_loader)

    # Save metrics JSON
    metrics_file = run_output_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save training curves and confusion matrix plots for project reports
    try:
        from seismic_ml.utils.visualization import plot_confusion_matrix, plot_training_history
        import matplotlib.pyplot as plt

        history_fig = plot_training_history(results["history"])
        history_path = run_output_dir / "training_curves.png"
        history_fig.savefig(history_path, dpi=200, bbox_inches="tight")
        plt.close(history_fig)
        logger.info("Saved training curves graph to %s", history_path)

        best_ckpt = run_output_dir / "best_model.pt"
        if best_ckpt.exists():
            trainer.load_checkpoint(best_ckpt)
        val_eval = trainer.evaluate(val_loader)
        cm = np.array(val_eval.get("confusion_matrix", []))
        if cm.size > 0:
            cm_fig = plot_confusion_matrix(cm, class_names)
            cm_path = run_output_dir / "val_confusion_matrix.png"
            cm_fig.savefig(cm_path, dpi=200, bbox_inches="tight")
            plt.close(cm_fig)
            logger.info("Saved validation confusion matrix graph to %s", cm_path)
    except Exception as exc:
        logger.warning("Could not generate report plots: %s", exc)

    logger.info(
        "Training complete. Best validation score: %.4f. Artifacts saved to %s",
        results.get("best_val_score", 0.0),
        run_output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
