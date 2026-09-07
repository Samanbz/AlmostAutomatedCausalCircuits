"""Optional Weights & Biases logging, controlled by the ``wandb`` config section.

Example config block::

    "wandb": {
      "enabled": true,
      "project": "monarch-causal-circuits",
      "entity": null,
      "mode": "online",
      "tags": [],
      "log_plots": true
    }

``mode`` accepts ``"online"``, ``"offline"`` or ``"disabled"`` (passed through
to ``wandb.init``); use ``"offline"`` on machines without network access and
sync later with ``wandb sync``.
"""

import glob
import os
from typing import Any, Dict, Optional

from src.logger import logger as g_logger


logger = g_logger.getChild("wandb")


def init_wandb(cfg: Dict[str, Any]) -> Optional[Any]:
    """Start a wandb run if enabled in the config, else return ``None``."""
    wb_cfg = cfg.get("wandb", {})
    if not wb_cfg.get("enabled", False):
        return None
    try:
        import wandb
    except ImportError as e:
        raise ImportError(
            "wandb logging is enabled in the config but wandb is not installed. "
            "Install it with `pip install -e .[wandb]`."
        ) from e

    run = wandb.init(
        project=wb_cfg.get("project", "monarch-causal-circuits"),
        entity=wb_cfg.get("entity"),
        mode=wb_cfg.get("mode", "online"),
        name=cfg["experiment"]["id"],
        tags=wb_cfg.get("tags") or None,
        config=cfg,
    )
    return run


def log_epoch(
    wandb_run: Optional[Any],
    epoch: int,
    obs_nll: float,
    do_nll: float,
    step_size: float,
    epoch_seconds: float,
) -> None:
    """Log per-epoch training metrics."""
    if wandb_run is None:
        return
    wandb_run.log(
        {
            "train/obs_nll_epoch": obs_nll,
            "train/do_nll_epoch": do_nll,
            "train/step_size": step_size,
            "train/epoch_seconds": epoch_seconds,
        },
        step=epoch,
    )


def log_summary(wandb_run: Optional[Any], metrics: Dict[str, float]) -> None:
    """Record the final before/after NLLs on the run summary."""
    if wandb_run is None:
        return
    wandb_run.summary.update(metrics)


def log_plots(
    wandb_run: Optional[Any],
    output_dir: str,
    exp_id: str,
) -> None:
    """Upload every plot PNG in the results directory to the run."""
    if wandb_run is None:
        return
    import wandb

    for path in sorted(glob.glob(os.path.join(output_dir, "*.png"))):
        name = os.path.splitext(os.path.basename(path))[0]
        wandb_run.log({f"plots/{name}": wandb.Image(path)})
    logger.info("Logged plots to wandb run %s", wandb_run.name or exp_id)


def finish_wandb(wandb_run: Optional[Any]) -> None:
    """Close the run (no-op when wandb is disabled)."""
    if wandb_run is not None:
        wandb_run.finish()
