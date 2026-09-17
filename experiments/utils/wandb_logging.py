"""Optional Weights & Biases logging, controlled by the ``wandb`` config section.

Example config block::

    "wandb": {
      "enabled": true,
      "project": "monarch-causal-circuits",
      "entity": null,
      "mode": "online",
      "tags": []
    }

``mode`` accepts ``"online"``, ``"offline"`` or ``"disabled"`` (passed through
to ``wandb.init``); use ``"offline"`` on machines without network access and
sync later with ``wandb sync``.  Each seed run becomes a separate wandb run
named ``<exp_id>-seed<S>`` grouped under ``<exp_id>``.
"""

from typing import Any, Dict, Optional

from src.logger import logger as g_logger


logger = g_logger.getChild("wandb")


def init_wandb(cfg: Dict[str, Any], seed: int) -> Optional[Any]:
    """Start a wandb run if enabled in the config, else return ``None``."""
    wb_cfg = cfg.get("wandb", {})
    if not wb_cfg.get("enabled", False):
        return None
    try:
        import wandb
    except ImportError as e:
        raise ImportError(
            "wandb logging is enabled in the config but wandb is not installed. "
            "Install it with `pip install wandb`."
        ) from e

    exp_id = cfg["experiment"]["id"]
    run = wandb.init(
        project=wb_cfg.get("project", "monarch-causal-circuits"),
        entity=wb_cfg.get("entity"),
        mode=wb_cfg.get("mode", "online"),
        name=f"{exp_id}-seed{seed}",
        group=exp_id,
        tags=wb_cfg.get("tags") or None,
        config={**cfg, "seed": seed},
    )
    return run


def log_eval(
    wandb_run: Optional[Any],
    epoch: int,
    obs_test_nll: float,
    do_test_nll: Optional[float],
    step_size: float,
) -> None:
    """Log the periodic checkpoint-epoch evaluations."""
    if wandb_run is None:
        return
    payload = {
        "eval/obs_test_nll": obs_test_nll,
        "eval/step_size": step_size,
        "epoch": epoch,
    }
    if do_test_nll is not None:
        payload["eval/do_test_nll"] = do_test_nll
    wandb_run.log(payload, step=epoch)


def log_summary(wandb_run: Optional[Any], metrics: Dict[str, Any]) -> None:
    """Record the final metrics on the run summary."""
    if wandb_run is None:
        return
    wandb_run.summary.update(metrics)


def finish_wandb(wandb_run: Optional[Any]) -> None:
    """Close the run (no-op when wandb is disabled)."""
    if wandb_run is not None:
        wandb_run.finish()
