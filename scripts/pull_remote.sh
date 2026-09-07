#!/bin/bash
# Pull from the DGX. Default is source files only; results are pulled on demand.
#
#   ./scripts/pull_remote.sh           # source only (src/, tests/, experiments/ code, configs, ...)
#   ./scripts/pull_remote.sh results   # models/ + experiments/results/ (+ scratch/do_query_scaling/runs/)
#
# Datasets flow local -> DGX only (pushed by push_remote.sh); there is no data
# pull mode. Nothing local is ever deleted (no --delete). Safe to `source`:
# the script re-execs itself as a bash child, so nothing leaks into or can
# kill the interactive shell.

# If sourced (bash: BASH_SOURCE[0] != $0; zsh: ZSH_VERSION set) or otherwise
# not running as an executed bash script, re-exec under bash. This must run
# before `set -euo pipefail` so those options never touch the parent shell,
# and under bash so the rsync exclude braces below expand correctly.
if [ -n "${ZSH_VERSION:-}" ]; then
    _self="${(%):-%x}"
    bash "$_self" "$@"
    return $?
fi
if [ -n "${BASH_SOURCE:-}" ] && [ "${BASH_SOURCE[0]}" != "$0" ]; then
    bash "${BASH_SOURCE[0]}" "$@"
    return $?
fi

set -euo pipefail

# macOS ships openrsync, an old rsync 2.6.9 clone with several quirks; the
# real rsync (brew install rsync) is required.
if rsync --version 2>/dev/null | grep -qi openrsync; then
    echo "error: /usr/bin/rsync is openrsync; install the real rsync:  brew install rsync" >&2
    exit 1
fi

# Quoted: bash would otherwise tilde-expand the '~' after the ':' against the
# LOCAL home (tilde expansion applies in assignments after ':' or '=').
REMOTE='dgx:~/dev/MonarchCausalCircuits'
MODE="${1:-source}"

case "$MODE" in
  source)
    rsync -avzP \
      --exclude={'build/','.git/','__pycache__/','.DS_Store','papers/','causal-pc/','wandb/','logs/','data/','models/','experiments/results/','scratch/','scripts/'} \
      "$REMOTE/" ./
    ;;
  results)
    rsync -avzP "$REMOTE/models/" ./models/
    rsync -avzP "$REMOTE/experiments/results/" ./experiments/results/
    rsync -avzP "$REMOTE/scratch/do_query_scaling/runs/" ./scratch/do_query_scaling/runs/ \
      || echo "note: scratch/do_query_scaling/runs/ not present on DGX, skipped" >&2
    ;;
  *)
    echo "Unknown mode: $MODE (use: source | results)" >&2
    exit 1
    ;;
esac
