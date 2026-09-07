#!/bin/bash
# If sourced (bash: BASH_SOURCE[0] != $0; zsh: ZSH_VERSION set), re-exec under
# bash so nothing leaks into the interactive shell and the exclude/include
# braces below expand correctly.
if [ -n "${ZSH_VERSION:-}" ]; then
    _self="${(%):-%x}"
    bash "$_self" "$@"
    return $?
fi
if [ -n "${BASH_SOURCE:-}" ] && [ "${BASH_SOURCE[0]}" != "$0" ]; then
    bash "${BASH_SOURCE[0]}" "$@"
    return $?
fi

# macOS ships openrsync, an old rsync 2.6.9 clone with several quirks; the
# real rsync (brew install rsync) is required.
if rsync --version 2>/dev/null | grep -qi openrsync; then
    echo "error: /usr/bin/rsync is openrsync; install the real rsync:  brew install rsync" >&2
    exit 1
fi

rsync -avzP \
  --exclude={'build/','.git/','__pycache__/','.pytest_cache/','.ruff_cache/','.DS_Store','.claude/','.gemini/','.vscode/'} \
  --exclude={'papers/','logs/','wandb/','models/','experiments/results/','causal-pc/','scripts/'} \
  --exclude={'playground.ipynb','*.png','*.html'} \
  ./ dgx:~/dev/MonarchCausalCircuits
