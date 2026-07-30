#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "usage: $0 COMMAND [ARG ...]" >&2
    exit 2
fi

# Codex runs inside bubblewrap with a private /dev, which hides WSL's /dev/dxg.
# Re-enter the same distro through wsl.exe while preserving the caller's cwd
# and PATH.  Outside the sandbox, execute the command directly.
if [[ -e /dev/dxg ]]; then
    exec "$@"
fi

wsl_exe=/mnt/c/Windows/System32/wsl.exe
if [[ ! -x "$wsl_exe" ]]; then
    echo "WSL GPU bridge is unavailable: $wsl_exe" >&2
    exit 1
fi

caller_cwd=$(pwd -P)
distro=${WSL_DISTRO_NAME:-Ubuntu}
exec /init "$wsl_exe" "$wsl_exe" \
    -d "$distro" \
    --cd "$caller_cwd" \
    -- /usr/bin/env "PATH=$PATH" "$@"
