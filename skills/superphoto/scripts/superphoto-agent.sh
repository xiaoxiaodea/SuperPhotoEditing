#!/bin/bash
# Talk to the SuperPhoto already running on this Mac. Prints one JSON object
# on stdout. Progress lines go to stderr.
set -euo pipefail

script=$0
phys=$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$script")
root=$(cd "$(dirname "$phys")/../../.." && pwd)

fail() {
  printf '%s\n' "{\"ok\":false,\"error\":$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1")}"
  exit 1
}

if [ ! -f "$root/pyproject.toml" ]; then
  fail "找不到 SuperPhoto 工程，技能路径不对"
fi

uv=""
if command -v uv >/dev/null 2>&1; then
  uv=$(command -v uv)
else
  for candidate in "$HOME/.local/bin/uv" /opt/homebrew/bin/uv /usr/local/bin/uv; do
    if [ -x "$candidate" ]; then
      uv=$candidate
      break
    fi
  done
fi
if [ -z "$uv" ]; then
  fail "这台 Mac 上找不到 uv，SuperPhoto 的助手命令没法运行"
fi

export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:${PATH:-/usr/bin:/bin}"
exec "$uv" run --project "$root" superphoto agent "$@"
