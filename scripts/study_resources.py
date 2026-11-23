"""Inventory references locally; does not import or activate commercial presets.

Run with the project environment, optionally using Python -I as documented:
  uv run python -I scripts/study_resources.py resources --out .task-cache/resource-study
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from superphoto.resource_materials import inventory_resources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = inventory_resources(args.source)
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "preset-inventory.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {key: value for key, value in report["summary"].items() if key != "groups"}
    print(json.dumps({"summary": summary, "errors": report["errors"], "report": str(target)}, ensure_ascii=False, indent=2))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
