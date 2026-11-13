"""Audit the vision planner against an existing job without editing its photos."""
import argparse
import json
from pathlib import Path

from superphoto.config import Settings
from superphoto.directions import design_directions
from superphoto.models import Critique, TechnicalReport


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    critique = Critique.model_validate_json((args.job / "critique.json").read_text())
    technical = TechnicalReport.model_validate_json((args.job / "technical.json").read_text())
    directions = design_directions(Settings(), args.job / "critique_preview.jpg", critique, technical,
                                   audit_path=args.out)
    audit = json.loads(args.out.read_text())
    print(json.dumps({"issues": audit["issues"], "selected": [{"slot": d.slot, "title": d.title, "source": d.source}
                                                              for d in directions]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
