"""Recheck saved actual exports without editing photos or modifying job records."""
import argparse
import json
from pathlib import Path
from uuid import UUID

from superphoto.config import get_settings
from superphoto.models import Critique
from superphoto.review import review_render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id", type=UUID)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--slot", choices=list("ABC"), default="B")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.data / "jobs" / str(args.job_id)
    critique = Critique.model_validate_json((root / "critique.json").read_text())
    audit = json.loads((root / f"lightroom-audits/variant_{args.slot}.json").read_text())
    before = root / "lightroom-exports/baseline/baseline.jpg"
    after = Path(audit.get("accepted_export") or audit["rounds"][-1]["export"])
    result = review_render(get_settings(), before, after, critique.intent, "final")
    payload = {"before": str(before), "after": str(after), "review": result.model_dump()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(result.model_dump(), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
