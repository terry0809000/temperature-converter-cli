import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Build benchmark report table")
    parser.add_argument("--metrics_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    rows = []
    for path in metrics_dir.glob("*_metrics.jsonl"):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                rows.append(
                    {
                        "task": payload["task"],
                        "family": payload["family"],
                        "macro_f1": payload["test_metrics"].get("macro_f1", payload["test_metrics"].get("eval_macro_f1")),
                        "micro_f1": payload["test_metrics"].get("micro_f1", payload["test_metrics"].get("eval_micro_f1")),
                        "auroc": payload["test_metrics"].get("auroc", payload["test_metrics"].get("eval_auroc")),
                        "train_seconds": payload.get("train_seconds"),
                    }
                )
    table = pd.DataFrame(rows)
    table.to_csv(args.output_csv, index=False)
    table.to_markdown(args.output_md, index=False)


if __name__ == "__main__":
    main()
