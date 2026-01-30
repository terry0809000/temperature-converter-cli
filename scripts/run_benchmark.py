import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark across configurations")
    parser.add_argument("--configs", nargs="+", required=True)
    args = parser.parse_args()

    for config in args.configs:
        config_path = Path(config)
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config: {config}")
        subprocess.run(["python", "scripts/train.py", "--config", str(config_path)], check=True)


if __name__ == "__main__":
    main()
