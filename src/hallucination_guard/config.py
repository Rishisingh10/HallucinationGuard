from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config():
    config_path = ROOT / "configs" / "config.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)
