from dataclasses import dataclass
from pathlib import Path
import yaml
from platformdirs import user_config_dir


@dataclass
class OmeroConfig:
    port: int = 4064
    host: str = "omero-server.epfl.ch"
    group: str = "imaging-updepalma"
    default_user: str = "imaging-robot"


def get_config_path() -> Path:
    config_dir = Path(user_config_dir("lutuflow"))
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "config.yaml"


def load_config(path: str | Path | None = None) -> OmeroConfig:
    if path is None:
        path = get_config_path()

    config = OmeroConfig()

    if Path(path).exists():
        with open(path) as f:
            values = yaml.safe_load(f) or {}
        return OmeroConfig(**{**config.__dict__, **values})

    return config
