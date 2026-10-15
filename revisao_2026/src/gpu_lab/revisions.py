"""Revisões imutáveis para modelos públicos e fixtures locais de integração."""
from pathlib import Path
import hashlib
import re


def resolve_revision(model_id, revision="main"):
    path = Path(model_id)
    if path.is_dir():
        h = hashlib.sha256()
        for filename in sorted(path.iterdir()):
            if filename.is_file():
                h.update(filename.name.encode())
                h.update(filename.read_bytes())
        return "local-" + h.hexdigest()
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    from huggingface_hub import model_info
    return model_info(model_id, revision=revision).sha
