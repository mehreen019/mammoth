import importlib
import json
import sys
from pathlib import Path
from typing import Any


def _import_hf_datasets() -> Any:
    """
    Import the HuggingFace `datasets` package, skipping the local `datasets` folder.
    Mirrors the loader logic used inside the project to avoid circular imports.
    """
    repo_root = Path(__file__).resolve().parent

    original_path = list(sys.path)
    original_modules = {}

    try:
        filtered_path = []
        for entry in sys.path:
            if entry in ("", "."):
                continue
            try:
                if Path(entry).resolve() == repo_root:
                    continue
            except Exception:
                pass
            filtered_path.append(entry)

        sys.path = filtered_path

        for name in list(sys.modules.keys()):
            if name == "datasets" or name.startswith("datasets."):
                original_modules[name] = sys.modules[name]
                del sys.modules[name]

        return importlib.import_module("datasets")
    finally:
        sys.path = original_path
        sys.modules.update(original_modules)


hf_datasets = _import_hf_datasets()
load_dataset = hf_datasets.load_dataset


def dump(lang_code: str) -> None:
    dataset = load_dataset("wikiann", lang_code)
    out_dir = Path("data") / "wikiann" / lang_code
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "validation", "test"):
        samples = dataset[split]
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as handle:
            for record in samples:
                handle.write(
                    json.dumps(
                        {
                            "tokens": record["tokens"],
                            "ner_tags": record["ner_tags"],
                        }
                    )
                    + "\n"
                )


if __name__ == "__main__":
    for code in ("hi", "bn"):
        dump(code)
