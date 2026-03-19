"""Export FineWeb shards in byte-level format compatible with baseline train_gpt.py.

Features
- Downloads docs_selected.jsonl from the same HF dataset repo used by the cached downloader
- Encodes each document as raw UTF-8 bytes (no BOS/EOS, no special tokens)
- Writes .bin shards with the same uint16 format and header (magic/version) expected by train_gpt.py
- First NUM_VAL_DOCS documents form the fixed validation split; remainder are training

Usage
  python3 data/export_fineweb_bytes.py \
    --output-dir ./data/datasets/fineweb_bytes \
    --train-shards 10

Environment defaults match the baseline helpers:
- MATCHED_FINEWEB_REPO_ID (default: willdepueoai/parameter-golf)
- MATCHED_FINEWEB_REMOTE_ROOT_PREFIX (default: datasets)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterator

import numpy as np
from huggingface_hub import hf_hub_download


DATAFILE_MAGIC = 20240520
DATAFILE_VERSION = 1
DEFAULT_REPO_ID = os.environ.get("MATCHED_FINEWEB_REPO_ID", "willdepueoai/parameter-golf")
DEFAULT_REMOTE_ROOT = os.environ.get("MATCHED_FINEWEB_REMOTE_ROOT_PREFIX", "datasets")
DEFAULT_OUTPUT_DIR = Path("data/datasets/fineweb_bytes")
DEFAULT_SHARD_SIZE = 100_000_000  # tokens per shard
DEFAULT_NUM_VAL_DOCS = 50_000


def hf_get(repo_id: str, remote_root: str, relative_path: str) -> Path:
    remote_path = Path(remote_root) / relative_path if remote_root else Path(relative_path)
    cached = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=remote_path.name,
            subfolder=remote_path.parent.as_posix() if remote_path.parent != Path(".") else None,
            repo_type="dataset",
        )
    )
    return cached.resolve(strict=True)


def iter_docs(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                text = json.loads(line)["text"]
            except Exception:
                continue
            # Replace embedded NULs to avoid downstream surprises; keep text otherwise unmodified.
            yield text.replace("\x00", " ")


def write_datafile(path: Path, toks: np.ndarray) -> None:
    header = np.zeros(256, dtype="<i4")
    header[0] = DATAFILE_MAGIC
    header[1] = DATAFILE_VERSION
    header[2] = int(toks.size)
    toks16 = np.asarray(toks, dtype=np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header.tobytes())
        f.write(toks16.tobytes())


def export_shards(
    *,
    docs_jsonl: Path,
    output_dir: Path,
    shard_size: int,
    num_val_docs: int,
    max_train_shards: int | None,
) -> dict:
    stats = {
        "docs_total": 0,
        "docs_val": 0,
        "docs_train": 0,
        "files_val": 0,
        "files_train": 0,
        "tokens_val": 0,
        "tokens_train": 0,
    }
    buf = np.empty((shard_size,), dtype=np.uint16)
    fill = 0
    split = "val"
    shard_idx = {"val": 0, "train": 0}

    def flush() -> None:
        nonlocal fill
        if fill == 0:
            return
        out = output_dir / f"fineweb_{split}_{shard_idx[split]:06d}.bin"
        write_datafile(out, buf[:fill])
        stats[f"files_{split}"] += 1
        shard_idx[split] += 1
        fill = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    # Clean existing shards matching our patterns (avoid mixing variants).
    for pattern in ("fineweb_train_*.bin", "fineweb_val_*.bin"):
        for stale in output_dir.glob(pattern):
            stale.unlink()

    for i, text in enumerate(iter_docs(docs_jsonl)):
        target_split = "val" if i < num_val_docs else "train"
        if target_split != split:
            flush()
            split = target_split
        if split == "train" and max_train_shards is not None and shard_idx["train"] >= max_train_shards:
            break

        # Encode raw UTF-8 bytes. No BOS/EOS; strictly 1 token per byte.
        data = text.encode("utf-8", errors="replace")
        arr = np.frombuffer(data, dtype=np.uint8).astype(np.uint16, copy=False)

        stats[f"docs_{split}"] += 1
        stats["docs_total"] += 1
        stats[f"tokens_{split}"] += int(arr.size)

        pos = 0
        while pos < arr.size:
            take = min(shard_size - fill, arr.size - pos)
            buf[fill : fill + take] = arr[pos : pos + take]
            fill += take
            pos += take
            if fill == shard_size:
                flush()

        if (i + 1) % 100_000 == 0:
            print(
                f"{output_dir.name}: docs={i+1} files_val={stats['files_val']} files_train={stats['files_train']}",
                flush=True,
            )

    flush()
    return stats


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export FineWeb shards as raw UTF-8 byte tokens")
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HF dataset repo id (e.g., willdepueoai/parameter-golf)")
    p.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT, help="Remote subdir containing docs_selected.jsonl")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for shards")
    p.add_argument("--train-shards", type=int, default=80, help="Number of training shards to export")
    p.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE, help="Tokens per shard")
    p.add_argument(
        "--num-val-docs",
        type=int,
        default=None,
        help="Validation document count (defaults to sidecar or 50,000)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)

    # Download docs and optional sidecar to derive val doc count.
    docs_path = hf_get(args.repo_id, args.remote_root, "docs_selected.jsonl")
    sidecar_path = hf_get(args.repo_id, args.remote_root, "docs_selected.source_manifest.json")
    num_val_docs = args.num_val_docs or DEFAULT_NUM_VAL_DOCS
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        num_val_docs = int(sidecar.get("num_val_docs") or num_val_docs)
    except Exception:
        pass

    stats = export_shards(
        docs_jsonl=docs_path,
        output_dir=output_dir,
        shard_size=int(args.shard_size),
        num_val_docs=num_val_docs,
        max_train_shards=int(args.train_shards) if args.train_shards is not None else None,
    )

    total_files = stats["files_val"] + stats["files_train"]
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "files_total": total_files,
                **stats,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

