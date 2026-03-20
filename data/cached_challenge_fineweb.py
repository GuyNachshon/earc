"""Download cached FineWeb shards/tokenizer to match challenge SP-1024 layout.

This mirrors the structure referenced by train_gpt.py defaults:
- Dataset: ./data/datasets/fineweb10B_sp1024/fineweb_{train|val}_NNNNNN.bin
- Tokenizer: ./data/tokenizers/fineweb_1024_bpe.model

Usage
  python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

Notes
- Uses willdepueoai/parameter-golf (dataset repo) by default.
- Gracefully stops when a shard is missing; validates magic/version header.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download


DATAFILE_MAGIC = 20240520
DATAFILE_VERSION = 1
DEFAULT_REPO_ID = os.environ.get("MATCHED_FINEWEB_REPO_ID", "willdepueoai/parameter-golf")


def hf_get_dataset_file(repo_id: str, subpath: str) -> Path:
    p = Path(subpath)
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=p.name,
            subfolder=p.parent.as_posix() if p.parent != Path(".") else None,
            repo_type="dataset",
        )
    ).resolve(strict=True)


def validate_datafile(path: Path) -> None:
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256:
        raise ValueError(f"Bad header size for {path}")
    if int(header[0]) != DATAFILE_MAGIC or int(header[1]) != DATAFILE_VERSION:
        raise ValueError(f"Unexpected magic/version for {path}")
    num_tokens = int(header[2])
    expected_size = 256 * np.dtype("<i4").itemsize + num_tokens * np.dtype("<u2").itemsize
    if path.stat().st_size != expected_size:
        raise ValueError(f"Size mismatch for {path}: expected {expected_size} bytes")


def download_sp1024(repo_id: str, out_root: Path, train_shards: int | None) -> None:
    ds_dir = out_root / "datasets" / "fineweb10B_sp1024"
    tok_dir = out_root / "tokenizers"
    ds_dir.mkdir(parents=True, exist_ok=True)
    tok_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    # In the dataset repo, assets are nested under a top-level "datasets/" folder.
    remote_tok = "datasets/tokenizers/fineweb_1024_bpe.model"
    tok_src = hf_get_dataset_file(repo_id, remote_tok)
    tok_dst = tok_dir / Path(remote_tok).name
    if not tok_dst.exists():
        tok_dst.write_bytes(tok_src.read_bytes())

    # Validation shards: fetch sequentially until missing.
    val_idx = 0
    while True:
        name = f"fineweb_val_{val_idx:06d}.bin"
        try:
            src = hf_get_dataset_file(repo_id, f"datasets/datasets/fineweb10B_sp1024/{name}")
        except Exception:
            break
        dst = ds_dir / name
        if not dst.exists():
            dst.write_bytes(src.read_bytes())
            validate_datafile(dst)
        val_idx += 1
        if val_idx > 9999:
            break

    # Training shards: bounded by --train-shards if provided; otherwise keep going until missing.
    train_idx = 0
    while True:
        if train_shards is not None and train_idx >= train_shards:
            break
        name = f"fineweb_train_{train_idx:06d}.bin"
        try:
            src = hf_get_dataset_file(repo_id, f"datasets/datasets/fineweb10B_sp1024/{name}")
        except Exception:
            break
        dst = ds_dir / name
        if not dst.exists():
            dst.write_bytes(src.read_bytes())
            validate_datafile(dst)
        train_idx += 1
        if train_idx > 999999:
            break

    print(
        f"Downloaded SP-1024: tokenizer={tok_dst} val_files={val_idx} train_files={train_idx} into {ds_dir}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download cached FineWeb shards/tokenizer (SP-1024)")
    p.add_argument("--variant", default="sp1024", choices=["sp1024"], help="Dataset variant to download")
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HF dataset repo id")
    p.add_argument("--train-shards", type=int, default=10, help="Number of training shards to fetch (None=all)")
    p.add_argument("--out-root", default="data", help="Root output directory (contains datasets/ and tokenizers/)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_root = Path(args.out_root)
    if args.variant == "sp1024":
        download_sp1024(args.repo_id, out_root, int(args.train_shards) if args.train_shards is not None else None)
    else:
        raise NotImplementedError(args.variant)


if __name__ == "__main__":
    main()
