#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize EARC run logs")
    p.add_argument("--run-id", help="Run ID (derives logs/<RUN_ID>.jsonl/.txt)")
    p.add_argument("--json", dest="json_path", help="Path to JSONL log", default=None)
    p.add_argument("--txt", dest="txt_path", help="Path to text log", default=None)
    p.add_argument("--markdown", action="store_true", help="Emit a single Markdown table row")
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def parse_txt(path: Path) -> dict:
    res = {}
    if not path.exists():
        return res
    txt = path.read_text(encoding="utf-8")
    m = re.findall(r"final_int8_zlib_roundtrip val_loss:([0-9.]+) val_bpb:([0-9.]+) eval_time:([0-9]+)ms", txt)
    if m:
        res["roundtrip_loss"], res["roundtrip_bpb"], res["eval_ms"] = map(float, m[-1])
    m = re.findall(r"Serialized model int8\+zlib: ([0-9]+) bytes", txt)
    if m:
        res["int8_zlib_bytes"] = int(m[-1])
    m = re.findall(r"tok/s:([0-9]+)B/s:([0-9]+)", txt)
    if m:
        res["toks_per_sec"], res["bytes_per_sec"] = map(int, m[-1])
    return res


def main() -> None:
    args = parse_args()
    json_path = Path(args.json_path) if args.json_path else Path("logs") / f"{args.run_id}.jsonl"
    txt_path = Path(args.txt_path) if args.txt_path else Path("logs") / f"{args.run_id}.txt"
    recs = load_jsonl(json_path)
    summary: dict[str, float | int] = {}
    # Prefer JSONL roundtrip and artifact sizes
    for r in recs:
        if r.get("kind") == "artifact_int8_zlib":
            summary["int8_zlib_bytes"] = int(r.get("model_bytes_int8_zlib", 0))
        if r.get("kind") == "final_roundtrip":
            summary["roundtrip_loss"] = float(r.get("val_loss", 0.0))
            summary["roundtrip_bpb"] = float(r.get("val_bpb", 0.0))
            summary["eval_ms"] = float(r.get("eval_time_ms", 0.0))
        if r.get("kind") == "val":
            summary["toks_per_sec"] = float(r.get("toks_per_sec", 0.0))
            summary["bytes_per_sec"] = float(r.get("bytes_per_sec", 0.0))

    if not summary:
        summary = parse_txt(txt_path)

    if not summary:
        print("No logs found to summarize", flush=True)
        raise SystemExit(1)

    if args.markdown:
        rt_bpb = summary.get("roundtrip_bpb", "-")
        size = summary.get("int8_zlib_bytes", "-")
        tps = summary.get("toks_per_sec", "-")
        print(f"| {args.run_id} | {rt_bpb} | {size} B | {tps} tok/s |")
    else:
        for k in ("roundtrip_bpb", "roundtrip_loss", "int8_zlib_bytes", "toks_per_sec", "bytes_per_sec", "eval_ms"):
            if k in summary:
                print(f"{k}: {summary[k]}")


if __name__ == "__main__":
    main()

