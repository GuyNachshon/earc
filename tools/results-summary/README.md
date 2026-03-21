Results summary tool

This helper parses a run's JSONL log (when `LOG_JSON=1`) or falls back to the text log to produce a concise summary suitable for STATUS.md.

Usage
- Write JSON logs by enabling: `LOG_JSON=1` on the training command.
- Summarize:
  - python3 tools/results-summary/summary.py --run-id <RUN_ID>
  - Or provide explicit paths:
    - python3 tools/results-summary/summary.py --json logs/<RUN_ID>.jsonl --txt logs/<RUN_ID>.txt

Output
- Prints final roundtrip BPB, compressed size, selected validation throughput, and wallclock.
- With `--markdown`, prints a one-line Markdown row for easy paste into docs/STATUS.md.

Notes
- No external dependencies.
- If both JSON and text logs are missing, the tool exits with a non-zero status.

