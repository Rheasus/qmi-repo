#!/usr/bin/env python3
"""Queue runner: executes experiment specs from a JSONL queue, resumable.

Usage:
    python runner.py --queue ../infra/queue/cv_rerun.jsonl \
                     --results-dir ~/qmi-v2/results --data-dir ~/qmi-v2/datasets

Behaviour:
  * a run is skipped if results/<run_id>/result.json already exists;
  * a crashed run leaves ckpt.pt behind and is resumed on the next pass;
  * failures are recorded in results/<run_id>/error.txt and the queue moves on;
  * a heartbeat line is appended to results/runner.log for remote monitoring.
"""

import argparse
import json
import time
import traceback
from pathlib import Path


def dispatch(spec, results_dir, data_dir):
    domain = spec["domain"]
    if domain == "cv":
        from cv.train import run
    elif domain == "nlp":
        from nlp.train import run
    elif domain == "tabular":
        from tabular.train import run
    else:
        raise ValueError(f"Unknown domain: {domain}")
    return run(spec, results_dir, data_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()

    results_dir = Path(args.results_dir).expanduser()
    results_dir.mkdir(parents=True, exist_ok=True)
    log = results_dir / "runner.log"

    def note(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(log, "a") as f:
            f.write(line + "\n")

    specs = [json.loads(l) for l in Path(args.queue).read_text().splitlines() if l.strip()]
    note(f"queue={args.queue} items={len(specs)}")

    done = skipped = failed = 0
    for i, spec in enumerate(specs):
        run_dir = results_dir / spec["run_id"]
        if (run_dir / "result.json").exists():
            skipped += 1
            continue
        note(f"[{i+1}/{len(specs)}] start {spec['run_id']}")
        t0 = time.time()
        try:
            res = dispatch(spec, str(results_dir), str(Path(args.data_dir).expanduser()))
            done += 1
            (run_dir / "error.txt").unlink(missing_ok=True)  # clear stale failure
            note(f"[{i+1}/{len(specs)}] done {spec['run_id']} "
                 f"acc={res.get('test_accuracy')} in {time.time()-t0:.0f}s")
        except Exception as e:
            failed += 1
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "error.txt").write_text(
                f"{e}\n\n{traceback.format_exc()}")
            note(f"[{i+1}/{len(specs)}] FAILED {spec['run_id']}: {e}")

    note(f"queue finished: done={done} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
