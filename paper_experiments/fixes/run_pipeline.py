#!/usr/bin/env python3
"""
RQ Rerun Pipeline Driver

Runs experiments across multiple models in a controlled, subprocess-isolated way.
Each (model, RQ) pair is a fresh Python subprocess — GPU memory fully released
between runs.

Usage:
    python run_pipeline.py --models_file models.txt --rqs RQ2a RQ3 RQ6 --nsamples 30

Or inline:
    python run_pipeline.py --models "qwen3_0.6b qwen3_1.7b" --rqs RQ3 --nsamples 5

Output:
    logs/pipeline_<timestamp>/
        <model>__<rq>.log          per-run log
        summary.json               final status summary
        pipeline.log               driver-level log
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# --------- Model size tiers (for parallelism decisions) ---------
MODEL_SIZES_GB = {
    # tier: SMALL (<2B params, can run 2-3 parallel on 80GB)
    "qwen3_0.6b": 2, "qwen2.5_0.5b": 1, "gpt2": 1, "qwen3_1.7b": 4,
    # tier: MEDIUM (4-9B, run 1 at a time for safety)
    "qwen3_4b": 10, "gptj_6b": 14, "opt_6.7b": 14,
    "llama2_7b_chat": 15, "qwen2_7b": 15, "qwen2.5_7b": 15,
    "mistral_7b_v03": 15, "falcon_7b": 15, "bloom_7b1": 15,
    "qwen3_8b": 18, "llama3.1_8b": 18, "qwen3.5_9b": 20,
    "yi_9b": 20, "glm4_9b": 22,
    # tier: LARGE (13B+, always 1 at a time)
    "llama2_13b": 28, "qwen3_14b": 30, "qwen1.5_14b": 30,
    "qwen3.5_27b": 56, "qwen3_32b": 65, "glm4_32b": 70,
    # tier: XLARGE (MoE, skip in this run)
    "qwen3_30b_a3b": 70, "qwen3.5_35b_a3b": 75,
}
MOE_MODELS = {"qwen3_30b_a3b", "qwen3.5_35b_a3b"}


# --------- Per-RQ command templates ---------
def build_cmd(model: str, rq: str, nsamples: int, savedir: str, layer_id: int = None) -> list:
    """Build subprocess command for (model, rq). Returns argv list."""
    if rq == "RQ1":
        return [
            "python3", "RQ1_attention_contribution/exp1_feasibility_test.py",
            "--model", model, "--nsamples", str(nsamples),
            "--savedir", savedir,
        ]
    elif rq == "RQ2a":
        return [
            "python3", "RQ2_mlp_source/exp2a_mlp_feasibility_test.py",
            "--model", model, "--nsamples", str(nsamples),
            "--savedir", savedir,
        ]
    elif rq == "RQ2c":
        # RQ2c uses progressive ablation in RQ6 folder
        return [
            "python3", "RQ6_single_layer_activation/exp6_progressive_ablation.py",
            "--model", model, "--nsamples", str(nsamples),
            "--savedir", savedir,
        ]
    elif rq == "RQ3":
        if layer_id is None:
            raise ValueError("RQ3 requires --layer_id")
        return [
            "python3", "RQ3_function_words/exp5_function_words_svd_mapping.py",
            "--model", model, "--nsamples", str(nsamples),
            "--layer_id", str(layer_id),
            "--savedir", savedir,
        ]
    elif rq == "RQ4":
        if layer_id is None:
            raise ValueError("RQ4 requires --layer_id")
        return [
            "python3", "RQ4_svd_alignment/exp3_svd_alignment_analysis.py",
            "--model", model, "--nsamples", str(nsamples),
            "--layer_id", str(layer_id),
            "--savedir", savedir,
        ]
    elif rq == "RQ5s":  # RQ5 single-layer V ablation
        if layer_id is None:
            raise ValueError("RQ5s requires --layer_id")
        return [
            "python3", "RQ5_v_matrix_ablation/exp5_v_ablation.py",
            "--model", model, "--nsamples", str(nsamples),
            "--layer_id", str(layer_id),
            "--savedir", savedir,
        ]
    elif rq == "RQ5m":  # RQ5 macro V ablation
        return [
            "python3", "RQ5_v_matrix_ablation/exp5_macro_v_ablation.py",
            "--model", model, "--nsamples", str(nsamples),
            "--savedir", savedir,
        ]
    elif rq == "RQ6":
        # Note: RQ6 exp6 lives in experiments/ (changeHead layout)
        return [
            "python3", "../experiments/exp6_v_ablation/exp6_v_ablation.py",
            "--model", model, "--nsamples", str(nsamples),
            "--savedir", savedir,
        ]
    else:
        raise ValueError(f"Unknown RQ: {rq}")


def load_l_origin(repo_root: Path) -> dict:
    """Read origin layer JSON."""
    p = repo_root / "paper_experiments" / "origin_layer" / "output" / "L_ORIGIN.json"
    if p.exists():
        return json.load(open(p))
    return {}


def load_origin_layers_macro(repo_root: Path) -> dict:
    """Read origin layer macro JSON."""
    p = repo_root / "paper_experiments" / "origin_layer" / "output" / "ORIGIN_LAYERS_MACRO.json"
    if p.exists():
        return json.load(open(p))
    return {}


def wait_for_vram(threshold_gb: float = 5.0, max_wait_s: int = 600, poll_s: int = 5):
    """Wait until GPU VRAM used is below threshold (GB). Returns True if ready."""
    waited = 0
    while waited < max_wait_s:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            used_mb = int(r.stdout.strip().split("\n")[0])
            used_gb = used_mb / 1024
            if used_gb < threshold_gb:
                return True
        except Exception as e:
            print(f"  [warn] nvidia-smi failed: {e}")
            return True  # fail-open, proceed anyway
        time.sleep(poll_s)
        waited += poll_s
    return False


def run_one(model: str, rq: str, args, logdir: Path, layer_id: int = None) -> dict:
    """Run one (model, rq) subprocess. Returns status dict."""
    savedir = Path(args.results_base) / rq / model
    savedir.mkdir(parents=True, exist_ok=True)

    logfile = logdir / f"{model}__{rq}.log"

    try:
        cmd = build_cmd(model, rq, args.nsamples, str(savedir), layer_id)
    except ValueError as e:
        return {"model": model, "rq": rq, "status": "skipped", "reason": str(e)}

    start = time.time()
    with open(logfile, "w") as lf:
        lf.write(f"=== {datetime.now().isoformat()} ===\n")
        lf.write(f"Command: {' '.join(cmd)}\n")
        lf.write(f"=== STDOUT/STDERR ===\n")
        lf.flush()
        try:
            result = subprocess.run(
                cmd, stdout=lf, stderr=subprocess.STDOUT,
                cwd=args.cwd, timeout=args.timeout,
            )
            ok = result.returncode == 0
        except subprocess.TimeoutExpired:
            return {
                "model": model, "rq": rq, "status": "timeout",
                "elapsed": time.time() - start,
                "log": str(logfile),
            }
        except Exception as e:
            return {
                "model": model, "rq": rq, "status": "error", "reason": str(e),
                "elapsed": time.time() - start,
                "log": str(logfile),
            }

    elapsed = time.time() - start
    return {
        "model": model, "rq": rq,
        "status": "ok" if ok else "failed",
        "returncode": result.returncode if ok else result.returncode,
        "elapsed": elapsed,
        "log": str(logfile),
        "savedir": str(savedir),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", help="Inline model list")
    parser.add_argument("--models_file", help="File with model names, one per line")
    parser.add_argument("--rqs", nargs="+", required=True,
                        choices=["RQ1", "RQ2a", "RQ2c", "RQ3", "RQ4", "RQ5s", "RQ5m", "RQ6"])
    parser.add_argument("--nsamples", type=int, default=30)
    parser.add_argument("--cwd", default="paper_experiments",
                        help="Working directory for subprocess (where RQ scripts live)")
    parser.add_argument("--results_base", default="results/wikitext_run_2026_04_21",
                        help="Root for results (relative to --cwd)")
    parser.add_argument("--logdir_base", default="logs",
                        help="Root for logs (relative to --cwd)")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Per-run subprocess timeout (seconds)")
    parser.add_argument("--skip_moe", action="store_true", default=True,
                        help="Skip MoE models (Tier C, not in main sample)")
    parser.add_argument("--vram_threshold_gb", type=float, default=5.0,
                        help="Wait until VRAM usage < this before starting next run")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would run without executing")
    args = parser.parse_args()

    # Collect models
    models = []
    if args.models:
        models.extend(args.models)
    if args.models_file:
        with open(args.models_file) as f:
            models.extend(line.strip() for line in f if line.strip() and not line.startswith("#"))
    if not models:
        print("Error: no models specified", file=sys.stderr)
        sys.exit(1)

    # Filter MoE
    if args.skip_moe:
        pre = len(models)
        models = [m for m in models if m not in MOE_MODELS]
        if len(models) < pre:
            print(f"[info] Skipped {pre - len(models)} MoE models ({MOE_MODELS & set(models) ^ MOE_MODELS})")

    # Timestamp run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Paths are relative to cwd
    repo_root = Path(args.cwd).resolve().parent
    logdir = Path(args.cwd) / args.logdir_base / f"pipeline_{timestamp}"
    logdir.mkdir(parents=True, exist_ok=True)
    driver_log = logdir / "pipeline.log"

    l_origin = load_l_origin(repo_root)

    # Sort models by size (smallest first — fast feedback, OOM issues come later)
    models = sorted(models, key=lambda m: MODEL_SIZES_GB.get(m, 20))

    print(f"Pipeline started {timestamp}")
    print(f"  Models: {len(models)} ({', '.join(models)})")
    print(f"  RQs:    {args.rqs}")
    print(f"  nsamples={args.nsamples}, timeout={args.timeout}s, skip_moe={args.skip_moe}")
    print(f"  Logs:   {logdir}")
    print(f"  Results root: {Path(args.cwd) / args.results_base}")
    print()

    results = []
    total = len(models) * len(args.rqs)
    idx = 0

    for model in models:
        model_size = MODEL_SIZES_GB.get(model, 20)
        for rq in args.rqs:
            idx += 1
            layer_id = None
            if rq in ("RQ3", "RQ4", "RQ5s"):
                layer_id = l_origin.get(model)
                if layer_id is None:
                    print(f"[{idx}/{total}] {model} {rq} — SKIP (no L_ORIGIN)")
                    results.append({"model": model, "rq": rq, "status": "skipped",
                                    "reason": "no L_ORIGIN.json entry"})
                    continue

            label = f"{model:<22} {rq:<6} size≈{model_size}GB"
            if layer_id is not None:
                label += f" L={layer_id}"

            if args.dry_run:
                print(f"[{idx}/{total}] DRY-RUN: {label}")
                continue

            # Wait for VRAM to clear (critical for sequential subprocess safety)
            if not wait_for_vram(args.vram_threshold_gb):
                print(f"  [warn] VRAM didn't clear in time; proceeding anyway")

            print(f"[{idx}/{total}] {label} — RUNNING")
            t_start = time.time()
            res = run_one(model, rq, args, logdir, layer_id)
            elapsed = time.time() - t_start
            results.append(res)

            status_icon = {"ok": "✓", "failed": "✗", "timeout": "⏱", "skipped": "○", "error": "⚠"}.get(
                res["status"], "?")
            print(f"  {status_icon} {res['status']} ({elapsed:.0f}s)")

            # Save summary after each run (incremental)
            with open(logdir / "summary.json", "w") as f:
                json.dump({
                    "timestamp": timestamp,
                    "total": total, "done": idx,
                    "args": vars(args),
                    "results": results,
                }, f, indent=2)

    # Final summary
    ok = sum(1 for r in results if r["status"] == "ok")
    fail = sum(1 for r in results if r["status"] in ("failed", "error", "timeout"))
    skip = sum(1 for r in results if r["status"] == "skipped")
    print()
    print(f"=== Pipeline done ===")
    print(f"  OK: {ok}  FAIL: {fail}  SKIP: {skip}  (total {len(results)})")
    print(f"  Summary: {logdir / 'summary.json'}")

    if fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
