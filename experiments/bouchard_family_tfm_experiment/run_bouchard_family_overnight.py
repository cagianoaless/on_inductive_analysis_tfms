#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


@dataclass(frozen=True)
class Step:
    name: str
    cmd: list[str]
    log_path: Path
    output_dir: Path


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def experiment_root() -> Path:
    return Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    root = experiment_root()
    workspace_root = root.parent
    external_root = workspace_root.parent

    parser = argparse.ArgumentParser(
        description="Run the Bouchard family TFM experiment as a smoke check plus a paper-style full suite."
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to launch child runs.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=root / "results" / f"overnight_{utc_timestamp()}",
        help="Directory for logs and per-step outputs.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=workspace_root / "datasets" / "bouchard_family_relational",
        help="Directory containing the generated family dataset.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="tabpfn,tabicl",
        help="Comma-separated model list forwarded to the experiment script.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--holdout-family-id",
        type=str,
        default="family_05",
        help="Held-out family used for the family split.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of runs for the full paper-style suite.",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.5,
        help="Fixed decision threshold forwarded to the experiment script.",
    )
    parser.add_argument(
        "--threshold-mode",
        type=str,
        default="fixed",
        choices=["fixed", "tune_global_f1", "tune_per_relation_f1"],
        help="Threshold mode forwarded to the experiment script.",
    )
    parser.add_argument(
        "--tabpfn-model-path",
        type=Path,
        default=external_root / "binary_gbdt_vs_tabpfn" / "tabpfn-v2.5-classifier-v2.5_default-2.ckpt",
    )
    parser.add_argument("--tabpfn-n-estimators", type=int, default=4)
    parser.add_argument(
        "--tabpfn-fit-mode",
        type=str,
        default="batched",
        choices=["low_memory", "fit_preprocessors", "fit_with_cache", "batched"],
    )
    parser.add_argument("--tabpfn-memory-saving-mode", type=str, default="auto")
    parser.add_argument("--tabpfn-inference-precision", type=str, default="autocast")
    parser.add_argument(
        "--tabicl-model-path",
        type=Path,
        default=external_root / "TabPFNvsTabICL" / "tabicl-classifier-v2-20260212.ckpt",
    )
    parser.add_argument("--tabicl-n-estimators", type=int, default=4)
    parser.add_argument("--tabicl-batch-size", type=int, default=16)
    parser.add_argument("--tabicl-use-amp", type=str, default="auto")
    parser.add_argument("--tabicl-use-fa3", type=str, default="auto")
    parser.add_argument("--tabicl-offload-mode", type=str, default="auto")
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip the smoke step.",
    )
    parser.add_argument(
        "--skip-full",
        action="store_true",
        help="Skip the full paper-style run.",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run only the smoke step.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.data_root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {args.data_root}")
    if not (args.data_root / "pair_relation_atoms.csv").exists():
        raise FileNotFoundError(f"pair_relation_atoms.csv not found under {args.data_root}")
    if not args.tabpfn_model_path.exists():
        raise FileNotFoundError(f"TabPFN checkpoint not found: {args.tabpfn_model_path}")
    if not args.tabicl_model_path.exists():
        raise FileNotFoundError(f"TabICL checkpoint not found: {args.tabicl_model_path}")
    if args.num_runs <= 0:
        raise ValueError("--num-runs must be positive.")


def base_command(args: argparse.Namespace) -> list[str]:
    root = experiment_root()
    return [
        args.python,
        str(root / "run_bouchard_family_experiment.py"),
        "--data-root",
        str(args.data_root.resolve()),
        "--models",
        args.models,
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--holdout-family-id",
        args.holdout_family_id,
        "--decision-threshold",
        str(args.decision_threshold),
        "--threshold-mode",
        args.threshold_mode,
        "--tabpfn-model-path",
        str(args.tabpfn_model_path),
        "--tabpfn-n-estimators",
        str(args.tabpfn_n_estimators),
        "--tabpfn-fit-mode",
        args.tabpfn_fit_mode,
        "--tabpfn-memory-saving-mode",
        args.tabpfn_memory_saving_mode,
        "--tabpfn-inference-precision",
        args.tabpfn_inference_precision,
        "--tabicl-model-path",
        str(args.tabicl_model_path),
        "--tabicl-n-estimators",
        str(args.tabicl_n_estimators),
        "--tabicl-batch-size",
        str(args.tabicl_batch_size),
        "--tabicl-use-amp",
        args.tabicl_use_amp,
        "--tabicl-use-fa3",
        args.tabicl_use_fa3,
        "--tabicl-offload-mode",
        args.tabicl_offload_mode,
    ]


def make_step(name: str, cmd: list[str], run_root: Path) -> Step:
    return Step(
        name=name,
        cmd=cmd,
        log_path=run_root / "logs" / f"{name}.log",
        output_dir=run_root / name,
    )


def build_steps(args: argparse.Namespace) -> list[Step]:
    root = args.run_root.resolve()
    steps: list[Step] = []
    common = base_command(args)

    if not args.skip_smoke:
        smoke_cmd = common + [
            "--output-dir",
            str(root / "smoke_family_p10"),
            "--splits",
            "family",
            "--p-values",
            "0.1",
            "--num-runs",
            "1",
            "--max-train-rows",
            "12000",
            "--max-valid-rows",
            "1000",
            "--max-test-rows",
            "2500",
        ]
        steps.append(make_step("smoke_family_p10", smoke_cmd, root))

    if args.smoke_only:
        return steps

    if not args.skip_full:
        full_cmd = common + [
            "--output-dir",
            str(root / "paper_full"),
            "--splits",
            "random,evidence,family",
            "--p-values",
            "0.8,0.4,0.2,0.1",
            "--include-family-zero",
            "--num-runs",
            str(args.num_runs),
        ]
        steps.append(make_step("paper_full", full_cmd, root))

    if not steps:
        raise ValueError("No steps selected.")
    return steps


def write_manifest(
    *,
    args: argparse.Namespace,
    steps: list[Step],
    statuses: list[dict[str, object]],
) -> None:
    payload = {
        "launcher": "run_bouchard_family_overnight.py",
        "created_utc": utc_timestamp(),
        "run_root": str(args.run_root.resolve()),
        "steps": [
            {
                "name": step.name,
                "command": step.cmd,
                "log_path": str(step.log_path),
                "output_dir": str(step.output_dir),
            }
            for step in steps
        ],
        "statuses": statuses,
        "config": {
            "data_root": str(args.data_root.resolve()),
            "device": args.device,
            "models": args.models,
            "seed": int(args.seed),
            "holdout_family_id": args.holdout_family_id,
            "num_runs": int(args.num_runs),
            "decision_threshold": float(args.decision_threshold),
            "threshold_mode": args.threshold_mode,
            "tabpfn_model_path": str(args.tabpfn_model_path.resolve()),
            "tabicl_model_path": str(args.tabicl_model_path.resolve()),
        },
    }
    manifest_path = args.run_root.resolve() / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2))


def run_step(step: Step, *, env: dict[str, str]) -> int:
    step.log_path.parent.mkdir(parents=True, exist_ok=True)
    step.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[overnight] starting {step.name}", flush=True)
    print(f"[overnight] command: {shlex.join(step.cmd)}", flush=True)

    with step.log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {shlex.join(step.cmd)}\n\n")
        log_file.flush()

        process = subprocess.Popen(
            step.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        return process.wait()


def main() -> int:
    args = parse_args()
    validate_args(args)
    args.run_root = args.run_root.resolve()
    steps = build_steps(args)

    if args.dry_run:
        for step in steps:
            print(shlex.join(step.cmd))
        return 0

    args.run_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["TABPFN_DISABLE_TELEMETRY"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    statuses: list[dict[str, object]] = []
    write_manifest(args=args, steps=steps, statuses=statuses)

    for step in steps:
        status = {"name": step.name, "status": "running"}
        statuses.append(status)
        write_manifest(args=args, steps=steps, statuses=statuses)

        return_code = run_step(step, env=env)
        status["return_code"] = int(return_code)
        status["status"] = "ok" if return_code == 0 else "failed"
        write_manifest(args=args, steps=steps, statuses=statuses)
        if return_code != 0:
            print(f"[overnight] step failed: {step.name}", flush=True)
            return return_code

    print(f"[overnight] completed successfully: {args.run_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
