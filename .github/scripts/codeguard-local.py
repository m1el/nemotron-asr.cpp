#!/usr/bin/env python3
#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "unidiff>=0.7.5",
# ]
# ///

"""
Local CodeGuard runner — exercises the same DiffAnalyzer + RiskClassifier +
evidence-bundle logic that the GitHub Action uses, but against a local git diff.

Usage:
    uv run .github/scripts/codeguard-local.py                   # diff HEAD~1..HEAD
    uv run .github/scripts/codeguard-local.py --ref HEAD~3      # diff HEAD~3..HEAD
    uv run .github/scripts/codeguard-local.py --ref main         # diff main..HEAD
    uv run .github/scripts/codeguard-local.py --diff /path/to/file.patch
"""

import argparse
import hashlib
import json
import subprocess
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the codeguard-action source is importable
# ---------------------------------------------------------------------------
ACTION_DIR = Path(__file__).resolve().parents[2] / ".." / "codeguard-action"
if not ACTION_DIR.exists():
    # Try sibling location under codegauard/
    ACTION_DIR = Path("/var/data/codegauard/codeguard-action")

if ACTION_DIR.exists():
    sys.path.insert(0, str(ACTION_DIR))
else:
    print(f"ERROR: codeguard-action not found at {ACTION_DIR}", file=sys.stderr)
    sys.exit(1)

from src.analyzer import DiffAnalyzer
from src.risk_classifier import RiskClassifier

# ---------------------------------------------------------------------------
# Minimal evidence-bundle generator (no PyGithub dependency)
# ---------------------------------------------------------------------------

def compute_event_hash(event: dict, previous_hash: str) -> str:
    content = json.dumps(
        {
            "event_type": event["event_type"],
            "timestamp": event["timestamp"],
            "actor": event["actor"],
            "data": event["data"],
            "previous_hash": previous_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(content.encode()).hexdigest()


def build_local_bundle(
    diff_content: str,
    analysis: dict,
    risk_result: dict,
    repo_name: str,
    commit_sha: str,
) -> dict:
    """Build an evidence bundle without requiring a GitHub PullRequest object."""
    now = datetime.now(timezone.utc).isoformat()
    bundle_id_raw = f"{repo_name}:local:{commit_sha}:{now}"
    bundle_id = f"gsb_{hashlib.sha256(bundle_id_raw.encode()).hexdigest()[:12]}"

    events = []
    previous_hash = ""

    # Event 1 — diff submitted
    e1 = {
        "event_type": "local_diff_submitted",
        "timestamp": now,
        "actor": "codeguard-local",
        "data": {
            "repository": repo_name,
            "commit_sha": commit_sha,
            "diff_bytes": len(diff_content),
        },
    }
    e1["hash"] = compute_event_hash(e1, previous_hash)
    previous_hash = e1["hash"]
    events.append(e1)

    # Event 2 — analysis completed
    e2 = {
        "event_type": "analysis_completed",
        "timestamp": now,
        "actor": "guardspine-codeguard",
        "data": {
            "files_changed": analysis.get("files_changed", 0),
            "lines_added": analysis.get("lines_added", 0),
            "lines_removed": analysis.get("lines_removed", 0),
            "sensitive_zones_count": len(analysis.get("sensitive_zones", [])),
            "diff_hash": analysis.get("diff_hash", ""),
        },
    }
    e2["hash"] = compute_event_hash(e2, previous_hash)
    previous_hash = e2["hash"]
    events.append(e2)

    # Event 3 — risk classified
    e3 = {
        "event_type": "risk_classified",
        "timestamp": now,
        "actor": "guardspine-codeguard",
        "data": {
            "risk_tier": risk_result.get("risk_tier", "L2"),
            "findings_count": len(risk_result.get("findings", [])),
            "scores": risk_result.get("scores", {}),
        },
    }
    e3["hash"] = compute_event_hash(e3, previous_hash)
    previous_hash = e3["hash"]
    events.append(e3)

    return {
        "guardspine_spec_version": "1.0.0",
        "bundle_id": bundle_id,
        "created_at": now,
        "context": {
            "repository": repo_name,
            "commit_sha": commit_sha,
            "mode": "local",
        },
        "events": events,
        "hash_chain": {
            "algorithm": "sha256",
            "final_hash": previous_hash,
            "event_count": len(events),
        },
        "summary": {
            "risk_tier": risk_result.get("risk_tier", "L2"),
            "risk_drivers": risk_result.get("risk_drivers", []),
            "findings": risk_result.get("findings", []),
            "rationale": risk_result.get("rationale", ""),
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def git(*args: str, cwd: str | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    if result.returncode != 0:
        print(f"git {' '.join(args)} failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result.stdout


def get_repo_name(cwd: str) -> str:
    try:
        url = git("remote", "get-url", "origin", cwd=cwd).strip()
        # git@github.com:user/repo.git  or  https://github.com/user/repo.git
        name = url.rstrip(".git").rsplit("/", 1)[-1]
        owner = url.rstrip(".git").rsplit("/", 2)[-2]
        owner = owner.split(":")[-1]  # handle ssh form
        return f"{owner}/{name}"
    except SystemExit:
        return "local/nemotron-asr.cpp"


# ---------------------------------------------------------------------------
# Pretty console output
# ---------------------------------------------------------------------------

TIER_BADGE = {
    "L0": "\033[92m[L0 Trivial]\033[0m",
    "L1": "\033[94m[L1 Low]\033[0m",
    "L2": "\033[93m[L2 Medium]\033[0m",
    "L3": "\033[33m[L3 High]\033[0m",
    "L4": "\033[91m[L4 Critical]\033[0m",
}


def print_report(analysis: dict, risk_result: dict) -> None:
    tier = risk_result["risk_tier"]
    badge = TIER_BADGE.get(tier, tier)

    print()
    print("=" * 60)
    print(f"  GuardSpine CodeGuard — Local Analysis")
    print("=" * 60)
    print()
    print(f"  Risk tier:       {badge}")
    print(f"  Files changed:   {analysis.get('files_changed', 0)}")
    print(f"  Lines added:     {analysis.get('lines_added', 0)}")
    print(f"  Lines removed:   {analysis.get('lines_removed', 0)}")
    print(f"  Sensitive zones: {len(analysis.get('sensitive_zones', []))}")
    print()

    drivers = risk_result.get("risk_drivers", [])
    if drivers:
        print("  Risk drivers:")
        for d in drivers[:5]:
            print(f"    - {d.get('description', 'N/A')}")
        print()

    findings = risk_result.get("findings", [])
    if findings:
        print(f"  Findings ({len(findings)}):")
        for f in findings[:10]:
            sev = f.get("severity", "?").upper()
            msg = f.get("message", "")
            loc = f.get("file", "?")
            line = f.get("line")
            loc_str = f"{loc}:{line}" if line else loc
            print(f"    [{sev}] {loc_str} — {msg}")
        if len(findings) > 10:
            print(f"    ... and {len(findings) - 10} more")
        print()

    print(f"  Rationale: {risk_result.get('rationale', 'N/A')}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run CodeGuard locally on a git diff")
    parser.add_argument(
        "--ref",
        default="HEAD~1",
        help="Git ref to diff against HEAD (default: HEAD~1)",
    )
    parser.add_argument(
        "--diff",
        type=Path,
        help="Path to a .patch / .diff file instead of using git diff",
    )
    parser.add_argument(
        "--rubric",
        default="default",
        choices=["default", "soc2", "hipaa", "pci-dss"],
        help="Policy rubric (default: default)",
    )
    parser.add_argument(
        "--threshold",
        default="L3",
        help="Risk tier threshold for pass/fail (default: L3)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write evidence bundle JSON to this path",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full results as JSON instead of human-readable report",
    )
    args = parser.parse_args()

    repo_root = git("rev-parse", "--show-toplevel").strip()

    # ---- Get the diff ----
    if args.diff:
        diff_content = args.diff.read_text()
    else:
        diff_content = git("diff", f"{args.ref}..HEAD", cwd=repo_root)
        if not diff_content.strip():
            # Try staged changes
            diff_content = git("diff", "--cached", cwd=repo_root)
        if not diff_content.strip():
            print("No diff found. Try --ref HEAD~2 or --diff path/to/file.patch")
            sys.exit(0)

    commit_sha = git("rev-parse", "HEAD", cwd=repo_root).strip()
    repo_name = get_repo_name(repo_root)

    # ---- Analyze ----
    analyzer = DiffAnalyzer(ai_review=True)
    analysis = analyzer.analyze(diff_content, rubric=args.rubric)

    # ---- Classify risk ----
    classifier = RiskClassifier(rubric=args.rubric)
    risk_result = classifier.classify(analysis)

    # ---- Build evidence bundle ----
    bundle = build_local_bundle(
        diff_content=diff_content,
        analysis=analysis,
        risk_result=risk_result,
        repo_name=repo_name,
        commit_sha=commit_sha,
    )

    # ---- Write bundle to file ----
    bundle_path = args.output
    if not bundle_path:
        out_dir = Path(repo_root) / ".guardspine"
        out_dir.mkdir(exist_ok=True)
        bundle_path = out_dir / f"bundle-local-{commit_sha[:7]}.json"

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(json.dumps(bundle, indent=2, default=str))

    # ---- Output ----
    if args.json:
        print(json.dumps({"analysis": analysis, "risk": risk_result, "bundle": bundle}, indent=2, default=str))
    else:
        print_report(analysis, risk_result)
        print(f"  Bundle ID:   {bundle['bundle_id']}")
        print(f"  Hash chain:  {bundle['hash_chain']['final_hash'][:16]}...")
        print()
        print(f"  Evidence bundle written to: {bundle_path}")

    # ---- Exit code ----
    tier_order = ["L0", "L1", "L2", "L3", "L4"]
    threshold_idx = tier_order.index(args.threshold)
    risk_idx = tier_order.index(risk_result["risk_tier"])

    if risk_idx >= threshold_idx:
        if not args.json:
            print(f"\n  FAIL: Risk {risk_result['risk_tier']} >= threshold {args.threshold}")
        sys.exit(1)
    else:
        if not args.json:
            print(f"\n  PASS: Risk {risk_result['risk_tier']} < threshold {args.threshold}")
        sys.exit(0)


if __name__ == "__main__":
    main()
