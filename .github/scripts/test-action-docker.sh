#!/usr/bin/env bash
#
# Simulate a GitHub Actions PR event locally using the codeguard-action Docker image.
#
# Usage:
#   .github/scripts/test-action-docker.sh              # uses HEAD~1..HEAD diff
#   .github/scripts/test-action-docker.sh HEAD~5       # uses HEAD~5..HEAD diff
#
set -euo pipefail

REF="${1:-HEAD~1}"
REPO_ROOT="$(git rev-parse --show-toplevel)"
SHA="$(git rev-parse HEAD)"
REPO_NAME="$(git remote get-url origin 2>/dev/null | sed 's|.*[:/]\([^/]*/[^/]*\)\.git$|\1|' || echo 'local/nemotron-asr.cpp')"

echo "=== CodeGuard Docker Integration Test ==="
echo "  Repo:   $REPO_NAME"
echo "  Diff:   $REF..HEAD"
echo "  SHA:    $SHA"
echo ""

# ---- Build the action image (if not already built) ----
ACTION_DIR="${REPO_ROOT}/../codeguard-action"
if [ ! -d "$ACTION_DIR" ]; then
    ACTION_DIR="/var/data/codegauard/codeguard-action"
fi

if ! docker image inspect codeguard-action:latest >/dev/null 2>&1; then
    echo "Building codeguard-action Docker image..."
    docker build -t codeguard-action "$ACTION_DIR"
fi

# ---- Create a fake GitHub event payload ----
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

cat > "$TMPDIR/event.json" <<EVEOF
{
  "pull_request": {
    "number": 999,
    "title": "Local test PR ($REF..HEAD)",
    "user": { "login": "local-tester" },
    "head": { "ref": "test-branch", "sha": "$SHA" },
    "base": { "ref": "master" },
    "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "diff_url": "file:///workspace/.codeguard-diff.patch",
    "html_url": "http://localhost/pr/999"
  }
}
EVEOF

# ---- Generate the diff and place it where the container can see it ----
git diff "$REF..HEAD" > "$TMPDIR/diff.patch"
DIFF_LINES=$(wc -l < "$TMPDIR/diff.patch")
echo "  Diff:    $DIFF_LINES lines"

# ---- Create a minimal Python shim that runs the core pipeline ----
#  The real entrypoint.py calls GitHub API to fetch the diff and post comments.
#  We replace it with a shim that reads the diff from a local file.
cat > "$TMPDIR/local_entrypoint.py" <<'PYEOF'
#!/usr/bin/env python3
"""Shim entrypoint that reads diff from local file instead of GitHub API."""
import json, os, sys, hashlib
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "/action")
from src.analyzer import DiffAnalyzer
from src.risk_classifier import RiskClassifier

def main():
    risk_threshold = os.environ.get("INPUT_RISK_THRESHOLD", "L3")
    rubric = os.environ.get("INPUT_RUBRIC", "default")

    event_path = os.environ.get("GITHUB_EVENT_PATH", "")
    with open(event_path) as f:
        event = json.load(f)

    pr = event.get("pull_request", {})
    print(f"::group::GuardSpine CodeGuard Analysis")
    print(f"PR: #{pr.get('number', 0)}")
    print(f"Risk threshold: {risk_threshold}")
    print(f"Rubric: {rubric}")
    print(f"::endgroup::")

    diff_file = os.environ.get("CODEGUARD_DIFF_FILE", "/workspace/diff.patch")
    diff_content = Path(diff_file).read_text()
    print(f"Diff size: {len(diff_content)} bytes")

    analyzer = DiffAnalyzer(ai_review=False)
    analysis = analyzer.analyze(diff_content, rubric=rubric)
    print(f"Files changed: {analysis['files_changed']}")
    print(f"Lines added: {analysis['lines_added']}")
    print(f"Lines removed: {analysis['lines_removed']}")

    classifier = RiskClassifier(rubric=rubric)
    risk_result = classifier.classify(analysis)
    risk_tier = risk_result["risk_tier"]
    print(f"Risk tier: {risk_tier}")
    print(f"Findings: {len(risk_result.get('findings', []))}")
    print(f"Rationale: {risk_result.get('rationale', '')}")

    tier_order = ["L0", "L1", "L2", "L3", "L4"]
    threshold_index = tier_order.index(risk_threshold)
    risk_index = tier_order.index(risk_tier)
    requires_approval = risk_index >= threshold_index

    # Build evidence bundle
    sha = os.environ.get("GITHUB_SHA", "local")[:7]
    now = datetime.now(timezone.utc).isoformat()
    bundle_id = f"gsb_{hashlib.sha256(f'{now}{sha}'.encode()).hexdigest()[:12]}"

    events_list = []
    prev = ""
    for ev in [
        {"event_type": "pr_submitted", "timestamp": now, "actor": pr.get("user",{}).get("login","?"),
         "data": {"pr_number": pr.get("number",0), "title": pr.get("title",""), "head_sha": sha}},
        {"event_type": "analysis_completed", "timestamp": now, "actor": "guardspine-codeguard",
         "data": {"files_changed": analysis["files_changed"], "lines_added": analysis["lines_added"],
                  "lines_removed": analysis["lines_removed"], "diff_hash": analysis.get("diff_hash","")}},
        {"event_type": "risk_classified", "timestamp": now, "actor": "guardspine-codeguard",
         "data": {"risk_tier": risk_tier, "findings_count": len(risk_result.get("findings",[]))}},
    ]:
        content = json.dumps({**{k:ev[k] for k in ["event_type","timestamp","actor","data"]}, "previous_hash": prev},
                             sort_keys=True, separators=(",",":"))
        h = hashlib.sha256(content.encode()).hexdigest()
        ev["hash"] = h
        prev = h
        events_list.append(ev)

    bundle = {
        "guardspine_spec_version": "1.0.0", "bundle_id": bundle_id, "created_at": now,
        "events": events_list,
        "hash_chain": {"algorithm": "sha256", "final_hash": prev, "event_count": len(events_list)},
        "summary": {"risk_tier": risk_tier, "risk_drivers": risk_result.get("risk_drivers",[]),
                     "findings": risk_result.get("findings",[]), "rationale": risk_result.get("rationale","")},
    }

    out_dir = Path("/workspace/.guardspine")
    out_dir.mkdir(exist_ok=True)
    bundle_path = out_dir / f"bundle-docker-pr{pr.get('number',0)}-{sha}.json"
    bundle_path.write_text(json.dumps(bundle, indent=2, default=str))
    print(f"\nBundle saved: {bundle_path}")
    print(f"Bundle ID: {bundle_id}")

    if requires_approval:
        print(f"\n::warning::Risk {risk_tier} >= threshold {risk_threshold} — would BLOCK merge")
        sys.exit(1)
    else:
        print(f"\n::notice::Risk {risk_tier} < threshold {risk_threshold} — would PASS")
        sys.exit(0)

if __name__ == "__main__":
    main()
PYEOF

# ---- Run the container ----
echo ""
echo "--- Running codeguard-action in Docker ---"
echo ""

set +e
docker run --rm \
    -v "$TMPDIR/event.json:/github/event.json:ro" \
    -v "$TMPDIR/diff.patch:/workspace/diff.patch:ro" \
    -v "$TMPDIR/local_entrypoint.py:/action/local_entrypoint.py:ro" \
    -v "$REPO_ROOT/.guardspine:/workspace/.guardspine" \
    -e GITHUB_EVENT_PATH=/github/event.json \
    -e GITHUB_REPOSITORY="$REPO_NAME" \
    -e GITHUB_SHA="$SHA" \
    -e INPUT_RISK_THRESHOLD=L3 \
    -e INPUT_RUBRIC=default \
    -e INPUT_POST_COMMENT=false \
    -e INPUT_GENERATE_BUNDLE=true \
    -e INPUT_FAIL_ON_HIGH_RISK=true \
    -e INPUT_AI_REVIEW=false \
    -e CODEGUARD_DIFF_FILE=/workspace/diff.patch \
    --entrypoint python \
    codeguard-action:latest \
    /action/local_entrypoint.py

EXIT_CODE=$?
set -e

echo ""
echo "--- Docker test complete (exit: $EXIT_CODE) ---"

# Show the bundle
BUNDLE_FILE=$(ls -t "$REPO_ROOT/.guardspine"/bundle-docker-*.json 2>/dev/null | head -1)
if [ -n "$BUNDLE_FILE" ]; then
    echo ""
    echo "Evidence bundle: $BUNDLE_FILE"
    python3 -c "
import json, hashlib
with open('$BUNDLE_FILE') as f:
    b = json.load(f)
prev = ''
ok = True
for ev in b['events']:
    content = json.dumps({k:ev[k] for k in ['event_type','timestamp','actor','data']} | {'previous_hash': prev},
                         sort_keys=True, separators=(',',':'))
    h = hashlib.sha256(content.encode()).hexdigest()
    if h != ev['hash']:
        print(f'  HASH MISMATCH event {ev[\"event_type\"]}')
        ok = False
    prev = h
if prev != b['hash_chain']['final_hash']:
    print('  FINAL HASH MISMATCH')
    ok = False
if ok:
    print('  Hash chain: VERIFIED')
print(f'  Risk tier:  {b[\"summary\"][\"risk_tier\"]}')
print(f'  Findings:   {len(b[\"summary\"][\"findings\"])}')
"
fi

exit $EXIT_CODE
