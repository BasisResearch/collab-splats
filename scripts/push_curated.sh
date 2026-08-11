#!/bin/bash
# Push the flat curated video tree to the environments-curated GCS bucket via rclone.
#
# Expects the layout produced by scripts/preprocess_gdrive_videos.py:
#   <source>/YYYY_MM_DD-PARENT-VIDEO/<video>.mp4
#
# Re-runnable: rclone skips objects that already match on size and modtime, so a
# second run transfers nothing.
#
# Usage:
#   ./scripts/push_curated.sh --dry-run          # show what would transfer
#   ./scripts/push_curated.sh                    # push
#   ./scripts/push_curated.sh --source /data/x --bucket other-bucket
#
# One-time setup (see README "Data access"):
#   brew install rclone jq
#   rclone config create collab-data "google cloud storage" \
#     service_account_file="$PWD/config-local/collab-data.json" \
#     project_number=collab-data-463313 bucket_policy_only=true
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SOURCE="${REPO_ROOT}/../environments-curated"
REMOTE="collab-data"
BUCKET="environments-curated"
DRY_RUN=""

# Parse args; unknown flags are a hard error rather than a silent no-op
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN="--dry-run"; shift ;;
        --source)  SOURCE="$2"; shift 2 ;;
        --remote)  REMOTE="$2"; shift 2 ;;
        --bucket)  BUCKET="$2"; shift 2 ;;
        # Print the header comment block (everything after the shebang up to the first
        # non-comment line), stripped of its leading "# "
        -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# Preflight: fail early with the fix, not with an opaque rclone error
if ! command -v rclone >/dev/null 2>&1; then
    echo "rclone not found. Install it: brew install rclone jq" >&2
    exit 1
fi
if ! rclone listremotes | grep -qx "${REMOTE}:"; then
    echo "rclone remote '${REMOTE}' not configured. See the setup block in this script's header." >&2
    exit 1
fi
if [[ ! -d "$SOURCE" ]]; then
    echo "source directory does not exist: $SOURCE" >&2
    exit 1
fi

echo "Pushing ${SOURCE} -> ${REMOTE}:${BUCKET} ${DRY_RUN}"

# Flags mirror SessionSource.push_outputs (collab_splats/dashboard/sources.py) so uploads
# behave identically to the dashboard's. No --progress: it redraws with carriage returns,
# which is unreadable once the output is piped or logged; --stats-one-line covers progress.
rclone copy "$SOURCE" "${REMOTE}:${BUCKET}" \
    --gcs-bucket-policy-only \
    --transfers 8 \
    --retries 3 \
    --timeout 300s \
    --contimeout 60s \
    --stats 2s \
    --stats-one-line \
    --exclude ".DS_Store" \
    ${DRY_RUN}

# Report what actually landed remotely
echo
rclone size "${REMOTE}:${BUCKET}"
