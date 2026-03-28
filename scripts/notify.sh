#!/bin/bash
set -euo pipefail
# Slack webhook通知スクリプト
# Usage: ./scripts/notify.sh "メッセージ" [success|failure]
# Requires: SLACK_WEBHOOK_URL environment variable

WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"
if [ -z "$WEBHOOK_URL" ]; then
  echo "SLACK_WEBHOOK_URL not set, skipping notification" >&2
  exit 0
fi
MESSAGE="${1:?Usage: $0 \"message\" [success|failure]}"
STATUS="${2:-success}"
HOSTNAME=$(hostname)
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

if [ "$STATUS" = "success" ]; then
  COLOR="#36a64f"
else
  COLOR="#dc3545"
fi

# Build JSON payload safely
if command -v jq &> /dev/null; then
  PAYLOAD=$(jq -n \
    --arg color "$COLOR" \
    --arg text "$MESSAGE" \
    --arg footer "$HOSTNAME | $TIMESTAMP" \
    '{ attachments: [{ color: $color, text: $text, footer: $footer }] }')
else
  # Fallback: escape special characters manually
  ESCAPED_MESSAGE=$(printf '%s' "$MESSAGE" | sed 's/\\/\\\\/g; s/"/\\"/g')
  ESCAPED_FOOTER=$(printf '%s' "$HOSTNAME | $TIMESTAMP" | sed 's/\\/\\\\/g; s/"/\\"/g')
  PAYLOAD="{\"attachments\":[{\"color\":\"$COLOR\",\"text\":\"$ESCAPED_MESSAGE\",\"footer\":\"$ESCAPED_FOOTER\"}]}"
fi

if ! curl -s -f -X POST "$WEBHOOK_URL" \
  -H 'Content-Type: application/json' \
  -d "$PAYLOAD"; then
  echo "ERROR: Failed to send Slack notification" >&2
  exit 1
fi
