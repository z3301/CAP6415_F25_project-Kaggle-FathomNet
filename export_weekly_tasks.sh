#!/bin/bash
# ============================================================
#  export_weekly_tasks.sh
#  Exports all GitHub issue fields grouped by milestone
#  and writes weekly logs to /docs/weekNlog.txt
# ============================================================

# --- CONFIGURATION ---
REPO="z3301/CAP6415_F25_project-Kaggle-FathomNet"
DOCS_DIR="./docs"
MILESTONES=("Week 1" "Week 2" "Week 3" "Week 4" "Week 5")

# --- DEPENDENCY CHECK ---
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) not found. Install it with: brew install gh"
    exit 1
fi

mkdir -p "$DOCS_DIR"

# --- EXPORT LOOP ---
for MILESTONE in "${MILESTONES[@]}"; do
    WEEK=$(echo "$MILESTONE" | grep -oE '[0-9]+')
    OUTPUT_FILE="$DOCS_DIR/week${WEEK}log.txt"

    echo "📝 Exporting tasks for $MILESTONE..."
    {
        echo "==================================================="
        echo "FathomNet 2025 Project — Weekly Log $WEEK"
        echo "Generated on: $(date)"
        echo "===================================================\n"
    } > "$OUTPUT_FILE"

    # Export all issue fields
    gh issue list --repo "$REPO" --milestone "$MILESTONE" --state all \
      --json author,assignees,body,closedAt,createdAt,labels,milestone,number,state,title,updatedAt,url,comments \
      --jq '.[] |
        "• Title: \(.title)\n" +
        "  Number: #\(.number)\n" +
        "  URL: \(.url)\n" +
        "  Author: \(.author.login // "unknown")\n" +
        "  Assignees: \([.assignees[].login] | join(", "))\n" +
        "  Labels: \([.labels[].name] | join(", "))\n" +
        "  State: \(.state)\n" +
        "  Milestone: \(.milestone.title // "none")\n" +
        "  Created At: \(.createdAt)\n" +
        "  Updated At: \(.updatedAt)\n" +
        "  Closed At: \(.closedAt // "still open")\n" +
        "  Description:\n\(.body // "No description")\n" +
        "------------------------------------------------------------\n"' \
      >> "$OUTPUT_FILE"

    echo "✅ Saved: $OUTPUT_FILE"
done

echo "🎉 All weekly logs exported successfully to $DOCS_DIR/"
