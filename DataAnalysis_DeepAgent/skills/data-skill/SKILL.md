---
name: data-skill
description: Use this skill for parsing data files, executing local python calculations, generating charts, and formatting corporate dashboard summaries for Slack.
---

# data-skill

## Overview

This skill equips the agent with a rigid, secure workflow for processing user data files, writing sandboxed visualization code, and reporting metrics directly to external channels using the Model Context Protocol (MCP).

## Instructions

### 1. Ingest Data Context

Use the `CSV_reading_tool` to load the contents of the target dataset file. 
- Always examine the headers, data types, and first few rows to understand the structure before writing code.
- If the file is missing or corrupted, fail early and inform the user.

### 2. Plan and Write Analysis Code

Formulate a complete, clean Python script designed to compute the user's requested metrics or charts.
- Utilize robust libraries like `pandas`, `matplotlib`, or `seaborn`.
- **CRITICAL:** Every script that generates a visualization must explicitly save the file directly to the active working directory using: `plt.savefig('./YOUR_PLOT_NAME.png', dpi=300, bbox_inches='tight')`.
- Ensure all plots include appropriate labels, titles, and legends for high readability.

### 3. Execute and Validate

Pass your full script string into the `Execute_code_sandbox` tool.
- Review the returned `STDOUT` to extract the calculated numerical insights.
- Review the `STDERR` output. If a runtime or syntax error occurs, debug the code immediately and run it again. Do not proceed to reporting if the script failed.

### 4. Synthesize and Report to Slack

Consolidate your mathematical findings into a clean, executive summary.
- Format the message with bold section headers, clear data highlights, and bulleted key insights.
- Call the `Slack_message_tool`, passing the target channel, your text summary, and the exact filename of the generated `.png` plot (if applicable).
- Confirm the deployment to the user in the final CLI summary string.