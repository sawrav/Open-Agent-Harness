---
name: summarize
description: Instructions for summarizing files, documents, and web pages
---

## Summarize Skill

When the user asks to summarize content:

- If summarizing a file, always read it first using the `read_file` tool.
- If summarizing a URL, fetch it with the `fetch_url` tool before summarizing.
- Lead with a single-sentence TL;DR.
- Follow with 3–5 bullet points covering the key ideas.
- If the content contains action items or decisions, list them separately at the end.
- Keep the summary concise — aim for clarity over completeness.
- Do not reproduce large verbatim sections; paraphrase instead.
