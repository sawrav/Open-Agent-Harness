## Google Docs Tools

New Google Docs integration added under `tools/google/`:

### Files created:
- `tools/google/__init__.py` - Package initialization
- `tools/google/docs.py` - Google Docs read/write tools with OAuth support
- `tools/google/SKILL.md` - Skill guidelines for Google Docs usage

### Tools implemented:
1. **read_google_doc** - Read content from Google Doc using OAuth token
2. **update_google_doc** - Append content to Google Doc using OAuth token

### Features:
- OAuth2 authorization token support
- Google Docs API integration (docs.googleapis.com)
- Read and write operations with proper error handling
- Base64 encoding support for special characters
- Response capped at 3000 characters
- Async httpx client usage consistent with existing tools
- Skill documentation with usage guidelines

### Usage:
Tools are auto-discovered by registry and available immediately. Requires OAuth token with Google Docs API permissions.
