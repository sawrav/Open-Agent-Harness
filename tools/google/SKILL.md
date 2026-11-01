---
name: google_docs
description: Guidelines for reading and modifying Google Docs using OAuth authorization
---

# Google Docs Tools

## Overview
Tools for reading and modifying Google Docs via the Google Docs API with OAuth2 authorization.

## Available Tools

### read_google_doc
Read content from a Google Doc.

**Parameters:**
- `document_id` (required): Google Doc ID from URL `docs.google.com/document/d/{ID}/edit`
- `authorization_token` (required): OAuth2 authorization token with Google Docs read permissions

**Returns:** Document text content (capped at 3000 characters)

### update_google_doc
Update content in a Google Doc by appending text.

**Parameters:**
- `document_id` (required): Google Doc ID
- `authorization_token` (required): OAuth2 authorization token with Google Docs write permissions
- `content` (required): Content to append
- `encoding` (optional): `utf-8` or `base64` (use base64 for code/special characters)

**Returns:** Success confirmation message

## Usage Guidelines

1. **Always require authorization token**: Never proceed without OAuth token
2. **Extract document ID**: From URL format `docs.google.com/document/d/{ID}/edit`
3. **Use base64 for code**: When content contains quotes, backslashes, or code, use base64 encoding
4. **Handle errors**: Check for HTTP errors and invalid tokens
5. **Respect limits**: Read operations capped at 3000 characters

## Example Workflow

1. User provides document ID and authorization token
2. Call `read_google_doc` to fetch current content
3. Process content as needed
4. Call `update_google_doc` with modified content if required

## Security Notes

- Authorization tokens should be treated as sensitive credentials
- Never log or expose tokens in responses
- Tokens may expire - handle authentication errors gracefully
